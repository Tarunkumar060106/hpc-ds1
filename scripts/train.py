#!/usr/bin/env python3
"""
Train and evaluate HPC-DS1 on real measurements from this machine.

  data:     data/real/measurements.jsonl (scripts/collect.py)
  split:    one problem size per kernel (Kernel.test_size) is held out entirely as the
            test set (the model never sees any run of it); the other sizes
            are split 85/15 into train/validation by state
  model:    size picked on validation loss from a small sweep
  calib.:   temperature scaling fitted on validation
  eval:     decision accuracy and regret against measured runtimes; full
            tuning sessions replayed over the measured test problems from
            every starting config; and live sessions that run every
            suggested config on the hardware right now
  outputs:  hpc_ds1/models/hpc_ds1.pt, hpc_ds1/results/metrics.json, hpc_ds1/ui/model_internals.json

    python scripts/train.py            # train + evaluate + live check
    python scripts/train.py --no-live  # skip the live hardware sessions
"""

import argparse
import json
import math
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hpc_ds1.api import HPCDS1  # noqa: E402
from hpc_ds1.dataset import build_states, build_tables, load_hardware, load_measurements  # noqa: E402
from hpc_ds1.features import FEATURE_VERSION, STOP_INDEX, featurize  # noqa: E402
from hpc_ds1.kernels import KERNELS, WORKLOAD_NAMES  # noqa: E402
from hpc_ds1.net import HPCDS1Net  # noqa: E402
from hpc_ds1.space import ACTION_HELP, ACTIONS, make_config  # noqa: E402
from hpc_ds1.tuner import (LiveRunner, TableRunner, heuristic_policy, make_greedy_search_policy,  # noqa: E402
                           make_random_policy, run_tuning)

warnings.filterwarnings("ignore", category=UserWarning)

from hpc_ds1.training import (GOOD_TOL, SIZES, TAU, evaluate, fit_guard, fit_temperature,  # noqa: E402,F401
                             good_rate, is_good, loss_fn, reliability, save_checkpoint, soft_ce,
                             soft_target, split, take, tensors, train_one)

THRESHOLDS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]


def log(msg=""):
    print(msg, flush=True)



def regret(rows, choices):
    """How much slower the best runtime reachable after the chosen action is than
    the best reachable from this state at all (both from measured runtimes).
    STOP keeps the current runtime; a failing config leaves the job where it was."""
    vals, worse, near, failed = [], 0, 0, 0
    for r, a in zip(rows, choices):
        base, best = r["runtime"], r["reachable_from_here"]
        if a == "STOP_TUNING":
            got, nxt = base, base
        elif r["outcomes"].get(a) is None:
            failed += 1
            got, nxt = base, base
        else:
            nxt = r["outcomes"][a]
            got = min(base, r["reachable"][a]) if nxt < base else base
        vals.append(got / best - 1.0)
        worse += nxt > base * 1.0001
        near += got <= best * 1.05
    n = len(rows)
    # Geometric mean: runtime ratios are multiplicative, and an arithmetic mean
    # gets dominated by a single pathological config (e.g. a 50 s bf16 CPU run).
    return {"mean_regret_pct": round(100 * (geo_mean([1 + v for v in vals]) - 1), 2),
            "worst_regret_x": round(1 + max(vals), 2),
            "median_regret_pct": round(100 * float(np.median(vals)), 2),
            "within_5pct_of_best_pct": round(100 * near / n, 1),
            "made_slower_pct": round(100 * worse / n, 1), "picked_failing_config_pct": round(100 * failed / n, 1)}


def geo_mean(xs):
    return float(math.exp(np.mean(np.log(np.maximum(np.asarray(xs, dtype=float), 1e-12)))))


# ----------------------------------------------------------------------------
# closed loop
# ----------------------------------------------------------------------------

def replay_sessions(model, tables, hardware, instances, max_steps=8):
    """Every measured starting config of every test problem, replayed on real measurements."""
    names = ["HPC-DS1 (autonomous)", "Rules of thumb", "Random moves (8)", "Greedy search (tries all neighbours)"]
    res = {n: defaultdict(list) for n in names}
    per_inst = []
    for inst in instances:
        table = tables[inst]
        runner = TableRunner(table, hardware)
        best_rt = min(t["runtime_sec"] for t in table.values())
        inst_vals = defaultdict(list)
        for i, key in enumerate(sorted(table)):
            start = make_config(*key)
            pols = {names[0]: (model.policy(0.0), True), names[1]: (heuristic_policy, True),
                    names[2]: (make_random_policy(1000 + i), False),
                    names[3]: (make_greedy_search_policy(runner), True)}
            for n, (pol, stop_on_reject) in pols.items():
                r = run_tuning(pol, inst[0], inst[1], start, runner, max_steps=max_steps,
                               stop_on_reject=stop_on_reject)
                res[n]["speedup"].append(r.speedup)
                res[n]["of_best"].append(best_rt / r.final_runtime)
                res[n]["experiments"].append(r.experiments)
                inst_vals[n].append(best_rt / r.final_runtime)
        per_inst.append({"workload": inst[0], "size": inst[1], "starts": len(table),
                         **{n: round(100 * geo_mean(v), 1) for n, v in inst_vals.items()}})
    summary = [{"policy": n, "sessions": len(res[n]["speedup"]),
                "geo_mean_speedup": round(geo_mean(res[n]["speedup"]), 3),
                "pct_of_best_config": round(100 * geo_mean(res[n]["of_best"]), 1),
                "sessions_within_10pct_of_best": round(100 * float(np.mean(np.array(res[n]["of_best"]) >= 1 / 1.1)), 1),
                "mean_experiments": round(float(np.mean(res[n]["experiments"])), 2)} for n in names]
    return {"summary": summary, "per_problem": per_inst, "max_steps": max_steps}


def live_sessions(model, tables, instances, seed=3):
    """Run tuning for real: every suggested config is executed on the hardware now."""
    rng = random.Random(seed)
    out = []
    for inst in instances:
        table = tables[inst]
        slowish = sorted(table, key=lambda k: -table[k]["runtime_sec"])[: max(3, len(table) // 3)]
        start = make_config(*rng.choice(slowish))
        t0 = time.time()
        r = model.tune(inst[0], inst[1], start, threshold=0.0)
        wall = time.time() - t0
        # Re-run both the session's final config and the best config from the
        # collection back to back, now, so the comparison isn't skewed by
        # thermal/clock drift since the collection.
        best_key = min(table, key=lambda k: table[k]["runtime_sec"])
        runner = LiveRunner(inst[0], inst[1], model.engine)
        best_now = runner(make_config(*best_key))["runtime_sec"]
        final_now = runner(r.final_config)["runtime_sec"]
        out.append({**r.to_dict(), "best_known_config": make_config(*best_key),
                    "best_known_runtime_now": best_now, "final_runtime_now": final_now,
                    "pct_of_best_measured": round(100 * best_now / final_now, 1), "wall_sec": round(wall, 1)})
        log(f"  live {inst[0]:10s} {inst[1]:>9}: x{r.speedup:6.2f} in {r.experiments} runs; final config re-run "
            f"{final_now * 1000:.2f} ms vs best known {best_now * 1000:.2f} ms ({out[-1]['pct_of_best_measured']}%)")
    return out


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--sizes", default="tiny,small,medium")
    ap.add_argument("--ensemble", type=int, default=5,
                    help="also train an ensemble of this many seeds; kept only if it is better on validation")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = load_hardware()
    rows = load_measurements()
    tables, failed = build_tables(rows)
    states = build_states(tables, hardware)
    tr_rows, va_rows, te_rows = split(states)
    log(f"{len(rows)} real measurements -> {len(states)} states over {len(tables)} problems "
        f"(train {len(tr_rows)}, val {len(va_rows)}, test {len(te_rows)} on held-out sizes)")
    log("label mix: " + ", ".join(f"{k} {v}" for k, v in Counter(s['best_action'] for s in states).most_common()))
    tr, va, te = tensors(tr_rows, device), tensors(va_rows, device), tensors(te_rows, device)

    sweep, nets = [], {}
    for size in args.sizes.split(","):
        net, info = train_one(size, tr, va, device)
        _, test_acc, _ = evaluate(net, te)
        info["test_acc"] = round(test_acc, 4)
        log(f"  {size:6s} {info['parameters']:>9,} params  val {info['val_loss']:.4f}/{info['val_acc']:.3f}"
            f"  test acc {test_acc:.3f}  ({info['epochs_run']} epochs, {info['train_time_sec']}s)")
        sweep.append(info)
        nets[size] = net
    chosen = min(sweep, key=lambda s: s["val_loss"])
    net = nets[chosen["size"]]
    log(f"selected {chosen['size']} on validation loss")

    # Single network vs an ensemble of seeds, decided on validation only (loss and calibration
    # after temperature scaling). The test set plays no part in the choice.
    def val_score(m):
        vl, vacc, vo = evaluate(m, va)
        t = fit_temperature(vo["logits"], va["good"])
        p = F.softmax(vo["logits"] / t, 1)
        c, pr = p.max(1)
        return vl, vacc, reliability(c.cpu(), va["good"].gather(1, pr.unsqueeze(1)).squeeze(1).cpu())[0]
    ensemble_choice = {"members": 1}
    if args.ensemble > 1:
        from hpc_ds1.net import EnsembleNet
        members = [net] + [train_one(chosen["size"], tr, va, device, seed=s)[0] for s in range(1, args.ensemble)]
        ens = EnsembleNet(members).to(device)
        sv, se = val_score(net), val_score(ens)
        log(f"validation  single: loss {sv[0]:.4f} acc {sv[1]:.3f} ECE {sv[2]:.3f}  |  "
            f"ensemble of {args.ensemble}: loss {se[0]:.4f} acc {se[1]:.3f} ECE {se[2]:.3f}")
        if se[2] < sv[2] and se[0] <= sv[0] * 1.01:
            net = ens
            log(f"using the ensemble of {args.ensemble} (better calibrated on validation)")
        else:
            log("keeping the single network (ensemble not better on validation)")
        ensemble_choice = {"members": len(getattr(net, "members", [net])), "candidates": args.ensemble,
                           "val_single": {"loss": round(sv[0], 5), "acc": round(sv[1], 4), "ece": sv[2]},
                           "val_ensemble": {"loss": round(se[0], 5), "acc": round(se[1], 4), "ece": se[2]}}

    _, _, vout = evaluate(net, va)
    temp = fit_temperature(vout["logits"], va["good"])
    _, test_acc, tout = evaluate(net, te)
    y = te["y"]
    raw = F.softmax(tout["logits"], 1)
    exact = (raw.argmax(1) == y).float().mean().item()
    ece_raw, _ = reliability(raw.max(1).values.cpu(), te["good"].gather(1, raw.argmax(1, keepdim=True)).squeeze(1).cpu())
    net.temperature.fill_(temp)
    probs = F.softmax(tout["logits"] / temp, 1)
    conf, pred = probs.max(1)
    correct = te["good"].gather(1, pred.unsqueeze(1)).squeeze(1)
    ece, rel = reliability(conf.cpu(), correct.cpu())
    pred_names = [ACTIONS[i] for i in pred.tolist()]

    esc = []
    for t in THRESHOLDS:
        m = conf >= t
        esc.append({"threshold": t, "coverage": round(m.float().mean().item(), 4),
                    "auto_accuracy": round(correct[m].float().mean().item(), 4) if m.any() else None,
                    "auto_regret_pct": regret([r for r, k in zip(te_rows, m.tolist()) if k],
                                              [a for a, k in zip(pred_names, m.tolist()) if k])["mean_regret_pct"] if m.any() else None})

    def by(key):
        g = defaultdict(lambda: [0, 0])
        for r, c in zip(te_rows, correct.tolist()):
            g[r[key]][0] += c
            g[r[key]][1] += 1
        return {str(k): {"correct": v[0], "total": v[1], "accuracy": round(v[0] / v[1], 4)} for k, v in sorted(g.items())}

    conf_m = [[0] * len(ACTIONS) for _ in ACTIONS]
    for a, b in zip(y.tolist(), pred.tolist()):
        conf_m[a][b] += 1
    heur = [heuristic_policy(r["state"])[0] for r in te_rows]
    majority = Counter(r["best_action"] for r in tr_rows).most_common(1)[0][0]

    metrics = {
        "test_states": len(te_rows), "test_problems": sorted({f"{r['workload']}:{r['size']}" for r in te_rows}),
        "accuracy": round(test_acc, 4), "exact_label_accuracy": round(exact, 4),
        "accuracy_definition": "chosen move can reach within 2% of the best runtime reachable from this state",
        "rules_accuracy": round(float(np.mean([is_good(r, h) for h, r in zip(heur, te_rows)])), 4),
        "majority_class": majority,
        "majority_accuracy": round(float(np.mean([is_good(r, majority) for r in te_rows])), 4),
        "temperature": round(temp, 4), "ece_raw": ece_raw, "ece": ece, "reliability": rel,
        "escalation_curve": esc, "per_workload": by("workload"), "confusion": conf_m,
        "regret_model": regret(te_rows, pred_names), "regret_rules": regret(te_rows, heur),
        "stop_head_accuracy": round(((torch.sigmoid(tout["stop_logit"]) > 0.5) == (y == STOP_INDEX)).float().mean().item(), 4),
        "improvement_mae": round((tout["improvement"].clamp(0, 1) - te["imp"]).abs().mean().item(), 4),
        "ensemble": ensemble_choice,
    }
    log(f"\npicks a best move {metrics['accuracy']:.3f} (exact label {exact:.3f}; rules {metrics['rules_accuracy']:.3f}, "
        f"majority '{majority}' {metrics['majority_accuracy']:.3f})  ECE {ece_raw:.3f} -> {ece:.3f} (T={temp:.2f})")
    log(f"regret: model {metrics['regret_model']}\n        rules {metrics['regret_rules']}")
    for e in esc:
        log(f"  threshold {e['threshold']:.2f}: auto-run {100 * e['coverage']:5.1f}%  accuracy {e['auto_accuracy']}")

    from hpc_ds1.paths import MODELS_DIR, RESULTS_DIR, UI_DIR
    ckpt = MODELS_DIR / "hpc_ds1.pt"
    meta = {"size": chosen["size"], "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "trained_on": hardware["name"],
            "test_metrics": {k: metrics[k] for k in ("accuracy", "ece", "temperature")}}
    guard = fit_guard(tr_rows, va_rows, hardware)
    save_checkpoint(ckpt, net, meta, guard)
    model = HPCDS1.load(ckpt, device=device)
    log(f"saved {ckpt}")

    test_insts = sorted({(r["workload"], r["size"]) for r in te_rows})
    log("\nreplaying tuning sessions on measured test problems ...")
    loop = replay_sessions(model, tables, hardware, test_insts)
    for s in loop["summary"]:
        log(f"  {s['policy']:38s} x{s['geo_mean_speedup']:<7} {s['pct_of_best_config']:5.1f}% of best config"
            f"  {s['mean_experiments']:5.2f} runs")

    live = []
    if not args.no_live:
        log("\nlive tuning sessions on the hardware ...")
        live = live_sessions(model, tables, test_insts)
        model.close()

    picks, rng = [], random.Random(5)
    for w in WORKLOAD_NAMES:
        idx = [i for i, r in enumerate(te_rows) if r["workload"] == w]
        pools = ([i for i in idx if correct[i] and conf[i] >= 0.85], [i for i in idx if conf[i] < 0.85],
                 [i for i in idx if not correct[i]])
        for pool in pools:
            if pool:
                picks.append(rng.choice(pool))
    traces = []
    for i in picks:
        r = te_rows[i]
        t = model.trace(r["state"])
        traces.append({"id": f"{r['workload']}-{r['size']}-{i}", "workload": r["workload"], "size": r["size"],
                       "label": r["best_action"], "correct": is_good(r, t["decision"]["choice"]),
                       "good_moves": [a for a in ACTIONS if is_good(r, a)],
                       "outcomes": r["outcomes"], "reachable": r["reachable"], "runtime": r["runtime"],
                       "instance_best_runtime": r["instance_best_runtime"], "state": r["state"], **t})

    landscapes = {}
    for (w, s), table in tables.items():
        landscapes[f"{w}:{s}"] = [{"config": dict(zip(("ranks", "threads", "gpu", "precision"), k)),
                                   "runtime": v["runtime_sec"]} for k, v in table.items()]
    export = {
        "model": {"name": "HPC-DS1", "version": "1.0", "size": chosen["size"], "parameters": net.count_parameters(),
                  "hparams": net.hparams, "temperature": round(temp, 4), "device": str(device),
                  "trained_at": meta["trained_at"], "train_time_sec": chosen["train_time_sec"]},
        "hardware": hardware,
        "catalog": net.layer_catalog(), "actions": ACTIONS, "action_help": ACTION_HELP,
        "workloads": {w: {"display_name": k.display_name, "character": k.character, "sizes": k.sizes,
                          "size_unit": k.size_unit, "test_size": k.test_size} for w, k in KERNELS.items()},
        "dataset": {"measurements": len(rows), "ok_and_correct": sum(len(t) for t in tables.values()),
                    "failed_or_wrong": sum(len(f) for f in failed.values()),
                    "failures": [{"problem": f"{w}:{s}", "config": dict(zip(("ranks", "threads", "gpu", "precision"), k)),
                                  "reason": why} for (w, s), d in failed.items() for k, why in list(d.items())[:3]][:24],
                    "states": len(states), "train": len(tr_rows), "val": len(va_rows), "test": len(te_rows),
                    "label_distribution": dict(Counter(s["best_action"] for s in states).most_common())},
        "size_sweep": [{k: v for k, v in s.items() if k != "history"} for s in sweep],
        "training_curve": chosen["history"], "metrics": metrics, "replay": loop, "live": live,
        "landscapes": landscapes, "traces": traces,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(
        {k: export[k] for k in ("model", "hardware", "dataset", "size_sweep", "metrics", "replay", "live")}, indent=2))
    (UI_DIR / "model_internals.json").write_text(json.dumps(export))
    log(f"\nexported {len(traces)} traces -> hpc_ds1/ui/model_internals.json")


if __name__ == "__main__":
    main()
