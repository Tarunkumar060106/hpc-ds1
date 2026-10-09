#!/usr/bin/env python3
"""
Phase 3 v5: batched training, size sweep, calibration, closed-loop evaluation.

What changed vs v4 (per-sample loop, 900 isolated configs):
  - Vectorized mini-batch training (batch 256, warmup + cosine) instead of a
    Python loop over single samples — epochs take seconds, not ~80s.
  - Trains on the rollout dataset (17.6k states, 5 hardware profiles).
  - Invalid actions are masked in the loss and at inference.
  - The bottleneck head is actually trained now (it had no loss before), the
    stop head gets a real BCE target, and the old "confidence head" (an MSE
    regression onto improvement, which is not a confidence) becomes an
    explicit expected-improvement head.
  - Confidence = temperature-scaled probability of the chosen action, with
    the temperature fitted on the validation set; ECE reported before/after.
  - Size sweep (small/medium/large), best checkpoint picked on val loss.
  - Evaluation mirrors how Jev is used: accuracy-vs-automation curve by
    confidence threshold, one-step regret, and full closed-loop tuning
    sessions against expert rules, random search and an exhaustive oracle.
"""

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hpc_ds1.api import HPCDS1  # noqa: E402
from hpc_ds1.features import ACTIONS, BOTTLENECKS, FEATURE_VERSION, STOP_INDEX, featurize  # noqa: E402
from hpc_ds1.net import HPCDS1Net  # noqa: E402
from hpc_ds1.tuner import (heuristic_policy, make_oracle_policy, make_random_policy,  # noqa: E402
                           run_tuning)
from hpc_ds1.workloads import (HARDWARE_NAMES, HARDWARE_PROFILES, WORKLOAD_NAMES, WORKLOADS,  # noqa: E402
                               measure, sample_scenario)

SIZES = {
    "small": dict(dim=256, encoder_blocks=1, fusion_layers=2, heads=4, ffn_dim=1024),
    "medium": dict(dim=512, encoder_blocks=2, fusion_layers=3, heads=8, ffn_dim=2048),
    "large": dict(dim=768, encoder_blocks=3, fusion_layers=4, heads=12, ffn_dim=3072),
}
LR = {"small": 1e-3, "medium": 5e-4, "large": 3e-4}
DATA_DIR = PROJECT_ROOT / "data" / "phase2_v3_results"
THRESHOLDS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]


def log(msg=""):
    print(msg, flush=True)


def load_split(name):
    with open(DATA_DIR / f"{name}.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_tensors(rows, device):
    feats = featurize([r["state"] for r in rows], device=device)
    y = torch.tensor([ACTIONS.index(r["best_action"]) for r in rows], device=device)
    return {
        "feats": feats,
        "y": y,
        "stop": (y == STOP_INDEX).float(),
        "bn": torch.tensor([BOTTLENECKS.index(r["bottleneck"]) for r in rows], device=device),
        "imp": torch.tensor([r["best_improvement"] for r in rows], dtype=torch.float32, device=device),
    }


def take(batch, idx):
    return {
        "feats": {k: v[idx] for k, v in batch["feats"].items()},
        "y": batch["y"][idx], "stop": batch["stop"][idx], "bn": batch["bn"][idx], "imp": batch["imp"][idx],
    }


def losses(out, b, class_weights):
    action = F.cross_entropy(out["logits"], b["y"], weight=class_weights)
    stop = F.binary_cross_entropy_with_logits(out["stop_logit"], b["stop"])
    bottleneck = F.cross_entropy(out["bottleneck_logits"], b["bn"])
    improvement = F.smooth_l1_loss(out["improvement"], b["imp"], beta=0.05)
    total = action + 0.5 * stop + 0.3 * bottleneck + 2.0 * improvement
    return total, {"action": action.item(), "stop": stop.item(), "bottleneck": bottleneck.item(),
                   "improvement": improvement.item()}


@torch.no_grad()
def evaluate_split(net, b):
    net.eval()
    out = net(b["feats"])
    ce = F.cross_entropy(out["logits"], b["y"]).item()
    acc = (out["logits"].argmax(1) == b["y"]).float().mean().item()
    return ce, acc, out


def train_one(size, train, val, device, epochs, seed, batch_size=256, patience=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = HPCDS1Net(**SIZES[size]).to(device)

    counts = torch.bincount(train["y"], minlength=len(ACTIONS)).float()
    n = counts.sum()
    cw = torch.where(counts > 0, (n / (len(ACTIONS) * counts.clamp(min=1))) ** 0.5,
                     torch.ones_like(counts)).clamp(0.5, 5.0)

    opt = torch.optim.AdamW(net.parameters(), lr=LR[size], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train["y"]) / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup = min(300, total_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, best = [], {"val_loss": float("inf")}
    bad_epochs, t0 = 0, time.time()
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(train["y"]), device=device)
        tl, correct = 0.0, 0
        for i in range(0, len(perm), batch_size):
            b = take(train, perm[i:i + batch_size])
            out = net(b["feats"])
            loss, _ = losses(out, b, cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item() * len(b["y"])
            correct += (out["logits"].argmax(1) == b["y"]).sum().item()

        val_loss, val_acc, _ = evaluate_split(net, val)
        history.append({"epoch": epoch + 1, "train_loss": round(tl / len(train["y"]), 5),
                        "train_acc": round(correct / len(train["y"]), 5),
                        "val_loss": round(val_loss, 5), "val_acc": round(val_acc, 5),
                        "lr": round(sched.get_last_lr()[0], 7)})
        if val_loss < best["val_loss"] - 1e-4:
            best = {"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch + 1,
                    "state": {k: v.detach().clone() for k, v in net.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"  [{size}] epoch {epoch + 1:3d}  train {tl / len(train['y']):.4f}/{correct / len(train['y']):.3f}"
                f"  val {val_loss:.4f}/{val_acc:.3f}")
        if bad_epochs >= patience:
            log(f"  [{size}] early stop at epoch {epoch + 1} (best epoch {best['epoch']})")
            break

    net.load_state_dict(best["state"])
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    return net, {
        "size": size, "hparams": SIZES[size], "parameters": net.count_parameters(),
        "best_epoch": best["epoch"], "val_loss": round(best["val_loss"], 5),
        "val_acc": round(best["val_acc"], 5), "train_time_sec": round(time.time() - t0, 1),
        "peak_vram_mb": round(peak, 1), "history": history,
    }


# ----------------------------------------------------------------------------
# Calibration + metrics
# ----------------------------------------------------------------------------

@torch.no_grad()
def fit_temperature(logits, y):
    grid = torch.exp(torch.linspace(math.log(0.25), math.log(6.0), 120, device=logits.device))
    nll = torch.stack([F.cross_entropy(logits / t, y) for t in grid])
    return grid[nll.argmin()].item(), nll.min().item()


def reliability(conf, correct, bins=10):
    edges = torch.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc = correct[m].float().mean().item()
            c = conf[m].mean().item()
            ece += m.float().mean().item() * abs(acc - c)
            table.append({"lo": round(lo.item(), 2), "hi": round(hi.item(), 2),
                          "count": int(m.sum()), "accuracy": round(acc, 4), "confidence": round(c, 4)})
    return round(ece, 4), table


def one_step_regret(rows, choices):
    regrets, worse, near_opt = [], 0, 0
    for r, choice in zip(rows, choices):
        base = r["baseline_runtime"]
        cands = r["candidates"]
        rt = base if choice == "STOP_TUNING" else cands.get(choice, base)
        opt = min([base] + list(cands.values()))
        regrets.append(rt / opt - 1.0)
        worse += rt > base * 1.0001
        near_opt += rt <= opt * 1.01
    n = len(rows)
    return {"mean_regret_pct": round(100 * float(np.mean(regrets)), 2),
            "median_regret_pct": round(100 * float(np.median(regrets)), 3),
            "within_1pct_of_best_pct": round(100 * near_opt / n, 2),
            "made_things_worse_pct": round(100 * worse / n, 2)}


def geo_mean(xs):
    return float(math.exp(np.mean(np.log(np.maximum(xs, 1e-12)))))


def closed_loop(model, episodes, seed=2024, max_steps=8):
    rng = random.Random(seed)
    scenarios = [sample_scenario(rng) for _ in range(episodes)]
    auto_policy = model.policy(threshold=0.0)

    def expert_review(workload, problem_size, hw):
        def approve(action, confidence, state):
            from hpc_ds1.workloads import apply_action
            before = state["telemetry"]["runtime_sec"]
            after, _ = measure(workload, problem_size, apply_action(state["current_config"], action, workload), hw)
            return after["runtime_sec"] < before
        return approve

    names = ["HPC-DS1 (autonomous)", "HPC-DS1 + escalation @0.85", "Expert heuristic",
             "Random search (8 tries)", "Exhaustive oracle"]
    res = {n: {"speedups": [], "experiments": [], "escalations": [], "vs_oracle": []} for n in names}
    examples = []
    for i, (wl, hw, cfg, ps) in enumerate(scenarios):
        runs = {
            names[0]: run_tuning(auto_policy, wl, ps, hw, cfg, max_steps=max_steps),
            names[1]: run_tuning(model.policy(threshold=0.85), wl, ps, hw, cfg, max_steps=max_steps,
                                 threshold=0.85, on_escalate=expert_review(wl, ps, hw)),
            names[2]: run_tuning(heuristic_policy, wl, ps, hw, cfg, max_steps=max_steps),
            names[3]: run_tuning(make_random_policy(seed + i), wl, ps, hw, cfg, max_steps=max_steps,
                                 stop_on_reject=False),
            names[4]: run_tuning(make_oracle_policy(wl, ps, hw), wl, ps, hw, cfg, max_steps=max_steps),
        }
        oracle = runs[names[4]].speedup
        for n, r in runs.items():
            res[n]["speedups"].append(r.speedup)
            res[n]["experiments"].append(r.experiments)
            res[n]["escalations"].append(r.escalations)
            res[n]["vs_oracle"].append(r.speedup / oracle)
        if i < 6:
            examples.append({"workload": wl, "hardware": hw, "problem_size": ps,
                             "hpc_ds1": runs[names[0]].to_dict(), "oracle_speedup": round(oracle, 3),
                             "heuristic_speedup": round(runs[names[2]].speedup, 3)})

    summary = []
    for n in names:
        r = res[n]
        summary.append({
            "policy": n,
            "geo_mean_speedup": round(geo_mean(r["speedups"]), 3),
            "median_speedup": round(float(np.median(r["speedups"])), 3),
            "pct_of_oracle": round(100 * geo_mean(r["vs_oracle"]), 1),
            "mean_experiments": round(float(np.mean(r["experiments"])), 2),
            "mean_escalations": round(float(np.mean(r["escalations"])), 2),
            "sessions_reaching_95pct_oracle": round(100 * float(np.mean(np.array(r["vs_oracle"]) >= 0.95)), 1),
        })
    return {"episodes": episodes, "max_steps": max_steps, "summary": summary, "examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="small,medium,large")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=300)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    rows = {s: load_split(s) for s in ("train", "val", "test")}
    data = {s: to_tensors(rows[s], device) for s in rows}
    log(f"states: train {len(rows['train'])}  val {len(rows['val'])}  test {len(rows['test'])}")

    # ---------------- size sweep ----------------
    sweep, nets = [], {}
    for size in args.sizes.split(","):
        log(f"\n== training {size} ({SIZES[size]}) ==")
        net, info = train_one(size, data["train"], data["val"], device, args.epochs, seed=0)
        _, test_acc, _ = evaluate_split(net, data["test"])
        info["test_acc"] = round(test_acc, 5)
        log(f"  [{size}] params {info['parameters']:,}  val {info['val_loss']:.4f}/{info['val_acc']:.3f}"
            f"  test acc {test_acc:.3f}  {info['train_time_sec']}s  peak {info['peak_vram_mb']} MB")
        sweep.append(info)
        nets[size] = net

    chosen = min(sweep, key=lambda s: s["val_loss"])
    net = nets[chosen["size"]]
    log(f"\nselected: {chosen['size']} (lowest val loss {chosen['val_loss']})")

    # ---------------- calibration ----------------
    _, _, val_out = evaluate_split(net, data["val"])
    temp, _ = fit_temperature(val_out["logits"], data["val"]["y"])
    _, test_acc, test_out = evaluate_split(net, data["test"])
    y = data["test"]["y"]
    raw_probs = F.softmax(test_out["logits"], dim=1)
    ece_before, _ = reliability(raw_probs.max(1).values.cpu(), (raw_probs.argmax(1) == y).cpu())
    net.temperature.fill_(temp)
    probs = F.softmax(test_out["logits"] / temp, dim=1)
    conf, pred = probs.max(1)
    correct = pred == y
    ece_after, rel_table = reliability(conf.cpu(), correct.cpu())
    log(f"temperature {temp:.3f}  ECE {ece_before:.4f} -> {ece_after:.4f}")

    # ---------------- metrics ----------------
    test_rows = rows["test"]
    pred_names = [ACTIONS[i] for i in pred.tolist()]
    top2 = (test_out["logits"].topk(2, dim=1).indices == y.unsqueeze(1)).any(1).float().mean().item()

    escalation_curve = []
    for t in THRESHOLDS:
        m = conf >= t
        escalation_curve.append({
            "threshold": t, "coverage": round(m.float().mean().item(), 4),
            "auto_accuracy": round(correct[m].float().mean().item(), 4) if m.any() else None,
            "escalated": int((~m).sum()),
        })

    def group_acc(key):
        g = defaultdict(lambda: [0, 0])
        for r, c in zip(test_rows, correct.tolist()):
            g[r[key]][0] += c
            g[r[key]][1] += 1
        return {k: {"correct": v[0], "total": v[1], "accuracy": round(v[0] / v[1], 4)} for k, v in sorted(g.items())}

    confusion = [[0] * len(ACTIONS) for _ in ACTIONS]
    for t_, p_ in zip(y.tolist(), pred.tolist()):
        confusion[t_][p_] += 1
    per_action = {}
    for i, a in enumerate(ACTIONS):
        tp = confusion[i][i]
        support = sum(confusion[i])
        predicted = sum(row[i] for row in confusion)
        if support or predicted:
            per_action[a] = {"support": support, "recall": round(tp / support, 4) if support else None,
                             "precision": round(tp / predicted, 4) if predicted else None}

    stop_pred = torch.sigmoid(test_out["stop_logit"]) > 0.5
    stop_acc = (stop_pred == (y == STOP_INDEX)).float().mean().item()
    bn_acc = (test_out["bottleneck_logits"].argmax(1) == data["test"]["bn"]).float().mean().item()
    imp_mae = (test_out["improvement"].clamp(0, 1) - data["test"]["imp"]).abs().mean().item()

    heur_choices = [heuristic_policy(r["state"])[0] for r in test_rows]
    heur_acc = float(np.mean([h == r["best_action"] for h, r in zip(heur_choices, test_rows)]))
    majority = Counter(r["best_action"] for r in rows["train"]).most_common(1)[0][0]
    majority_acc = float(np.mean([r["best_action"] == majority for r in test_rows]))

    metrics = {
        "test_states": len(test_rows),
        "accuracy": round(test_acc, 4), "top2_accuracy": round(top2, 4),
        "majority_class_accuracy": round(majority_acc, 4), "majority_class": majority,
        "heuristic_accuracy": round(heur_acc, 4),
        "temperature": round(temp, 4), "ece_before": ece_before, "ece_after": ece_after,
        "reliability": rel_table, "escalation_curve": escalation_curve,
        "per_workload": group_acc("workload"), "per_hardware": group_acc("hardware"),
        "per_action": per_action, "confusion": confusion,
        "stop_head_accuracy": round(stop_acc, 4), "bottleneck_accuracy": round(bn_acc, 4),
        "improvement_mae": round(imp_mae, 4),
        "regret_hpc_ds1": one_step_regret(test_rows, pred_names),
        "regret_heuristic": one_step_regret(test_rows, heur_choices),
    }
    log(f"\ntest acc {test_acc:.4f}  top2 {top2:.4f}  heuristic {heur_acc:.4f}  majority {majority_acc:.4f}")
    log(f"stop head {stop_acc:.4f}  bottleneck {bn_acc:.4f}  improvement MAE {imp_mae:.4f}")
    log(f"regret: model {metrics['regret_hpc_ds1']}  heuristic {metrics['regret_heuristic']}")
    for e in escalation_curve:
        log(f"  threshold {e['threshold']:.2f}: auto {100 * e['coverage']:5.1f}%  acc {e['auto_accuracy']}")

    # ---------------- save checkpoint ----------------
    ckpt_path = PROJECT_ROOT / "model" / "hpc_ds1_v5.pt"
    meta = {"size": chosen["size"], "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "test_metrics": {k: metrics[k] for k in ("accuracy", "top2_accuracy", "ece_after", "temperature")}}
    torch.save({"hparams": net.hparams, "state_dict": net.state_dict(), "feature_version": FEATURE_VERSION,
                "meta": meta}, ckpt_path)
    log(f"saved {ckpt_path}")

    model = HPCDS1.load(ckpt_path, device=device)

    # ---------------- closed loop ----------------
    log(f"\n== closed-loop tuning: {args.episodes} fresh scenarios ==")
    loop = closed_loop(model, args.episodes)
    for s in loop["summary"]:
        log(f"  {s['policy']:28s} speedup x{s['geo_mean_speedup']:<7} {s['pct_of_oracle']:5.1f}% of oracle"
            f"  experiments {s['mean_experiments']:5.2f}  escalations {s['mean_escalations']:.2f}")

    # ---------------- visualizer export ----------------
    picks, per_wl = [], defaultdict(list)
    for i, r in enumerate(test_rows):
        per_wl[r["workload"]].append(i)
    rnd = random.Random(5)
    for wl in WORKLOAD_NAMES:
        idx = per_wl[wl]
        confident = [i for i in idx if correct[i] and conf[i] >= 0.85]
        unsure = [i for i in idx if conf[i] < 0.85]
        wrong = [i for i in idx if not correct[i]]
        for pool in (confident, unsure, wrong):
            if pool:
                picks.append(rnd.choice(pool))
    traces = []
    for i in picks:
        r = test_rows[i]
        t = model.trace(r["state"])
        traces.append({"id": f"{r['workload']}-{i}", "workload": r["workload"], "hardware": r["hardware"],
                       "label": r["best_action"], "correct": t["decision"]["choice"] == r["best_action"],
                       "candidates": r["candidates"], "baseline_runtime": r["baseline_runtime"],
                       "state": r["state"], **t})

    export = {
        "model": {"name": "HPC-DS1", "version": "v5", "size": chosen["size"], "parameters": net.count_parameters(),
                  "hparams": net.hparams, "temperature": round(temp, 4), "device": str(device),
                  "trained_at": meta["trained_at"], "peak_vram_mb": chosen["peak_vram_mb"],
                  "train_time_sec": chosen["train_time_sec"]},
        "catalog": net.layer_catalog(),
        "actions": ACTIONS, "bottlenecks": BOTTLENECKS,
        "workloads": {w: {"display_name": WORKLOADS[w]["display_name"], "problem_sizes": WORKLOADS[w]["problem_sizes"]}
                      for w in WORKLOAD_NAMES},
        "hardware_profiles": HARDWARE_PROFILES,
        "dataset": {"train": len(rows["train"]), "val": len(rows["val"]), "test": len(rows["test"]),
                    "label_distribution": dict(Counter(r["best_action"] for r in rows["train"]).most_common())},
        "size_sweep": [{k: v for k, v in s.items() if k != "history"} for s in sweep],
        "training_curve": chosen["history"],
        "metrics": metrics,
        "closed_loop": loop,
        "traces": traces,
    }
    (PROJECT_ROOT / "results").mkdir(exist_ok=True)
    with open(PROJECT_ROOT / "results" / "v5_metrics.json", "w") as f:
        json.dump({k: export[k] for k in ("model", "size_sweep", "metrics", "closed_loop")}, f, indent=2)
    with open(PROJECT_ROOT / "visualizer" / "model_internals.json", "w") as f:
        json.dump(export, f)
    log(f"\nexported {len(traces)} traces -> visualizer/model_internals.json")


if __name__ == "__main__":
    main()
