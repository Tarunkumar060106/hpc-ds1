#!/usr/bin/env python3
"""
Phase 4: does HPC-DS1 generalize, or memorize?

1. Leave-one-workload-out: train on 5 kernels, test on the 6th (never seen).
2. Leave-one-machine-out: train on 4 hardware profiles, test on the 5th.
3. Input ablations on the standard split: retrain with one input module
   zeroed out (telemetry, config, hardware, workload identity) to see which
   inputs the decisions actually depend on.

Each held-out setting reports accuracy, one-step regret, calibration (ECE)
and closed-loop tuning on fresh scenarios from the held-out group, next to
the expert heuristic on the same states. Same small model and recipe as
phase3_train_v5.py. Results -> results/generalization.json and the
visualizer export.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from phase3_train_v5 import (evaluate_split, fit_temperature, geo_mean, load_split,  # noqa: E402
                             one_step_regret, reliability, to_tensors, train_one)
from hpc_ds1.api import HPCDS1  # noqa: E402
from hpc_ds1.features import ACTIONS, WL_FEATURES  # noqa: E402
from hpc_ds1.tuner import heuristic_policy, make_oracle_policy, run_tuning  # noqa: E402
from hpc_ds1.workloads import HARDWARE_NAMES, WORKLOAD_NAMES, sample_scenario  # noqa: E402

WL_ONEHOT = slice(WL_FEATURES.index(f"is_{WORKLOAD_NAMES[0]}"), len(WL_FEATURES))
ABLATIONS = {
    "full model": None,
    "no telemetry": ("telemetry", slice(None)),
    "no current config": ("config", slice(None)),
    "no hardware": ("hardware", slice(None)),
    "no workload identity": ("workload", WL_ONEHOT),
}


def log(msg=""):
    print(msg, flush=True)


def split_episodes(rows, val_frac, seed):
    eps = sorted({r["episode"] for r in rows})
    random.Random(seed).shuffle(eps)
    val_eps = set(eps[: max(1, int(len(eps) * val_frac))])
    return [r for r in rows if r["episode"] not in val_eps], [r for r in rows if r["episode"] in val_eps]


def ablate(batch, spec):
    if spec:
        module, cols = spec
        batch["feats"][module][:, cols] = 0.0
    return batch


def evaluate(net, test_b, test_rows, val_b):
    _, _, val_out = evaluate_split(net, val_b)
    temp, _ = fit_temperature(val_out["logits"], val_b["y"])
    _, acc, out = evaluate_split(net, test_b)
    probs = F.softmax(out["logits"] / temp, dim=1)
    conf, pred = probs.max(1)
    ece, _ = reliability(conf.cpu(), (pred == test_b["y"]).cpu())
    net.temperature.fill_(temp)
    choices = [ACTIONS[i] for i in pred.tolist()]
    heur = [heuristic_policy(r["state"])[0] for r in test_rows]
    auto = conf >= 0.85
    return {
        "test_states": len(test_rows),
        "accuracy": round(acc, 4),
        "heuristic_accuracy": round(float(np.mean([h == r["best_action"] for h, r in zip(heur, test_rows)])), 4),
        "ece": ece,
        "auto_run_at_085": round(auto.float().mean().item(), 4),
        "auto_run_accuracy": round((pred == test_b["y"])[auto].float().mean().item(), 4) if auto.any() else None,
        "regret": one_step_regret(test_rows, choices),
        "regret_heuristic": one_step_regret(test_rows, heur),
    }


def closed_loop_subset(model, key, value, episodes, seed=99):
    rng = random.Random(seed)
    scen = []
    while len(scen) < episodes:
        s = sample_scenario(rng)
        if (key == "workload" and s[0] == value) or (key == "hardware" and s[1] == value):
            scen.append(s)
    pol = model.policy(threshold=0.0)
    m_vs, h_vs, m_exp, o_exp = [], [], [], []
    for wl, hw, cfg, ps in scen:
        o = run_tuning(make_oracle_policy(wl, ps, hw), wl, ps, hw, cfg)
        m = run_tuning(pol, wl, ps, hw, cfg)
        h = run_tuning(heuristic_policy, wl, ps, hw, cfg)
        m_vs.append(m.speedup / o.speedup)
        h_vs.append(h.speedup / o.speedup)
        m_exp.append(m.experiments)
        o_exp.append(o.experiments)
    return {
        "episodes": episodes,
        "model_pct_of_oracle": round(100 * geo_mean(m_vs), 1),
        "heuristic_pct_of_oracle": round(100 * geo_mean(h_vs), 1),
        "model_sessions_reaching_95pct": round(100 * float(np.mean(np.array(m_vs) >= 0.95)), 1),
        "model_experiments": round(float(np.mean(m_exp)), 2),
        "oracle_experiments": round(float(np.mean(o_exp)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--only", choices=["workload", "hardware", "ablation"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = {s: load_split(s) for s in ("train", "val", "test")}
    all_rows = splits["train"] + splits["val"] + splits["test"]
    log(f"device {device}; {len(all_rows)} states")
    t_start = time.time()
    results = {"recipe": {"size": "small", "epochs": args.epochs, "closed_loop_episodes": args.episodes}}

    for key, values in (("workload", WORKLOAD_NAMES), ("hardware", HARDWARE_NAMES)):
        if args.only and args.only != key:
            continue
        log(f"\n=== leave one {key} out ===")
        out = []
        for v in values:
            test_rows = [r for r in all_rows if r[key] == v]
            train_rows, val_rows = split_episodes([r for r in all_rows if r[key] != v], 0.1, seed=1)
            tr, va, te = (to_tensors(x, device) for x in (train_rows, val_rows, test_rows))
            net, info = train_one("small", tr, va, device, args.epochs, seed=0)
            res = evaluate(net, te, test_rows, va)
            res.update({"held_out": v, "train_states": len(train_rows), "train_time_sec": info["train_time_sec"],
                        "label_mix": dict(Counter(r["best_action"] for r in test_rows).most_common(4))})
            res["closed_loop"] = closed_loop_subset(HPCDS1(net, device), key, v, args.episodes)
            log(f"  held out {v:20s} acc {res['accuracy']:.3f} (heuristic {res['heuristic_accuracy']:.3f})"
                f"  ECE {res['ece']:.3f}  regret {res['regret']['mean_regret_pct']}%"
                f"  closed loop {res['closed_loop']['model_pct_of_oracle']}% of oracle"
                f" (heuristic {res['closed_loop']['heuristic_pct_of_oracle']}%)")
            out.append(res)
        results[f"leave_one_{key}_out"] = out

    if not args.only or args.only == "ablation":
        log("\n=== input ablations (standard split) ===")
        out = []
        for name, spec in ABLATIONS.items():
            tr, va, te = (ablate(to_tensors(splits[s], device), spec) for s in ("train", "val", "test"))
            net, info = train_one("small", tr, va, device, args.epochs, seed=0)
            res = evaluate(net, te, splits["test"], va)
            res.update({"variant": name, "train_time_sec": info["train_time_sec"]})
            log(f"  {name:22s} acc {res['accuracy']:.3f}  ECE {res['ece']:.3f}"
                f"  regret {res['regret']['mean_regret_pct']}%  auto@0.85 {res['auto_run_at_085']:.3f}")
            out.append(res)
        results["ablations"] = out

    results["wall_time_sec"] = round(time.time() - t_start, 1)
    path = PROJECT_ROOT / "results" / "generalization.json"
    if args.only and path.exists():
        prev = json.loads(path.read_text())
        prev.update(results)
        results = prev
    path.write_text(json.dumps(results, indent=2))

    viz = PROJECT_ROOT / "visualizer" / "model_internals.json"
    export = json.loads(viz.read_text())
    export["generalization"] = results
    viz.write_text(json.dumps(export))
    log(f"\nsaved {path} and updated {viz.name} ({results['wall_time_sec']}s)")


if __name__ == "__main__":
    main()
