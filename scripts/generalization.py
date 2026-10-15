#!/usr/bin/env python3
"""
Does HPC-DS1 generalize, or memorize? All on real measurements.

1. Leave-one-kernel-out: train on 5 kernels (every size), test on all
   measured states of the 6th, which the model has never seen any run of.
2. Input ablations on the standard split (held-out sizes): retrain with one
   input group zeroed to see which inputs decisions depend on.

Uses the size picked by scripts/train.py. Results -> results/generalization.json
and the visualizer export.
"""

import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train import evaluate, fit_temperature, geo_mean, is_good, regret, reliability, split, tensors, train_one  # noqa: E402
from hpc_ds1.api import HPCDS1  # noqa: E402
from hpc_ds1.dataset import build_states, build_tables, load_hardware, load_measurements  # noqa: E402
from hpc_ds1.features import WL_FEATURES  # noqa: E402
from hpc_ds1.kernels import WORKLOAD_NAMES  # noqa: E402
from hpc_ds1.space import ACTIONS, make_config  # noqa: E402
from hpc_ds1.tuner import TableRunner, heuristic_policy, run_tuning  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

ABLATIONS = {
    "full model": None,
    "no telemetry": ("telemetry", slice(None)),
    "no current config": ("config", slice(None)),
    "no kernel identity": ("workload", slice(WL_FEATURES.index(f"is_{WORKLOAD_NAMES[0]}"), len(WL_FEATURES))),
}


def log(m=""):
    print(m, flush=True)


def ablate(b, spec):
    if spec:
        b["feats"][spec[0]][:, spec[1]] = 0.0
    return b


def assess(net, te, te_rows, va):
    _, _, vo = evaluate(net, va)
    temp = fit_temperature(vo["logits"], va["good"])
    _, acc, out = evaluate(net, te)
    probs = F.softmax(out["logits"] / temp, 1)
    conf, pred = probs.max(1)
    good = te["good"].gather(1, pred.unsqueeze(1)).squeeze(1)
    ece, _ = reliability(conf.cpu(), good.cpu())
    net.temperature.fill_(temp)
    names = [ACTIONS[i] for i in pred.tolist()]
    heur = [heuristic_policy(r["state"])[0] for r in te_rows]
    auto = conf >= 0.85
    return {"test_states": len(te_rows), "accuracy": round(acc, 4),
            "rules_accuracy": round(float(np.mean([is_good(r, h) for h, r in zip(heur, te_rows)])), 4),
            "ece": ece, "auto_run_at_085": round(auto.float().mean().item(), 4),
            "auto_run_accuracy": round(good[auto].float().mean().item(), 4) if auto.any() else None,
            "regret": regret(te_rows, names), "regret_rules": regret(te_rows, heur)}


def replay(model, tables, hardware, insts):
    m_vals, h_vals = [], []
    for inst in insts:
        table = tables[inst]
        runner = TableRunner(table, hardware)
        best = min(t["runtime_sec"] for t in table.values())
        for key in table:
            start = make_config(*key)
            m_vals.append(best / run_tuning(model.policy(0.0), inst[0], inst[1], start, runner).final_runtime)
            h_vals.append(best / run_tuning(heuristic_policy, inst[0], inst[1], start, runner).final_runtime)
    return {"sessions": len(m_vals), "model_pct_of_best": round(100 * geo_mean(m_vals), 1),
            "rules_pct_of_best": round(100 * geo_mean(h_vals), 1)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from hpc_ds1.paths import RESULTS_DIR, UI_DIR
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())
    size = metrics["model"]["size"]
    hardware = load_hardware()
    tables, _ = build_tables(load_measurements())
    states = build_states(tables, hardware)
    results = {"model_size": size}

    log(f"=== leave one kernel out ({size} model) ===")
    out = []
    for w in WORKLOAD_NAMES:
        te_rows = [s for s in states if s["workload"] == w]
        rest = [s for s in states if s["workload"] != w]
        random.Random(1).shuffle(rest)
        n_val = int(len(rest) * 0.15)
        tr, va, te = tensors(rest[n_val:], device), tensors(rest[:n_val], device), tensors(te_rows, device)
        net, _ = train_one(size, tr, va, device)
        r = assess(net, te, te_rows, va)
        r["held_out"] = w
        r["replay"] = replay(HPCDS1(net, device), tables, hardware, sorted({(s["workload"], s["size"]) for s in te_rows}))
        log(f"  {w:10s} acc {r['accuracy']:.3f} (rules {r['rules_accuracy']:.3f})  ECE {r['ece']:.3f}  "
            f"regret {r['regret']['mean_regret_pct']}%  tuning {r['replay']['model_pct_of_best']}% of best "
            f"(rules {r['replay']['rules_pct_of_best']}%)")
        out.append(r)
    results["leave_one_kernel_out"] = out

    log("\n=== input ablations (held-out sizes) ===")
    tr_rows, va_rows, te_rows = split(states)
    out = []
    for name, spec in ABLATIONS.items():
        tr, va, te = (ablate(tensors(x, device), spec) for x in (tr_rows, va_rows, te_rows))
        net, _ = train_one(size, tr, va, device)
        r = assess(net, te, te_rows, va)
        r["variant"] = name
        log(f"  {name:20s} acc {r['accuracy']:.3f}  regret {r['regret']['mean_regret_pct']}%  ECE {r['ece']:.3f}")
        out.append(r)
    results["ablations"] = out

    (RESULTS_DIR / "generalization.json").write_text(json.dumps(results, indent=2))
    viz = UI_DIR / "model_internals.json"
    export = json.loads(viz.read_text())
    export["generalization"] = results
    viz.write_text(json.dumps(export))
    log("\nsaved results/generalization.json")


if __name__ == "__main__":
    main()
