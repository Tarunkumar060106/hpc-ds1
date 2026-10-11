#!/usr/bin/env python3
"""
Which signals tell when System 1's instant pick is right?

Uses the out-of-fold estimates (scripts/system1_oof.py): for each of the 72
measured problems, the pick of a System 1 that never saw that problem, scored
against the measurements (right = within 5% of the best config). Each candidate
signal is scored by AUROC (0.5 = chance), and a leave-one-out logistic
combination is scored the same way, so nothing is tuned on the problem it is
evaluated on.

    python scripts/system1_confidence_study.py  -> results/experiments/system1_confidence_signal_study.json
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hpc_ds1.dataset import build_tables, load_measurements  # noqa: E402


def parse(k):
    r, t, g, p = k.split(",")
    return (int(r), int(t), g in ("1", "True"), p)


def auroc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if not len(pos) or not len(neg):
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def logistic_loo(X, y, l2=1.0, steps=3000, lr=0.1):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    out = np.zeros(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        w, b = np.zeros(X.shape[1]), 0.0
        for _ in range(steps):
            z = 1 / (1 + np.exp(-(X[m] @ w + b)))
            g = z - y[m]
            w -= lr * (X[m].T @ g / m.sum() + l2 * w / m.sum())
            b -= lr * g.mean()
        out[i] = 1 / (1 + np.exp(-(X[i] @ w + b)))
    return out


def main():
    oof = json.loads((PROJECT_ROOT / "results" / "system1_oof.json").read_text())
    tables, _ = build_tables(load_measurements())
    rows = []
    for prob, est in oof.items():
        w, s = prob.split(":")
        table = tables[(w, int(s))]
        rt = {parse(k): v for k, v in est["rt"].items()}
        fail = {parse(k): v for k, v in est["fail"].items()}
        usable = {k: v for k, v in rt.items() if fail[k] < 0.5} or rt
        order = sorted(usable, key=usable.get)
        pick = order[0]
        best = min(t["runtime_sec"] for t in table.values())
        got = table.get(pick)
        good = got is not None and got["runtime_sec"] <= best * 1.05
        other = [k for k in order if k[2] != pick[2]]
        rows.append({"workload": w, "size": int(s), "good": int(good),
                     "margin": math.log10(usable[order[1]] / usable[pick]) if len(order) > 1 else 1.0,
                     "device_margin": math.log10(usable[other[0]] / usable[pick]) if other else 1.0,
                     "log_runtime": math.log10(usable[pick]),
                     "pick_fail_prob": fail[pick],
                     "n_near": sum(usable[k] <= usable[pick] * 1.05 for k in order)})
    # leave-one-out "how often was System 1 right on this workload before"
    for r in rows:
        others = [o["good"] for o in rows if o["workload"] == r["workload"] and o is not r]
        r["workload_track_record"] = float(np.mean(others)) if others else 0.5
    y = np.array([r["good"] for r in rows], float)
    signals = ["margin", "device_margin", "log_runtime", "pick_fail_prob", "n_near", "workload_track_record"]
    res = {"problems": len(rows), "right_share": round(float(y.mean()), 4), "signals": {}}
    for sname in signals:
        x = np.array([r[sname] for r in rows], float)
        a = auroc(x, y)
        res["signals"][sname] = round(max(a, 1 - a), 3) if a is not None else None  # direction-free
        print(f"  {sname:22s} AUROC {res['signals'][sname]}")
    X = np.array([[r[s] for s in signals] for r in rows], float)
    combo = logistic_loo(X, y)
    res["combined_leave_one_out_auroc"] = round(auroc(combo, y), 3)
    print(f"right picks: {int(y.sum())}/{len(y)}; combined (leave-one-out logistic) AUROC {res['combined_leave_one_out_auroc']}")
    (PROJECT_ROOT / "results" / "experiments" / "system1_confidence_signal_study.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
