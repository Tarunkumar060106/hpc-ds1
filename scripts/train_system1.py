#!/usr/bin/env python3
"""
Train and evaluate System 1 (hpc_ds1/system1.py) on the real measurements.

  cross-validation  6 folds by problem size: fold k holds out the k-th size of
                    every workload, so each of the 48 problems is answered by
                    an ensemble that never saw any run of it
  calibration       confidence = logistic map from ensemble signals to
                    P(pick within 5% of best), fitted on the out-of-fold picks;
                    its own reliability is checked leave-one-out
  dual process      on the held-out test sizes (never seen by System 1 *or*
                    the step model): System 1 alone vs System 1 -> System 2
                    vs System 2 alone vs rules of thumb, replayed on the
                    measured runtimes
  unseen workload   leave-one-workload-out: does System 1 know a job it never saw?
  outputs           model/system1.pt (trained on every measured problem),
                    results/system1.json, the visualizer export

    python scripts/train_system1.py
"""

import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hpc_ds1.api import HPCDS1  # noqa: E402
from hpc_ds1.dataset import build_tables, load_hardware, load_measurements  # noqa: E402
from hpc_ds1.features import _hw  # noqa: E402
from hpc_ds1.kernels import KERNELS, WORKLOAD_NAMES  # noqa: E402
from hpc_ds1.space import config_key, make_config  # noqa: E402
from hpc_ds1.system1 import (GOOD_WITHIN, Calibrator, System1, confidence_features, pick,  # noqa: E402
                             rows_from_measurements, score_configs, size_ranges, train_ensemble)
from hpc_ds1.training import reliability  # noqa: E402
from hpc_ds1.tuner import TableRunner, heuristic_policy, run_tuning  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
DEFAULT_START = make_config(1, 1, False, "fp32")
THRESHOLD = 0.85


def log(m=""):
    print(m, flush=True)


def geo(xs):
    return float(math.exp(np.mean(np.log(np.maximum(np.asarray(xs, float), 1e-12)))))


def answer(members, hardware, w, size, table, device):
    """Out-of-fold System 1 answer for one measured problem, scored against the measurements."""
    configs, log_rt, p_fail = score_configs(members, hardware, w, size, WORKLOAD_NAMES, device)
    best_i, sig = pick(configs, log_rt, p_fail)
    cfg = configs[best_i]
    best_rt = min(t["runtime_sec"] for t in table.values())
    got = table.get(config_key(cfg))
    picked_rt = got["runtime_sec"] if got else None
    return {"workload": w, "size": size, "config": cfg, "signals": sig,
            "picked_runtime": picked_rt, "best_runtime": best_rt, "failed": got is None,
            "good": picked_rt is not None and picked_rt <= best_rt * GOOD_WITHIN,
            "pct_of_best": 0.0 if picked_rt is None else best_rt / picked_rt}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = load_hardware()
    rows = load_measurements()
    tables, _ = build_tables(rows)
    sizes = {w: sorted(KERNELS[w].sizes) for w in WORKLOAD_NAMES}
    results = {"good_within": GOOD_WITHIN, "threshold": THRESHOLD}
    t_all = time.time()

    # ---------------- 6-fold cross-validation by size
    log("=== System 1 cross-validation (6 folds by problem size) ===")
    oof = []
    for k in range(6):
        held = {(w, sizes[w][k]) for w in WORKLOAD_NAMES}
        x, y, f = rows_from_measurements(rows, hardware, keep=lambda w, s: (w, s) not in held)
        members = train_ensemble(x, y, f, device)
        for w, s in sorted(held):
            if (w, s) in tables:
                oof.append({**answer(members, hardware, w, s, tables[(w, s)], device), "fold": k})
        fk = [r for r in oof if r["fold"] == k]
        log(f"  fold {k}: {sum(r['good'] for r in fk)}/{len(fk)} picks within 5% of best, "
            f"{100 * geo([r['pct_of_best'] or 1e-3 for r in fk]):.1f}% of best speed")

    # ---------------- calibration (fit on all out-of-fold picks, checked leave-one-out)
    feats = [confidence_features(r["signals"]) for r in oof]
    labels = [float(r["good"]) for r in oof]
    cal = Calibrator().fit(feats, labels)
    loo = []
    for i in range(len(oof)):
        c = Calibrator().fit(feats[:i] + feats[i + 1:], labels[:i] + labels[i + 1:])
        loo.append(c(feats[i]))
    for r, c in zip(oof, loo):
        r["confidence"] = c
    conf_t = torch.tensor(loo)
    ece, rel = reliability(conf_t, torch.tensor(labels).bool())
    auto = [r for r in oof if r["confidence"] >= THRESHOLD]
    s1 = {
        "problems": len(oof),
        "pick_within_5pct": round(float(np.mean(labels)), 4),
        "pct_of_best_speed": round(100 * geo([r["pct_of_best"] or 1e-3 for r in oof]), 1),
        "picked_failing_config": sum(r["failed"] for r in oof),
        "confidence_ece_leave_one_out": ece, "reliability": rel,
        "auto_share_at_085": round(len(auto) / len(oof), 4),
        "auto_within_5pct": round(float(np.mean([r["good"] for r in auto])), 4) if auto else None,
        "auto_pct_of_best": round(100 * geo([r["pct_of_best"] or 1e-3 for r in auto]), 1) if auto else None,
        "per_workload": {w: {"within_5pct": round(float(np.mean([r["good"] for r in oof if r["workload"] == w])), 3),
                             "pct_of_best": round(100 * geo([r["pct_of_best"] or 1e-3 for r in oof if r["workload"] == w]), 1)}
                         for w in WORKLOAD_NAMES},
    }
    results["system1_cv"] = s1
    log(f"System 1 alone, zero runs before deciding: {100 * s1['pick_within_5pct']:.1f}% of picks within 5% of best, "
        f"{s1['pct_of_best_speed']}% of best speed, {s1['picked_failing_config']} failing picks")
    log(f"confidence (leave-one-out calibration): ECE {ece:.3f}; at {THRESHOLD}: auto {100 * s1['auto_share_at_085']:.0f}% "
        f"of problems, {s1['auto_within_5pct']} within 5%, {s1['auto_pct_of_best']}% of best speed")

    # ---------------- dual process on the held-out test sizes
    log("\n=== dual process on held-out test sizes (unseen by System 1 and by the step model) ===")
    step = HPCDS1.load(device="cpu")
    test = {(w, KERNELS[w].test_size) for w in WORKLOAD_NAMES}
    x, y, f = rows_from_measurements(rows, hardware, keep=lambda w, s: (w, s) not in test)
    members = train_ensemble(x, y, f, device)
    cal_rows = [r for r in oof if (r["workload"], r["size"]) not in test]
    cal_t = Calibrator().fit([confidence_features(r["signals"]) for r in cal_rows], [float(r["good"]) for r in cal_rows])
    pol = defaultdict(lambda: {"pct": [], "runs": []})
    per_problem = []
    for w, s in sorted(test):
        table = tables[(w, s)]
        runner = TableRunner(table, hardware)
        best = min(t["runtime_sec"] for t in table.values())
        a = answer(members, hardware, w, s, table, device)
        conf = cal_t(confidence_features(a["signals"]))
        start = a["config"] if not a["failed"] else DEFAULT_START
        s1_rt, s1_runs = (a["picked_runtime"], 1) if not a["failed"] else (table[config_key(DEFAULT_START)]["runtime_sec"], 2)
        s12 = run_tuning(step.policy(0.0), w, s, start, runner)
        s2d = run_tuning(step.policy(0.0), w, s, DEFAULT_START, runner)
        rules = run_tuning(heuristic_policy, w, s, DEFAULT_START, runner)
        dual = (s1_rt, s1_runs) if conf >= THRESHOLD else (s12.final_runtime, s12.experiments)
        for name, (rt, runs) in {"System 1 alone": (s1_rt, s1_runs),
                                 "System 1 -> System 2": (s12.final_runtime, s12.experiments),
                                 "Dual process (System 2 only when unsure)": dual,
                                 "System 2 alone, from 1x1 CPU fp32": (s2d.final_runtime, s2d.experiments),
                                 "Rules of thumb, from 1x1 CPU fp32": (rules.final_runtime, rules.experiments)}.items():
            pol[name]["pct"].append(best / rt)
            pol[name]["runs"].append(runs)
        per_problem.append({"workload": w, "size": s, "system1_config": a["config"], "confidence": round(conf, 3),
                            "system1_pct_of_best": round(100 * a["pct_of_best"], 1),
                            "used_system2": conf < THRESHOLD, "dual_pct_of_best": round(100 * best / dual[0], 1),
                            "dual_runs": dual[1]})
        log(f"  {w:10s} {s:>9}: S1 {a['config']} conf {conf:.2f} -> {100 * a['pct_of_best']:.0f}% of best"
            f"{'' if conf >= THRESHOLD else ' -> System 2'}; dual {100 * best / dual[0]:.0f}% in {dual[1]} run(s)")
    summary = [{"policy": n, "pct_of_best": round(100 * geo(v["pct"]), 1), "mean_runs": round(float(np.mean(v["runs"])), 2)}
               for n, v in pol.items()]
    for p in summary:
        log(f"  {p['policy']:42s} {p['pct_of_best']:5.1f}% of best  {p['mean_runs']:.2f} real runs")
    results["dual_process_test_sizes"] = {"summary": summary, "per_problem": per_problem,
                                          "note": "8 problems, one session each: small sample"}

    # ---------------- unseen workload
    log("\n=== leave one workload out ===")
    lowo = []
    for w in WORKLOAD_NAMES:
        x, y, f = rows_from_measurements(rows, hardware, keep=lambda ww, s, w=w: ww != w)
        members = train_ensemble(x, y, f, device)
        ans = [answer(members, hardware, w, s, tables[(w, s)], device) for s in sizes[w] if (w, s) in tables]
        r = {"held_out": w, "within_5pct": round(float(np.mean([a["good"] for a in ans])), 3),
             "pct_of_best": round(100 * geo([a["pct_of_best"] or 1e-3 for a in ans]), 1),
             "failing_picks": sum(a["failed"] for a in ans)}
        lowo.append(r)
        log(f"  {w:10s} {100 * r['within_5pct']:5.1f}% within 5%  {r['pct_of_best']:5.1f}% of best  "
            f"{r['failing_picks']} failing picks  (guard escalates all of these)")
    results["leave_one_workload_out"] = lowo

    # ---------------- final model on everything measured
    x, y, f = rows_from_measurements(rows, hardware)
    members = train_ensemble(x, y, f, device)
    s1m = System1([m.cpu() for m in members], WORKLOAD_NAMES, cal, size_ranges(rows), _hw(hardware),
                  hardware.get("name"))
    torch.save(s1m.to_checkpoint(), PROJECT_ROOT / "model" / "system1.pt")
    lat = []
    for w in WORKLOAD_NAMES:
        for _ in range(20):
            lat.append(s1m.predict(hardware, w, KERNELS[w].test_size).latency_ms)
    results["latency_ms"] = {"median": round(float(np.median(lat)), 3), "p99": round(float(np.percentile(lat, 99)), 3)}
    log(f"\nsaved model/system1.pt; System 1 latency median {results['latency_ms']['median']} ms, "
        f"p99 {results['latency_ms']['p99']} ms (CPU, all 44 configs scored)")
    (PROJECT_ROOT / "results" / "system1.json").write_text(json.dumps(results, indent=2))
    viz = PROJECT_ROOT / "visualizer" / "model_internals.json"
    export = json.loads(viz.read_text())
    export["system1"] = results
    viz.write_text(json.dumps(export))
    log(f"saved results/system1.json ({time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
