#!/usr/bin/env python3
"""
Phase 1 benchmark: how much does each method cost to find a good configuration?

Every method tunes every one of the 72 measured problems against the REAL
measured runtimes (a recorder looks each configuration up in the measurements;
repeated configurations are cached, so they cost one run). Learned methods are
always evaluated out of fold: for fold k (the k-th size of every workload) they
are trained without any run of those problems.

Methods
  default            1 rank x 1 thread, CPU, fp32: no tuning
  rules              rules of thumb, step by step from the default
  random             random configurations (5 seeds)
  greedy             tries every neighbour each step (what an engineer does by hand)
  bo_tpe, bo_gp      Bayesian optimisation with Optuna (TPE; Gaussian process), 5/3 seeds
  xgb_rank           gradient-boosted runtime + failure model, runs configs in predicted order
  system1_rank       System 1, runs configs in predicted order (no tuning runs before the first)
  system2            the step model, from the default
  system2_rand       the step model, from random measured configurations (3 seeds)
  dual               System 1's pick, then System 2 from there

Reported, as geometric means over problems:
  % of the best measured configuration's speed reached after 1, 2, 3, 5, 10 real runs
  real runs needed to get within 5% of the best (and share of problems that get there)
  tuning cost of those runs in core-seconds and GPU-seconds, from the measured runtimes

    python scripts/benchmark.py     -> results/benchmark.json (+ hpc_ds1/results/benchmark.json)
"""

import json
import math
import random
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
from hpc_ds1.dataset import build_states, build_tables, load_hardware, load_measurements  # noqa: E402
from hpc_ds1.kernels import KERNELS, WORKLOAD_NAMES  # noqa: E402
from hpc_ds1.paths import RESULTS_DIR  # noqa: E402
from hpc_ds1.space import all_configs, config_key, make_config  # noqa: E402
from hpc_ds1.system1 import _cfg_vec, _job_vec  # noqa: E402
from hpc_ds1.training import split as _unused, tensors, train_one  # noqa: E402,F401
from hpc_ds1.tuner import heuristic_policy, make_greedy_search_policy, run_tuning  # noqa: E402

warnings.filterwarnings("ignore")
BUDGET = 10
CHECKPOINTS = (1, 2, 3, 5, 10)
DEFAULT = make_config(1, 1, False, "fp32")


def log(m=""):
    print(m, flush=True)


def geo(xs):
    return float(math.exp(np.mean(np.log(np.maximum(np.asarray(xs, float), 1e-6)))))


class Recorder:
    """Stands in for the machine: real measured runtimes, in the order a method asks."""

    def __init__(self, table, hardware):
        self.table, self.hardware = table, hardware
        self.order, self.seen = [], {}

    def __call__(self, config):
        k = config_key(make_config(**config))
        if k not in self.seen:
            t = self.table.get(k)
            self.seen[k] = t
            self.order.append((k, t["runtime_sec"] if t else None))
        return self.seen[k]


def cost_of(key, runtime):
    """(core-seconds, gpu-seconds) of one run. Failed runs are counted as a run but
    their (unknown, usually short) time is not."""
    if runtime is None:
        return 0.0, 0.0
    ranks, threads, gpu, _ = key
    return (threads if gpu else ranks * threads) * runtime, (runtime if gpu else 0.0)


def summarize(order, best, floor):
    """Best-so-far curve and costs from the runs a method made, in order. Until a method
    has found a working configuration it is credited with the untuned default (floor),
    the job the user would have run anyway, so one failed run does not count as 0%."""
    curve, found, core, gpu, to5 = [], float("inf"), 0.0, 0.0, None
    cum = []
    for i, (k, rt) in enumerate(order[:BUDGET], start=1):
        c, g = cost_of(k, rt)
        core, gpu = core + c, gpu + g
        if rt is not None:
            found = min(found, rt)
        curve.append(best / min(found, floor))
        cum.append((core, gpu))
        if to5 is None and found <= best * 1.05:
            to5 = i
    while len(curve) < BUDGET:  # a method that stopped keeps what it found
        curve.append(curve[-1] if curve else best / floor)
        cum.append(cum[-1] if cum else (0.0, 0.0))
    used = min(len(order), BUDGET)
    return {"curve": curve, "runs_used": used, "runs_to_5pct": to5,
            "core_seconds": cum[used - 1][0] if used else 0.0, "gpu_seconds": cum[used - 1][1] if used else 0.0,
            "core_to_5pct": cum[to5 - 1][0] if to5 else None, "final_pct": curve[used - 1] if used else best / floor}


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------

def m_default(table, hw, **_):
    r = Recorder(table, hw)
    r(DEFAULT)
    return r.order


def m_rules(table, hw, inst, **_):
    r = Recorder(table, hw)
    run_tuning(heuristic_policy, inst[0], inst[1], DEFAULT, r, max_steps=BUDGET)
    return r.order


def m_greedy(table, hw, inst, **_):
    r = Recorder(table, hw)
    run_tuning(make_greedy_search_policy(r), inst[0], inst[1], DEFAULT, r, max_steps=BUDGET)
    return r.order


def m_random(table, hw, seed, **_):
    r = Recorder(table, hw)
    cfgs = all_configs(bool(hw.get("gpu_model")))
    random.Random(seed).shuffle(cfgs)
    for c in cfgs[:BUDGET]:
        r(c)
    return r.order


def m_bo(table, hw, seed, sampler, **_):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    r = Recorder(table, hw)
    has_gpu = bool(hw.get("gpu_model"))
    worst = max(t["runtime_sec"] for t in table.values())
    if sampler == "tpe":
        samp = optuna.samplers.TPESampler(seed=seed, n_startup_trials=3)
    else:
        samp = optuna.samplers.GPSampler(seed=seed, n_startup_trials=3)
    study = optuna.create_study(sampler=samp)

    def objective(trial):
        gpu = trial.suggest_categorical("gpu", [False, True] if has_gpu else [False])
        prec = trial.suggest_categorical("precision", ["fp32", "bf16"])
        threads = trial.suggest_categorical("threads", [1, 2, 4, 8, 16])
        ranks = trial.suggest_categorical("ranks", [1, 2, 4, 8])
        c = make_config(1 if gpu else ranks, threads, gpu, prec)
        if not gpu and ranks * threads > 32:
            return math.log10(worst) + 2  # outside the space: rejected without running
        t = r(c)
        return math.log10(t["runtime_sec"]) if t else math.log10(worst) + 1

    for _ in range(60):  # trials outside the space or repeats are free; stop at BUDGET real runs
        if len(r.order) >= BUDGET:
            break
        study.optimize(objective, n_trials=1)
    return r.order


def m_rank(table, hw, ranking, **_):
    r = Recorder(table, hw)
    for k in ranking[:BUDGET]:
        r(make_config(*k))
    return r.order


def m_system2(table, hw, inst, model, start=DEFAULT, **_):
    r = Recorder(table, hw)
    run_tuning(model.policy(0.0), inst[0], inst[1], start, r, max_steps=BUDGET)
    return r.order


# ---------------------------------------------------------------------------
# out-of-fold learned models
# ---------------------------------------------------------------------------

def parse_key(k):
    a, b, g, p = k.split(",")
    return (int(a), int(b), g in ("1", "True"), p)


def system1_rankings():
    """Out-of-fold System 1 rankings (scripts/system1_oof.py): predicted-fast first,
    configs System 1 expects to fail last."""
    oof = json.loads((PROJECT_ROOT / "results" / "system1_oof.json").read_text())
    out = {}
    for prob, est in oof.items():
        w, s = prob.split(":")
        rt = {parse_key(k): v for k, v in est["rt"].items()}
        fail = {parse_key(k): v for k, v in est.get("fail", {}).items()}
        out[(w, int(s))] = sorted(rt, key=lambda k: (fail.get(k, 0) >= 0.5, rt[k]))
    return out


def xgb_rankings(rows, hardware, folds):
    import xgboost as xgb
    configs = [config_key(c) for c in all_configs(bool(hardware.get("gpu_model")))]
    out = {}
    for held in folds:
        train = [r for r in rows if (r["workload"], r["size"]) not in held]
        X = np.array([_job_vec(hardware, r["workload"], r["size"], WORKLOAD_NAMES) + _cfg_vec(r["config"], hardware)
                      for r in train])
        ok = np.array([r["ok"] and r.get("correct", False) for r in train])
        y = np.array([math.log10(r["telemetry"]["runtime_sec"]) if o else 0.0 for r, o in zip(train, ok)])
        reg = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
                               colsample_bytree=0.9, random_state=0).fit(X[ok], y[ok])
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=0).fit(X, (~ok).astype(int))
        for w, s in held:
            Q = np.array([_job_vec(hardware, w, s, WORKLOAD_NAMES) + _cfg_vec(make_config(*k), hardware) for k in configs])
            pr, pf = reg.predict(Q), clf.predict_proba(Q)[:, 1]
            out[(w, s)] = [configs[i] for i in sorted(range(len(configs)), key=lambda i: (pf[i] >= 0.5, pr[i]))]
    return out


def system2_models(states, folds, device, size):
    """Step model per fold, trained without any state of that fold's problems."""
    models = {}
    for k, held in enumerate(folds):
        rest = [s for s in states if (s["workload"], s["size"]) not in held]
        random.Random(k).shuffle(rest)
        n_val = max(1, int(len(rest) * 0.15))
        net, _ = train_one(size, tensors(rest[n_val:], device), tensors(rest[:n_val], device), device)
        m = HPCDS1(net.eval(), device)
        for p in held:
            models[p] = m
        log(f"  System 2 fold {k}: trained on {len(rest)} states")
    return models


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = load_hardware()
    rows = load_measurements()
    tables, _ = build_tables(rows)
    states = build_states(tables, hardware)
    sizes = {w: sorted(KERNELS[w].sizes) for w in WORKLOAD_NAMES}
    folds = [{(w, sizes[w][k]) for w in WORKLOAD_NAMES if k < len(sizes[w]) and (w, sizes[w][k]) in tables}
             for k in range(max(len(v) for v in sizes.values()))]
    problems = sorted(tables)
    step_size = json.loads((RESULTS_DIR / "metrics.json").read_text())["model"]["size"]

    log("training out-of-fold learned models ...")
    s1 = system1_rankings()
    xg = xgb_rankings(rows, hardware, folds)
    s2 = system2_models(states, folds, device, step_size)

    methods = {
        "default": [dict(fn=m_default)],
        "rules": [dict(fn=m_rules)],
        "random": [dict(fn=m_random, seed=s) for s in range(5)],
        "greedy": [dict(fn=m_greedy)],
        "bo_tpe": [dict(fn=m_bo, seed=s, sampler="tpe") for s in range(5)],
        "bo_gp": [dict(fn=m_bo, seed=s, sampler="gp") for s in range(3)],
        "xgb_rank": [dict(fn=m_rank, ranking="xgb")],
        "system1_rank": [dict(fn=m_rank, ranking="s1")],
        "system2": [dict(fn=m_system2)],
        "system2_rand": [dict(fn=m_system2, start="rand", seed=s) for s in range(3)],
        "dual": [dict(fn=m_system2, start="s1")],
    }
    per = defaultdict(list)
    for i, inst in enumerate(problems):
        table = tables[inst]
        best = min(t["runtime_sec"] for t in table.values())
        floor = table.get(config_key(DEFAULT), {"runtime_sec": max(t["runtime_sec"] for t in table.values())})["runtime_sec"]
        for name, variants in methods.items():
            runs = []
            for v in variants:
                kw = dict(v)
                fn = kw.pop("fn")
                if kw.get("ranking") == "xgb":
                    kw["ranking"] = xg[inst]
                elif kw.get("ranking") == "s1":
                    kw["ranking"] = s1[inst]
                if fn is m_system2:
                    kw["model"] = s2[inst]
                    if kw.get("start") == "s1":
                        kw["start"] = make_config(*s1[inst][0])
                    elif kw.get("start") == "rand":
                        kw["start"] = random.Random(kw.pop("seed")).choice(sorted(table, key=str))
                        kw["start"] = make_config(*kw["start"])
                    else:
                        kw["start"] = DEFAULT
                runs.append(summarize(fn(table, hardware, inst=inst, **kw), best, floor))
            per[name].append({"problem": f"{inst[0]}:{inst[1]}", "runs": runs})
        if (i + 1) % 12 == 0:
            log(f"  {i + 1}/{len(problems)} problems ({time.time() - t0:.0f}s)")

    report = {"problems": len(problems), "budget": BUDGET, "methods": {}}
    for name, items in per.items():
        flat = [r for it in items for r in it["runs"]]
        curve = {b: round(100 * geo([r["curve"][b - 1] for r in flat]), 1) for b in CHECKPOINTS}
        reached = [r for r in flat if r["runs_to_5pct"] is not None]
        report["methods"][name] = {
            "pct_of_best_after_runs": curve,
            "median_pct_of_best_after_runs": {b: round(100 * float(np.median([r["curve"][b - 1] for r in flat])), 1) for b in CHECKPOINTS},
            "within_5pct_share": round(len(reached) / len(flat), 3),
            "median_runs_to_5pct": float(np.median([r["runs_to_5pct"] for r in reached])) if reached else None,
            "mean_runs_used": round(float(np.mean([r["runs_used"] for r in flat])), 2),
            "core_hours_tuning_total": round(sum(r["core_seconds"] for r in flat) / len(flat) * len(items) / 3600, 4),
            "gpu_hours_tuning_total": round(sum(r["gpu_seconds"] for r in flat) / len(flat) * len(items) / 3600, 4),
        }
    log("\n% of best after N real runs (geometric mean over 72 problems), runs used, tuning cost:")
    log(f"{'method':14s} " + " ".join(f"{'@' + str(b):>7s}" for b in CHECKPOINTS) + "   reach5%  med.runs  runs  core-h  gpu-h")
    for name, m in report["methods"].items():
        c = m["pct_of_best_after_runs"]
        log(f"{name:14s} " + " ".join(f"{c[b]:7.1f}" for b in CHECKPOINTS) +
            f"   {100 * m['within_5pct_share']:6.0f}%  {m['median_runs_to_5pct'] or '-':>8}  {m['mean_runs_used']:4.1f}"
            f"  {m['core_hours_tuning_total']:6.3f}  {m['gpu_hours_tuning_total']:6.3f}")
    report["seconds"] = round(time.time() - t0)
    for path in (PROJECT_ROOT / "results" / "benchmark.json", RESULTS_DIR / "benchmark.json"):
        path.write_text(json.dumps(report, indent=2))
    (PROJECT_ROOT / "results" / "benchmark_per_problem.json").write_text(json.dumps(per))
    log(f"saved results/benchmark.json ({report['seconds']}s)")


if __name__ == "__main__":
    main()
