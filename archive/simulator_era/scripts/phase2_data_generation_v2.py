#!/usr/bin/env python3
"""
Phase 2 v2: Multi-workload dataset generation

Fixes the v1 bugs (runtime floor collapse + first-action tie-break) and
replaces the single N-body kernel with 6 HPC workloads that have genuinely
different scaling laws (see hpc_ds1/workloads.py), so the dataset actually
contains a mix of correct actions instead of one label repeated everywhere.

Generates N baseline configs per workload, evaluates ALL valid actions for
each one (not just 4), and labels the trajectory with whichever action
gives the largest real (>0.5%) runtime improvement, or STOP_TUNING if none
does. Exports train/val/test JSONL splits plus a full config CSV.
"""

import sys
import json
import csv
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hpc_ds1.workloads import (
    WORKLOADS, ACTION_NAMES, compute_runtime, apply_action, valid_actions_for,
    estimate_telemetry, classify_bottleneck,
)

CONFIGS_PER_WORKLOAD = 150
IMPROVEMENT_THRESHOLD = 0.005  # 0.5% — below this, action doesn't count as "better"

MPI_CHOICES = [1, 1, 1, 2, 2, 2, 4, 4, 8, 8, 16]
OMP_CHOICES = [1, 1, 1, 2, 2, 2, 4, 4, 8]
BLOCK_CHOICES = [32, 64, 128, 256, 512]
GPU_ENABLED_PROB = 0.5
SMALL_PROBLEM_PROB = 0.15  # forces some below-GPU-threshold scenarios


def print_phase(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def sample_config(workload, rng):
    spec = WORKLOADS[workload]
    gpu_enabled = rng.random() < GPU_ENABLED_PROB

    config = {
        "mpi_ranks": rng.choice(MPI_CHOICES),
        "omp_threads": rng.choice(OMP_CHOICES),
        "gpu_enabled": gpu_enabled,
        "cuda_block_size": rng.choice(BLOCK_CHOICES) if gpu_enabled else None,
        "reduced_precision": False,
        "prefetch": False,
    }

    if rng.random() < SMALL_PROBLEM_PROB:
        problem_size = max(1, spec["gpu_min_problem"] // 10)
    else:
        problem_size = rng.choice(spec["problem_sizes"])

    return config, problem_size


def build_trajectory(workload, config, problem_size, sample_id):
    base_runtime, base_compute, base_comm = compute_runtime(
        workload, config["mpi_ranks"], config["omp_threads"], config["gpu_enabled"],
        problem_size, config.get("cuda_block_size"), config["reduced_precision"], config["prefetch"],
    )

    telemetry = estimate_telemetry(
        workload, config["mpi_ranks"], config["omp_threads"], config["gpu_enabled"],
        problem_size, base_compute, base_comm,
    )
    bottleneck = classify_bottleneck(workload, config["mpi_ranks"], base_comm, base_compute,
                                      telemetry["memory_util_pct"])

    candidates = {}
    best_action, best_runtime = None, base_runtime

    for action in valid_actions_for(config):
        new_config = apply_action(config, action, workload)
        rt, _, _ = compute_runtime(
            workload, new_config["mpi_ranks"], new_config["omp_threads"], new_config["gpu_enabled"],
            problem_size, new_config.get("cuda_block_size"),
            new_config.get("reduced_precision", False), new_config.get("prefetch", False),
        )
        improvement = (base_runtime - rt) / base_runtime
        candidates[action] = {"runtime": round(rt, 5), "improvement": round(improvement, 5)}

        if rt < best_runtime * (1 - IMPROVEMENT_THRESHOLD) and rt < best_runtime:
            best_runtime = rt
            best_action = action

    if best_action is None:
        best_action = "STOP_TUNING"
        best_improvement = 0.0
    else:
        best_improvement = candidates[best_action]["improvement"]

    state = {
        "hardware": {
            "cpu_cores": 8, "cpu_freq_ghz": 3.5, "l3_cache_mb": 24,
            "gpu_model": "RTX 4050", "gpu_memory_gb": 6.0, "ram_gb": 24,
        },
        "workload": {
            "name": workload, "problem_size": problem_size,
            "algorithm_type": "embarrassingly_parallel" if WORKLOADS[workload]["serial_fraction"] < 0.03 else "moderate",
            "flops_estimate": problem_size ** WORKLOADS[workload]["work_exponent"],
        },
        "current_config": {
            "mpi_ranks": config["mpi_ranks"], "omp_threads": config["omp_threads"],
            "gpu_enabled": config["gpu_enabled"], "cuda_block_size": config.get("cuda_block_size"),
        },
        "telemetry": {
            "runtime_sec": round(base_runtime, 5),
            **telemetry,
        },
        "experiment_history": [],
    }

    return {
        "sample": sample_id,
        "workload": workload,
        "state": state,
        "candidates": candidates,
        "best_action": best_action,
        "best_improvement": best_improvement,
        "bottleneck": bottleneck,
    }


def main():
    print("\n" + "=" * 70)
    print("  PHASE 2 v2: MULTI-WORKLOAD DATASET GENERATION")
    print("=" * 70)
    print(f"\nWorkloads: {', '.join(WORKLOADS.keys())}")
    print(f"Configs per workload: {CONFIGS_PER_WORKLOAD}")

    rng = random.Random(7)
    data_dir = PROJECT_ROOT / "data" / "phase2_v2_results"
    data_dir.mkdir(parents=True, exist_ok=True)

    print_phase("GENERATING TRAJECTORIES")

    all_trajectories = []
    per_workload_counts = {}
    sample_id = 1

    for workload in WORKLOADS:
        wl_trajectories = []
        for _ in range(CONFIGS_PER_WORKLOAD):
            config, problem_size = sample_config(workload, rng)
            traj = build_trajectory(workload, config, problem_size, sample_id)
            wl_trajectories.append(traj)
            sample_id += 1

        action_counts = {}
        for t in wl_trajectories:
            action_counts[t["best_action"]] = action_counts.get(t["best_action"], 0) + 1
        per_workload_counts[workload] = action_counts

        print(f"[{workload:12s}] {len(wl_trajectories)} trajectories | "
              + ", ".join(f"{k}={v}" for k, v in sorted(action_counts.items(), key=lambda x: -x[1])))

        all_trajectories.extend(wl_trajectories)

    print(f"\n[OK] Generated {len(all_trajectories)} total trajectories across {len(WORKLOADS)} workloads")

    # ------------------------------------------------------------------
    print_phase("GLOBAL LABEL DISTRIBUTION")

    global_counts = {}
    for t in all_trajectories:
        global_counts[t["best_action"]] = global_counts.get(t["best_action"], 0) + 1
    total = len(all_trajectories)
    for k, v in sorted(global_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v:4d}  ({100*v/total:.1f}%)")

    # ------------------------------------------------------------------
    print_phase("TRAIN / VAL / TEST SPLIT")

    rng.shuffle(all_trajectories)
    n = len(all_trajectories)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train = all_trajectories[:n_train]
    val = all_trajectories[n_train:n_train + n_val]
    test = all_trajectories[n_train + n_val:]

    print(f"Total: {n}  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    # ------------------------------------------------------------------
    print_phase("SAVING DATASET")

    for name, split in [("train_trajectories.jsonl", train),
                         ("val_trajectories.jsonl", val),
                         ("test_trajectories.jsonl", test),
                         ("all_trajectories.jsonl", all_trajectories)]:
        path = data_dir / name
        with open(path, "w") as f:
            for t in split:
                f.write(json.dumps(t) + "\n")
        print(f"[OK] {path} ({len(split)} rows)")

    # Flat CSV for quick inspection
    csv_path = data_dir / "all_trajectories_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "workload", "mpi_ranks", "omp_threads", "gpu_enabled",
                          "problem_size", "runtime_sec", "best_action", "best_improvement", "bottleneck"])
        for t in all_trajectories:
            cfg = t["state"]["current_config"]
            writer.writerow([t["sample"], t["workload"], cfg["mpi_ranks"], cfg["omp_threads"],
                              cfg["gpu_enabled"], t["state"]["workload"]["problem_size"],
                              t["state"]["telemetry"]["runtime_sec"], t["best_action"],
                              t["best_improvement"], t["bottleneck"]])
    print(f"[OK] {csv_path}")

    print_phase("PHASE 2 v2 COMPLETE")
    print(f"Dataset ready: {n} trajectories across {len(WORKLOADS)} workloads")
    print(f"Action classes present: {len(global_counts)} / {len(ACTION_NAMES)+1}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
