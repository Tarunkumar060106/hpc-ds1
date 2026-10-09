"""Metrics calculation for HPC-DS1 evaluation."""

import numpy as np
from typing import List, Dict, Tuple


def compute_speedup(baseline_runtime: float, actual_runtime: float) -> float:
    """
    Compute speedup: baseline / actual.

    Args:
        baseline_runtime: Single-thread, no-GPU runtime
        actual_runtime: Time with current config

    Returns:
        Speedup factor (>1 is improvement)
    """
    if actual_runtime <= 0:
        return 1.0
    return baseline_runtime / actual_runtime


def compute_tuning_cost(runtimes: List[float]) -> float:
    """
    Compute total tuning cost in seconds.

    Args:
        runtimes: List of experiment runtimes

    Returns:
        Total cost (sum of all runtimes)
    """
    return sum(runtimes)


def compute_efficiency(speedup: float, tuning_cost_seconds: float) -> float:
    """
    Compute efficiency: speedup per minute spent tuning.

    Formula: (speedup - 1.0) / (cost / 60)

    Interpretation:
      - 0.1 = 0.1x improvement per minute
      - 1.0 = 1x improvement per minute (excellent)

    Args:
        speedup: Final speedup vs baseline
        tuning_cost_seconds: Total time spent tuning

    Returns:
        Efficiency (speedup gain per minute)
    """
    if tuning_cost_seconds <= 0:
        return 0.0

    improvement = max(0, speedup - 1.0)
    minutes = tuning_cost_seconds / 60.0
    return improvement / minutes if minutes > 0 else 0.0


def aggregate_metrics(runtimes: List[float], speedups: List[float]) -> Dict:
    """
    Aggregate metrics across multiple runs/seeds.

    Args:
        runtimes: List of total tuning costs per run
        speedups: List of final speedups per run

    Returns:
        {mean_speedup, std_speedup, mean_cost, efficiency, ...}
    """

    runtimes = np.array(runtimes)
    speedups = np.array(speedups)

    return {
        "mean_speedup": float(np.mean(speedups)),
        "std_speedup": float(np.std(speedups)),
        "min_speedup": float(np.min(speedups)),
        "max_speedup": float(np.max(speedups)),
        "mean_cost": float(np.mean(runtimes)),
        "std_cost": float(np.std(runtimes)),
        "total_cost": float(np.sum(runtimes)),
        "efficiency": float(np.mean([
            compute_efficiency(s, c) for s, c in zip(speedups, runtimes)
        ])),
    }
