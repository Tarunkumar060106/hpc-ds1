#!/usr/bin/env python3
"""
Phase 1: Experiment Infrastructure & Baselines (Days 2-4)

Goal: Build the plumbing for data collection and validation
  - Wire up data schemas (already created)
  - Implement 3 baseline methods (Random, Grid, Heuristic)
  - Create experiment logging
  - Create metrics calculator
  - Unit tests

This phase outputs:
  - hpc_ds1/baselines/  (3 baseline implementations)
  - hpc_ds1/logging/    (experiment database)
  - hpc_ds1/metrics/    (metrics calculator)
  - tests/              (unit tests)
"""

import sys
import os
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_phase(title):
    """Print phase header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def main():
    """Execute Phase 1 infrastructure setup."""

    print_phase("PHASE 1: INFRASTRUCTURE SETUP")
    print("Building experiment harness and baseline implementations\n")

    # Step 1: Verify schemas exist
    print("[1/5] Verifying data schemas...")
    try:
        from hpc_ds1.schemas import HPCState, HPC_DS1_Output, HPC_ACTIONS
        print("  [OK] HPCState imported")
        print("  [OK] HPC_DS1_Output imported")
        print(f"  [OK] 8 HPC_ACTIONS defined: {', '.join(HPC_ACTIONS[:3])}...")
    except ImportError as e:
        print(f"  [FAIL] Schema import failed: {e}")
        return 1

    # Step 2: Create baselines directory
    print("\n[2/5] Creating baseline implementations...")
    baselines_dir = PROJECT_ROOT / "hpc_ds1" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [OK] Created {baselines_dir}")

    # Step 2a: Random Baseline
    random_baseline = '''"""Random baseline - selects actions randomly."""

import random
from typing import List, Dict
from hpc_ds1.schemas import HPCState, HPC_ACTIONS


class RandomBaseline:
    """Random action selection (worst-case baseline)."""

    def __init__(self):
        self.name = "Random"

    def select_action(self, state: HPCState, valid_actions: List[str] = None) -> str:
        """Select a random valid action."""
        if valid_actions is None:
            valid_actions = HPC_ACTIONS
        return random.choice(valid_actions)

    def __repr__(self):
        return f"RandomBaseline()"


def create_random_baseline() -> RandomBaseline:
    """Factory function."""
    return RandomBaseline()
'''

    with open(baselines_dir / "random.py", "w") as f:
        f.write(random_baseline)
    print("  [OK] RandomBaseline created")

    # Step 2b: Grid Baseline
    grid_baseline = '''"""Grid search baseline - explores in fixed order."""

from typing import List
from hpc_ds1.schemas import HPCState, HPC_ACTIONS


class GridBaseline:
    """Grid search - tries actions in fixed order."""

    def __init__(self, step_size: float = 0.5):
        """
        Args:
            step_size: Factor for scaling parallelism (unused, for compatibility)
        """
        self.name = "Grid"
        self.action_order = [
            "INCREASE_MPI",
            "INCREASE_OMP",
            "ENABLE_GPU",
            "INCREASE_BLOCK_SIZE",
            "DECREASE_MPI",
            "DISABLE_GPU",
            "DECREASE_OMP",
            "DECREASE_BLOCK_SIZE",
        ]
        self.step_count = 0

    def select_action(self, state: HPCState, valid_actions: List[str] = None) -> str:
        """Select next action in sequence."""
        if valid_actions is None:
            valid_actions = HPC_ACTIONS

        # Cycle through action_order and pick first valid
        for action in self.action_order:
            if action in valid_actions:
                self.step_count += 1
                return action

        # Fallback: pick first valid action
        return valid_actions[0] if valid_actions else HPC_ACTIONS[0]

    def __repr__(self):
        return f"GridBaseline(step={self.step_count})"


def create_grid_baseline() -> GridBaseline:
    """Factory function."""
    return GridBaseline()
'''

    with open(baselines_dir / "grid.py", "w") as f:
        f.write(grid_baseline)
    print("  [OK] GridBaseline created")

    # Step 2c: Heuristic Baseline
    heuristic_baseline = '''"""Heuristic baseline - hand-coded domain rules."""

from typing import List
from hpc_ds1.schemas import HPCState, HPC_ACTIONS


class HeuristicBaseline:
    """Rule-based heuristic for HPC tuning."""

    def __init__(self):
        self.name = "Heuristic"
        self.no_improvement_steps = 0
        self.last_speedup = 1.0

    def select_action(self, state: HPCState, valid_actions: List[str] = None) -> str:
        """
        Select action based on hand-coded rules.

        Rules:
        1. If GPU util < 30% and CPU util > 80% -> ENABLE_GPU
        2. If MPI comm overhead > 25% -> DECREASE_MPI
        3. If memory util > 85% -> INCREASE_OMP (more threads, less MPI)
        4. If no improvement for 3 steps -> STOP
        5. Default: INCREASE_MPI
        """
        if valid_actions is None:
            valid_actions = HPC_ACTIONS

        tel = state.telemetry

        # Rule 1: GPU underutilized but CPU busy
        if (tel.gpu_util_pct is not None and tel.gpu_util_pct < 30 and
            tel.cpu_util_pct > 80 and "ENABLE_GPU" in valid_actions):
            return "ENABLE_GPU"

        # Rule 2: Communication overhead too high
        if tel.mpi_comm_overhead_pct > 25 and "DECREASE_MPI" in valid_actions:
            return "DECREASE_MPI"

        # Rule 3: Memory pressure
        if tel.memory_util_pct > 85 and "INCREASE_OMP" in valid_actions:
            return "INCREASE_OMP"

        # Rule 4: No improvement for too long
        speedup = state.experiment_history[-1].speedup_vs_baseline if state.experiment_history else 1.0
        if speedup <= self.last_speedup:
            self.no_improvement_steps += 1
        else:
            self.no_improvement_steps = 0
            self.last_speedup = speedup

        if self.no_improvement_steps >= 3 and "STOP" in valid_actions:
            return "STOP"

        # Rule 5: Default - increase MPI for parallelism
        if "INCREASE_MPI" in valid_actions:
            return "INCREASE_MPI"

        # Fallback
        return valid_actions[0] if valid_actions else HPC_ACTIONS[0]

    def __repr__(self):
        return f"HeuristicBaseline(no_improve={self.no_improvement_steps})"


def create_heuristic_baseline() -> HeuristicBaseline:
    """Factory function."""
    return HeuristicBaseline()
'''

    with open(baselines_dir / "heuristic.py", "w") as f:
        f.write(heuristic_baseline)
    print("  [OK] HeuristicBaseline created")

    # Step 2d: Init file
    init_baselines = '''"""HPC tuning baselines."""

from .random import RandomBaseline, create_random_baseline
from .grid import GridBaseline, create_grid_baseline
from .heuristic import HeuristicBaseline, create_heuristic_baseline

__all__ = [
    "RandomBaseline",
    "GridBaseline",
    "HeuristicBaseline",
    "create_random_baseline",
    "create_grid_baseline",
    "create_heuristic_baseline",
]
'''

    with open(baselines_dir / "__init__.py", "w") as f:
        f.write(init_baselines)
    print("  [OK] __init__.py created")

    # Step 3: Create logging module
    print("\n[3/5] Creating experiment logging...")
    logging_dir = PROJECT_ROOT / "hpc_ds1" / "logging"
    logging_dir.mkdir(parents=True, exist_ok=True)

    experiment_db = '''"""Experiment database for logging and querying."""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class ExperimentDB:
    """Simple file-based experiment database."""

    def __init__(self, db_file: Path = None):
        """
        Initialize experiment database.

        Args:
            db_file: Path to JSONL file (one JSON per line)
        """
        if db_file is None:
            db_file = Path("experiments.jsonl")

        self.db_file = Path(db_file)
        self.db_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing experiments
        self.experiments = []
        if self.db_file.exists():
            with open(self.db_file, "r") as f:
                for line in f:
                    if line.strip():
                        self.experiments.append(json.loads(line))

    def log_experiment(self, experiment: Dict[str, Any]) -> None:
        """Log a single experiment."""
        experiment["timestamp"] = experiment.get("timestamp", datetime.now().isoformat())
        self.experiments.append(experiment)

        # Append to file
        with open(self.db_file, "a") as f:
            f.write(json.dumps(experiment) + "\\n")

    def get_all_experiments(self) -> List[Dict]:
        """Get all experiments."""
        return self.experiments

    def get_experiment_count(self) -> int:
        """Get number of experiments."""
        return len(self.experiments)

    def filter_by_config(self, mpi: int = None, omp: int = None, gpu: bool = None) -> List[Dict]:
        """Filter experiments by configuration."""
        results = []
        for exp in self.experiments:
            config = exp.get("config", {})
            if (mpi is None or config.get("mpi_ranks") == mpi) and \\
               (omp is None or config.get("omp_threads") == omp) and \\
               (gpu is None or config.get("gpu_enabled") == gpu):
                results.append(exp)
        return results

    def export_csv(self, output_file: Path) -> None:
        """Export experiments to CSV."""
        import csv

        if not self.experiments:
            print(f"No experiments to export")
            return

        # Flatten nested dicts
        rows = []
        for exp in self.experiments:
            row = {
                "timestamp": exp.get("timestamp", ""),
                "mpi": exp.get("config", {}).get("mpi_ranks", ""),
                "omp": exp.get("config", {}).get("omp_threads", ""),
                "gpu": exp.get("config", {}).get("gpu_enabled", ""),
                "block_size": exp.get("config", {}).get("cuda_block_size", ""),
                "problem_size": exp.get("config", {}).get("problem_size", ""),
                "runtime": exp.get("runtime_sec", ""),
                "speedup": exp.get("speedup_vs_baseline", ""),
                "cpu_util": exp.get("telemetry", {}).get("cpu_util_pct", ""),
                "gpu_util": exp.get("telemetry", {}).get("gpu_util_pct", ""),
            }
            rows.append(row)

        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Exported {len(rows)} experiments to {output_file}")
'''

    with open(logging_dir / "experiment_db.py", "w") as f:
        f.write(experiment_db)
    print("  [OK] ExperimentDB created")

    # Step 3b: Logging init
    logging_init = '''"""Experiment logging and tracking."""

from .experiment_db import ExperimentDB

__all__ = ["ExperimentDB"]
'''

    with open(logging_dir / "__init__.py", "w") as f:
        f.write(logging_init)
    print("  [OK] logging/__init__.py created")

    # Step 4: Create metrics module
    print("\n[4/5] Creating metrics calculator...")
    metrics_dir = PROJECT_ROOT / "hpc_ds1" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics = '''"""Metrics calculation for HPC-DS1 evaluation."""

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
'''

    with open(metrics_dir / "metrics.py", "w") as f:
        f.write(metrics)
    print("  [OK] Metrics calculator created")

    # Step 4b: Metrics init
    metrics_init = '''"""Metrics for HPC-DS1 evaluation."""

from .metrics import (
    compute_speedup,
    compute_tuning_cost,
    compute_efficiency,
    aggregate_metrics,
)

__all__ = [
    "compute_speedup",
    "compute_tuning_cost",
    "compute_efficiency",
    "aggregate_metrics",
]
'''

    with open(metrics_dir / "__init__.py", "w") as f:
        f.write(metrics_init)
    print("  [OK] metrics/__init__.py created")

    # Step 5: Create unit tests
    print("\n[5/5] Creating unit tests...")
    tests_dir = PROJECT_ROOT / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    test_baselines = '''"""Unit tests for baselines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hpc_ds1.baselines import RandomBaseline, GridBaseline, HeuristicBaseline
from hpc_ds1.schemas import HPCState, Hardware, Workload, Configuration, Telemetry


def test_random_baseline():
    """Test RandomBaseline."""
    baseline = RandomBaseline()

    # Create dummy state
    state = HPCState(
        hardware=Hardware(cpu_cores=8, cpu_freq_ghz=3.5, l3_cache_mb=24,
                         gpu_memory_gb=6.0, ram_gb=24),
        workload=Workload(name="nbody", problem_size=50000,
                         algorithm_type="embarrassingly_parallel"),
        current_config=Configuration(mpi_ranks=1, omp_threads=1, gpu_enabled=False),
        telemetry=Telemetry(runtime_sec=32.1, cpu_util_pct=14, gpu_util_pct=0),
    )

    # Should select from valid actions
    valid_actions = ["INCREASE_MPI", "INCREASE_OMP", "ENABLE_GPU"]
    action = baseline.select_action(state, valid_actions)
    assert action in valid_actions
    print("[OK] RandomBaseline works")


def test_grid_baseline():
    """Test GridBaseline."""
    baseline = GridBaseline()

    state = HPCState(
        hardware=Hardware(cpu_cores=8, cpu_freq_ghz=3.5, l3_cache_mb=24,
                         gpu_memory_gb=6.0, ram_gb=24),
        workload=Workload(name="nbody", problem_size=50000,
                         algorithm_type="embarrassingly_parallel"),
        current_config=Configuration(mpi_ranks=1, omp_threads=1, gpu_enabled=False),
        telemetry=Telemetry(runtime_sec=32.1, cpu_util_pct=14, gpu_util_pct=0),
    )

    # Should follow action order
    valid_actions = ["INCREASE_MPI", "ENABLE_GPU"]
    action1 = baseline.select_action(state, valid_actions)
    assert action1 == "INCREASE_MPI"  # First in order
    print("[OK] GridBaseline works")


def test_heuristic_baseline():
    """Test HeuristicBaseline."""
    baseline = HeuristicBaseline()

    state = HPCState(
        hardware=Hardware(cpu_cores=8, cpu_freq_ghz=3.5, l3_cache_mb=24,
                         gpu_memory_gb=6.0, ram_gb=24),
        workload=Workload(name="nbody", problem_size=50000,
                         algorithm_type="embarrassingly_parallel"),
        current_config=Configuration(mpi_ranks=1, omp_threads=1, gpu_enabled=False),
        telemetry=Telemetry(runtime_sec=32.1, cpu_util_pct=80, gpu_util_pct=10),
    )

    # High CPU, low GPU -> should recommend ENABLE_GPU
    valid_actions = ["INCREASE_MPI", "ENABLE_GPU", "INCREASE_OMP"]
    action = baseline.select_action(state, valid_actions)
    assert action == "ENABLE_GPU"  # High CPU + low GPU rule
    print("[OK] HeuristicBaseline works")


if __name__ == "__main__":
    test_random_baseline()
    test_grid_baseline()
    test_heuristic_baseline()
    print("\\n[OK] All tests passed!")
'''

    with open(tests_dir / "test_baselines.py", "w") as f:
        f.write(test_baselines)
    print("  [OK] test_baselines.py created")

    # Step 5b: Tests init
    tests_init = '''"""Unit tests for HPC-DS1."""
'''

    with open(tests_dir / "__init__.py", "w") as f:
        f.write(tests_init)
    print("  [OK] tests/__init__.py created")

    # Summary
    print_phase("PHASE 1 COMPLETE")

    print("Deliverables created:")
    print("  [OK] hpc_ds1/baselines/")
    print("       - RandomBaseline (worst-case)")
    print("       - GridBaseline (systematic exploration)")
    print("       - HeuristicBaseline (domain rules)")
    print()
    print("  [OK] hpc_ds1/logging/")
    print("       - ExperimentDB (log & query experiments)")
    print()
    print("  [OK] hpc_ds1/metrics/")
    print("       - Speedup, cost, efficiency calculators")
    print()
    print("  [OK] tests/")
    print("       - Unit tests for all baselines")
    print()

    # Run tests
    print("Running unit tests...\n")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(tests_dir / "test_baselines.py")],
        cwd=PROJECT_ROOT
    )

    if result.returncode == 0:
        print("\n" + "="*70)
        print("  PHASE 1 INFRASTRUCTURE READY FOR PHASE 2!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Phase 2a: Collect 40 baseline configs")
        print("  2. Phase 2b: Generate 150 trajectories")
        print("  3. Phase 3: Train HPC-DS1 model")
        print()
        return 0
    else:
        print("\n[ERROR] Tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
