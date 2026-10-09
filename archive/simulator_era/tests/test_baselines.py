"""Unit tests for baselines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hpc_ds1.baselines import RandomBaseline, GridBaseline, HeuristicBaseline
from hpc_ds1.schemas import HPCState, Hardware, Workload, Configuration, Telemetry


def test_random_baseline():
    """Test RandomBaseline selects valid actions."""
    baseline = RandomBaseline()

    state = HPCState(
        hardware=Hardware(cpu_cores=8, cpu_freq_ghz=3.5, l3_cache_mb=24,
                         gpu_memory_gb=6.0, ram_gb=24),
        workload=Workload(name="nbody", problem_size=50000,
                         algorithm_type="embarrassingly_parallel"),
        current_config=Configuration(mpi_ranks=1, omp_threads=1, gpu_enabled=False),
        telemetry=Telemetry(runtime_sec=32.1, cpu_util_pct=14, gpu_util_pct=0),
    )

    valid_actions = ["INCREASE_MPI", "INCREASE_OMP", "ENABLE_GPU"]
    action = baseline.select_action(state, valid_actions)
    assert action in valid_actions, f"Expected valid action, got {action}"
    print("[OK] RandomBaseline: selects valid actions")


def test_grid_baseline():
    """Test GridBaseline follows fixed order."""
    baseline = GridBaseline()

    state = HPCState(
        hardware=Hardware(cpu_cores=8, cpu_freq_ghz=3.5, l3_cache_mb=24,
                         gpu_memory_gb=6.0, ram_gb=24),
        workload=Workload(name="nbody", problem_size=50000,
                         algorithm_type="embarrassingly_parallel"),
        current_config=Configuration(mpi_ranks=1, omp_threads=1, gpu_enabled=False),
        telemetry=Telemetry(runtime_sec=32.1, cpu_util_pct=14, gpu_util_pct=0),
    )

    valid_actions = ["INCREASE_MPI", "ENABLE_GPU"]
    action1 = baseline.select_action(state, valid_actions)
    action2 = baseline.select_action(state, valid_actions)

    assert action1 == "INCREASE_MPI", f"First action should be INCREASE_MPI, got {action1}"
    assert action2 == "INCREASE_MPI", f"Actions should follow pattern"
    print("[OK] GridBaseline: follows fixed order")


def test_heuristic_baseline():
    """Test HeuristicBaseline selects actions based on rules."""
    baseline = HeuristicBaseline()

    state = HPCState(
        hardware=Hardware(cpu_cores=8, cpu_freq_ghz=3.5, l3_cache_mb=24,
                         gpu_memory_gb=6.0, ram_gb=24),
        workload=Workload(name="nbody", problem_size=50000,
                         algorithm_type="embarrassingly_parallel"),
        current_config=Configuration(mpi_ranks=1, omp_threads=1, gpu_enabled=False),
        telemetry=Telemetry(runtime_sec=32.1, cpu_util_pct=85, gpu_util_pct=10),
    )

    valid_actions = ["INCREASE_MPI", "ENABLE_GPU", "INCREASE_OMP"]
    action = baseline.select_action(state, valid_actions)

    # Should select a valid action based on heuristic rules
    assert action in valid_actions, f"Expected valid action, got {action}"
    print("[OK] HeuristicBaseline: applies domain rules")


if __name__ == "__main__":
    test_random_baseline()
    test_grid_baseline()
    test_heuristic_baseline()
    print("\n[OK] All baseline tests passed!")
