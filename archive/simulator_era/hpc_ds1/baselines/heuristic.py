"""Heuristic baseline - hand-coded domain rules."""

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
