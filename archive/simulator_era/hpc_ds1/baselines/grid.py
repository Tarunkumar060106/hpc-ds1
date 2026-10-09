"""Grid search baseline - explores in fixed order."""

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
