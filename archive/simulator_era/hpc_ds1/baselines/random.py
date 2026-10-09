"""Random baseline - selects actions randomly."""

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
