"""Backwards-compatible entry point; the implementation lives in api.py."""

from .api import DEFAULT_THRESHOLD, HPCDS1


def select_next_experiment(state, model=None, confidence_threshold=DEFAULT_THRESHOLD, **_):
    """Return the next action for a state dict (see HPCDS1.decide)."""
    model = model or HPCDS1.load()
    return model.decide(state, threshold=confidence_threshold).to_dict()
