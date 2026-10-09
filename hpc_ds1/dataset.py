"""Turn real measurements into labelled tuning states.

Every configuration that ran correctly on this machine becomes a state. Its
label is the first move of the best chain of measured, clearly-faster moves
(each must beat the current runtime by more than the noise floor); if no
neighbouring config is clearly faster, the label is STOP_TUNING. No runtime is
predicted or modelled: every label is derived from real measured runtimes.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

from .space import ACTIONS, STOP, apply_action, config_key, make_config, valid_actions
from .tuner import build_state

REPO_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "real"


def data_dir():
    """$KAIRO_DATA_DIR, else this machine's measurements from `hpc-ds1 setup`,
    else the measurements shipped with the source tree."""
    if os.environ.get("KAIRO_DATA_DIR"):
        return Path(os.environ["KAIRO_DATA_DIR"])
    from .api import home_dir
    own = home_dir() / "data"
    if (own / "measurements.jsonl").exists():
        return own
    return REPO_DATA_DIR

MIN_GAIN = 0.05  # an action must be >=5% faster (and beat measurement noise) to count
WASTED_RUN = 1.02  # a move that isn't clearly faster gets reverted; it only costs a run


def load_measurements(path=None):
    path = Path(path or data_dir() / "measurements.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_hardware(path=None):
    return json.loads(Path(path or data_dir() / "hardware.json").read_text())


def build_tables(rows):
    """{(workload, size): {config_key: telemetry}} for configs that ran AND were correct,
    plus {(workload, size): {config_key: reason}} for the ones that didn't."""
    good, bad = defaultdict(dict), defaultdict(dict)
    for r in rows:
        k = config_key(make_config(**r["config"]))
        inst = (r["workload"], r["size"])
        if r["ok"] and r.get("correct", False):
            good[inst][k] = r["telemetry"]
        else:
            bad[inst][k] = r.get("error") or f"wrong result (rel. error {r.get('rel_error', 0):.2g})"
    return dict(good), dict(bad)


def _threshold(tel):
    return max(MIN_GAIN, tel.get("noise_pct", 0) / 100.0)


def path_values(table, has_gpu=True):
    """Best final runtime reachable from each config by a chain of measured,
    clearly-faster moves (the tuner only keeps changes that really help).
    Computed fastest-first, so every faster neighbour is already solved."""
    values = {}
    for key in sorted(table, key=lambda k: table[k]["runtime_sec"]):
        tel = table[key]
        rt, best = tel["runtime_sec"], tel["runtime_sec"]
        cfg = make_config(*key)
        for a in valid_actions(cfg, has_gpu):
            n = config_key(apply_action(cfg, a))
            if n in table and table[n]["runtime_sec"] < rt * (1 - _threshold(tel)):
                best = min(best, values[n])
        values[key] = best
    return values


def label(config, table, has_gpu=True, values=None):
    """(best_action, eventual_improvement, {action: measured runtime or None}).

    The best action is the first step of the best chain of clearly-faster
    measured moves, not just the biggest immediate gain: a myopic label stops
    at the first local optimum (e.g. moves Monte Carlo to the GPU and stops,
    when more CPU ranks would win)."""
    values = values or path_values(table, has_gpu)
    key = config_key(config)
    tel = table[key]
    base = tel["runtime_sec"]
    outcomes, scored = {}, {}
    for a in valid_actions(config, has_gpu):
        n = config_key(apply_action(config, a))
        nxt = table.get(n)
        outcomes[a] = nxt["runtime_sec"] if nxt else None
        if nxt and nxt["runtime_sec"] < base * (1 - _threshold(tel)):
            scored[a] = (values[n], nxt["runtime_sec"])
    if scored:
        best = min(scored, key=lambda a: scored[a])
        return best, (base - values[key]) / base, outcomes
    return STOP, 0.0, outcomes


def build_states(tables, hardware):
    has_gpu = bool(hardware.get("gpu_model"))
    out = []
    for (workload, size), table in sorted(tables.items()):
        best_rt = min(t["runtime_sec"] for t in table.values())
        values = path_values(table, has_gpu)
        for key, tel in table.items():
            cfg = make_config(*key)
            action, gain, outcomes = label(cfg, table, has_gpu, values)
            reach = {a: (values[config_key(apply_action(cfg, a))] if v is not None else None)
                     for a, v in outcomes.items()}
            # Final runtime each choice leads to if tuning then continues optimally.
            thr = _threshold(tel)
            action_reach = {STOP: tel["runtime_sec"]}
            for a, v in outcomes.items():
                if v is None:
                    action_reach[a] = None
                elif v < tel["runtime_sec"] * (1 - thr):
                    action_reach[a] = min(tel["runtime_sec"], reach[a])
                else:
                    action_reach[a] = tel["runtime_sec"] * WASTED_RUN
            out.append({
                "workload": workload, "size": size,
                "state": build_state(hardware, workload, size, cfg, tel),
                "best_action": action, "best_improvement": gain,
                "outcomes": outcomes, "reachable": reach, "reachable_from_here": values[key],
                "action_reach": action_reach,
                "runtime": tel["runtime_sec"], "instance_best_runtime": best_rt,
            })
    return out


def summarize(states):
    counts = defaultdict(int)
    for s in states:
        counts[s["best_action"]] += 1
    return {a: counts[a] for a in ACTIONS if counts[a]}
