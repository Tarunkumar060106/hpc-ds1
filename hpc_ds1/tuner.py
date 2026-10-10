"""Closed-loop tuning with confidence-based escalation (the Jev workflow).

    decide -> confident? run it      -> faster? keep it
           -> not sure?  ask a human (on_escalate) or stop
           -> STOP?      done

A runner turns a config into telemetry:
  LiveRunner   runs the kernel on this machine right now (the real thing)
  TableRunner  replays measurements already collected on this machine, so
               evaluation can try thousands of sessions without hours of runs
Both return None when a config fails or produces a wrong answer.
"""

import json
import os
import random
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from .kernels import KERNELS
from .runner import Engine
from .space import STOP, apply_action, config_key, make_config, valid_actions


@dataclass
class TuneStep:
    step: int
    action: str
    confidence: float
    escalated: bool
    runtime_before: float
    runtime_after: Optional[float]
    accepted: bool
    note: str = ""


@dataclass
class TuneResult:
    workload: str
    size: int
    initial_config: dict
    final_config: dict
    initial_runtime: float
    final_runtime: float
    experiments: int
    escalations: int
    stopped_reason: str
    steps: List[TuneStep] = field(default_factory=list)

    @property
    def speedup(self):
        return self.initial_runtime / max(self.final_runtime, 1e-12)

    def to_dict(self):
        d = asdict(self)
        d["speedup"] = round(self.speedup, 4)
        return d


def build_state(hardware, workload, size, config, telemetry):
    return {"hardware": dict(hardware), "workload": {"name": workload, "size": int(size)},
            "config": make_config(**config), "telemetry": dict(telemetry)}


_LOG_LOCK = threading.Lock()


def live_log_path():
    """Where real runs made while using HPC-DS1 are kept (System 2 teaching System 1)."""
    home = Path(os.environ.get("HPC_DS1_HOME", Path.home() / ".hpc_ds1"))
    return home / "data" / "live_runs.jsonl"


def log_live_run(hardware, workload, size, config, result, err=None, error=None):
    """Append one real run, in the same format as the collected measurements.
    Off with HPC_DS1_LOG_LIVE=0. Never lets a logging problem break a job."""
    if os.environ.get("HPC_DS1_LOG_LIVE", "1") == "0":
        return
    row = {"workload": workload, "size": int(size), "config": make_config(**config), "ok": result["ok"],
           "hardware": hardware.get("name"), "source": "live",
           "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if result["ok"]:
        row.update(telemetry=result["telemetry"], rel_error=err,
                   correct=err is not None and err <= KERNELS[workload].tolerance)
    else:
        row["error"] = error or result.get("error")
    try:
        path = live_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK, open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass


class LiveRunner:
    """Runs configs for real and rejects results that disagree with fp32.
    Every run is also logged for `hpc-ds1 learn` (see log_live_run)."""

    def __init__(self, workload, size, engine=None):
        self.workload, self.size = workload, size
        self.engine = engine or Engine()
        self.hardware = self.engine.hardware
        self._ref = None

    def reference(self):
        if self._ref is None:
            r = self.engine.measure(self.workload, self.size,
                                    {"ranks": 1, "threads": self.hardware["cpu_threads"]}, min_total=0.0, max_reps=1)
            if not r["ok"]:
                raise RuntimeError(r["error"])
            self._ref = r["result"]
        return self._ref

    def __call__(self, config):
        r = self.engine.measure(self.workload, self.size, config)
        if not r["ok"]:
            log_live_run(self.hardware, self.workload, self.size, config, r)
            return None
        a = np.asarray(r["result"]).astype(np.complex128 if np.iscomplexobj(r["result"]) else np.float64)
        b = np.asarray(self.reference()).astype(a.dtype)
        err = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
        log_live_run(self.hardware, self.workload, self.size, config, r, err)
        if err > KERNELS[self.workload].tolerance:
            return None
        return r["telemetry"]


class TableRunner:
    """Replays real measurements: {config_key: telemetry} for one problem."""

    def __init__(self, table, hardware):
        self.table, self.hardware = table, hardware

    def __call__(self, config):
        return self.table.get(config_key(make_config(**config)))


def _ask(policy, state, exclude):
    try:
        return policy(state, exclude=frozenset(exclude))
    except TypeError:
        return policy(state)


def run_tuning(policy, workload, size, initial_config, runner, max_steps=8, threshold=0.0,
               on_escalate=None, stop_on_reject=True, min_gain=0.0, max_rejects=2):
    """Drive a policy through a tuning session.

    policy(state, exclude=()) -> (action, confidence, extra_experiments[, needs_human])
    runner(config) -> telemetry dict, or None if the config fails / is wrong
    on_escalate(action, confidence, state) -> bool when a human must approve: the
        policy's needs_human if it returns one, else confidence < threshold

    A move that turns out slower (or fails) is reverted and excluded; the policy
    is asked again for this config. After `max_rejects` misses in a row the
    session ends (stop_on_reject=False never ends early: the random baseline).
    """
    hw = runner.hardware
    config = make_config(**initial_config)
    telemetry = runner(config)
    if telemetry is None:
        raise ValueError(f"initial config {config} does not run correctly for {workload} {size}")
    experiments, escalations = 1, 0
    initial_runtime = telemetry["runtime_sec"]
    steps, reason = [], "max_steps"
    rejected = set()

    for step in range(max_steps):
        state = build_state(hw, workload, size, config, telemetry)
        action, confidence, extra, *needs_human = _ask(policy, state, rejected)
        experiments += extra
        if action == STOP:
            steps.append(TuneStep(step, action, confidence, False, telemetry["runtime_sec"], None, True))
            reason = "model_stop"
            break
        escalated = needs_human[0] if needs_human else confidence < threshold
        if escalated:
            escalations += 1
            if on_escalate is None or not on_escalate(action, confidence, state):
                steps.append(TuneStep(step, action, confidence, True, telemetry["runtime_sec"], None, False,
                                      "escalated, not approved"))
                reason = "escalation_declined"
                break
        candidate = apply_action(config, action)
        new_tel = runner(candidate) if candidate is not None else None
        experiments += 1
        if new_tel is None:
            steps.append(TuneStep(step, action, confidence, escalated, telemetry["runtime_sec"], None, False,
                                  "failed or wrong result"))
            rejected.add(action)
            if stop_on_reject and len(rejected) >= max_rejects:
                reason = "action_failed"
                break
            continue
        accepted = new_tel["runtime_sec"] < telemetry["runtime_sec"] * (1 - min_gain)
        steps.append(TuneStep(step, action, confidence, escalated, telemetry["runtime_sec"],
                              new_tel["runtime_sec"], accepted))
        if accepted:
            config, telemetry = candidate, new_tel
            rejected.clear()
        else:
            rejected.add(action)
            if stop_on_reject and len(rejected) >= max_rejects:
                reason = "no_improvement"
                break

    return TuneResult(workload=workload, size=int(size), initial_config=make_config(**initial_config),
                      final_config=config, initial_runtime=initial_runtime,
                      final_runtime=telemetry["runtime_sec"], experiments=experiments,
                      escalations=escalations, stopped_reason=reason, steps=steps)


# ----------------------------------------------------------------------------
# Baseline policies (same signature as the model policy)
# ----------------------------------------------------------------------------

def heuristic_policy(state, exclude=()):
    """Rules of thumb an HPC engineer might script, reading the same telemetry."""
    c, t, hw = state["config"], state["telemetry"], state["hardware"]
    valid = set(valid_actions(c, bool(hw.get("gpu_model")))) - set(exclude)
    total = c["ranks"] * c["threads"]
    rules = []
    if not c["gpu"] and t["runtime_sec"] > 0.02:
        rules.append("USE_GPU")
    if c["gpu"] and c["precision"] == "fp32":
        rules.append("USE_BF16")
    if t["comm_overhead_pct"] > 40 and c["ranks"] > 1:
        rules.append("REMOVE_RANKS")
    if not c["gpu"] and total > hw["cpu_threads"]:
        rules.append("REMOVE_THREADS")
    if not c["gpu"] and total < hw["cpu_threads"]:
        rules.append("ADD_THREADS" if t["comm_overhead_pct"] > 15 else "ADD_RANKS")
    for a in rules:
        if a in valid:
            return a, 1.0, 0
    return STOP, 1.0, 0


def make_random_policy(seed=0):
    rng = random.Random(seed)

    def policy(state, exclude=()):
        valid = [a for a in valid_actions(state["config"], bool(state["hardware"].get("gpu_model"))) if a not in exclude]
        return (rng.choice(valid), 1.0, 0) if valid else (STOP, 1.0, 0)
    return policy


def make_greedy_search_policy(runner, min_gain=0.05):
    """Tries every neighbouring config each step and keeps the best: what a
    careful engineer does by hand. Pays one experiment per neighbour."""
    def policy(state, exclude=()):
        c = state["config"]
        base = state["telemetry"]["runtime_sec"]
        results = {}
        for a in valid_actions(c, bool(state["hardware"].get("gpu_model"))):
            tel = runner(apply_action(c, a))
            if tel is not None:
                results[a] = tel["runtime_sec"]
        n = len(valid_actions(c, bool(state["hardware"].get("gpu_model"))))
        if not results:
            return STOP, 1.0, n
        best = min(results, key=results.get)
        if results[best] >= base * (1 - min_gain):
            return STOP, 1.0, n
        return best, 1.0, n - 1
    return policy
