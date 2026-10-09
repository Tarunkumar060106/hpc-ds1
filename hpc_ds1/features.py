"""State dict -> model tensors, plus the action mask.

A state is what you know about a job after running it once:
    {"hardware": {...}, "workload": {"name", "size"},
     "config": {"ranks", "threads", "gpu", "precision"},
     "telemetry": {"runtime_sec", "cpu_util_pct", "gpu_busy_pct",
                   "comm_overhead_pct", "memory_gb", "noise_pct"}}
Every field is measured or detected on the machine (runner.py).
"""

import math

import torch

from .kernels import KERNELS, WORKLOAD_NAMES
from .space import ACTIONS, STOP, valid_actions

FEATURE_VERSION = 5  # 5: eight workloads (adds mltrain, server)
STOP_INDEX = ACTIONS.index(STOP)

HW_FEATURES = ["cpu_cores", "cpu_threads", "cpu_freq_ghz", "ram_gb", "gpu_present", "gpu_memory_gb"]
def workload_features(workloads):
    """Workload inputs for a model trained on `workloads` (its vocabulary is
    stored in the checkpoint, so tasks added later don't change its inputs)."""
    return ["log_ops", "log_size"] + [f"is_{w}" for w in workloads]


def module_features(workloads):
    return {"hardware": HW_FEATURES, "workload": workload_features(workloads), "config": CFG_FEATURES,
            "telemetry": TEL_FEATURES}


WL_FEATURES = workload_features(WORKLOAD_NAMES)
CFG_FEATURES = ["log2_ranks", "log2_threads", "gpu", "bf16", "log2_load"]
# memory_gb and noise_pct are recorded but not model inputs: both depend on what the
# measuring process did before (allocator caches, warm-up), not on the job itself, and
# made identical jobs look different live vs. during collection.
TEL_FEATURES = ["log_runtime", "cpu_util", "gpu_busy", "comm_overhead"]

MODULES = {"hardware": HW_FEATURES, "workload": WL_FEATURES, "config": CFG_FEATURES,
           "telemetry": TEL_FEATURES}
MODULE_ORDER = ["hardware", "workload", "config", "telemetry"]


def _hw(hw):
    return [hw.get("cpu_cores", 8) / 64.0, hw.get("cpu_threads", 16) / 128.0,
            hw.get("cpu_freq_ghz", 3.0) / 5.0, hw.get("ram_gb", 16) / 512.0,
            1.0 if hw.get("gpu_model") else 0.0, hw.get("gpu_memory_gb", 0.0) / 80.0]


def _wl(wl, workloads=WORKLOAD_NAMES):
    name, size = wl["name"], wl["size"]
    one_hot = [1.0 if name == w else 0.0 for w in workloads]  # all zero: a workload the model never saw
    ops = KERNELS[name].work(size) if name in KERNELS else 1.0
    return [math.log10(ops) / 12.0, math.log10(size) / 7.0] + one_hot


def _cfg(c, hw):
    ranks = 1 if c.get("gpu") else c["ranks"]
    load = ranks * c["threads"] / max(1, hw.get("cpu_threads", 16))
    return [math.log2(ranks) / 3.0, math.log2(c["threads"]) / 4.0, 1.0 if c.get("gpu") else 0.0,
            1.0 if c.get("precision") == "bf16" else 0.0, math.log2(load) / 3.0]


def _tel(t, hw):
    return [math.log10(max(t["runtime_sec"], 1e-5)) / 2.0, t.get("cpu_util_pct", 0) / 100.0,
            t.get("gpu_busy_pct", 0) / 100.0, t.get("comm_overhead_pct", 0) / 100.0]


def valid_action_mask(state):
    has_gpu = bool(state["hardware"].get("gpu_model"))
    valid = set(valid_actions(state["config"], has_gpu))
    return [a in valid for a in ACTIONS[:-1]] + [True]


def featurize(states, device="cpu", workloads=None):
    workloads = list(workloads or WORKLOAD_NAMES)
    rows = {m: [] for m in MODULE_ORDER}
    masks = []
    for s in states:
        hw = s["hardware"]
        rows["hardware"].append(_hw(hw))
        rows["workload"].append(_wl(s["workload"], workloads))
        rows["config"].append(_cfg(s["config"], hw))
        rows["telemetry"].append(_tel(s["telemetry"], hw))
        masks.append(valid_action_mask(s))
    out = {m: torch.tensor(rows[m], dtype=torch.float32, device=device) for m in MODULE_ORDER}
    out["mask"] = torch.tensor(masks, dtype=torch.bool, device=device)
    return out
