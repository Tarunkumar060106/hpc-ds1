"""Configuration space and tuning actions — all of them real knobs.

A configuration is {ranks, threads, gpu, precision}:
  ranks      separate processes, MPI-style (1, 2, 4, 8)
  threads    OpenMP/MKL threads per rank (1, 2, 4, 8, 16)
  gpu        run on the CUDA device instead of the CPU (single rank)
  precision  "fp32" or "bf16"

CPU configs allow up to 32 total threads (2x oversubscription of the 16
logical cores), so "too many" is a reachable, measurable state.
"""

RANKS = [1, 2, 4, 8]
THREADS = [1, 2, 4, 8, 16]
PRECISIONS = ["fp32", "bf16"]
MAX_TOTAL_THREADS = 32

TUNING_ACTIONS = ["ADD_RANKS", "REMOVE_RANKS", "ADD_THREADS", "REMOVE_THREADS",
                  "USE_GPU", "USE_CPU", "USE_BF16", "USE_FP32"]
ACTIONS = TUNING_ACTIONS + ["STOP_TUNING"]
STOP = "STOP_TUNING"

ACTION_HELP = {
    "ADD_RANKS": "double the number of ranks (processes)",
    "REMOVE_RANKS": "halve the number of ranks",
    "ADD_THREADS": "double threads per rank",
    "REMOVE_THREADS": "halve threads per rank",
    "USE_GPU": "move the job to the GPU (one rank)",
    "USE_CPU": "move the job back to the CPU",
    "USE_BF16": "switch to bfloat16",
    "USE_FP32": "switch back to fp32",
    "STOP_TUNING": "keep the current configuration",
}


def make_config(ranks=1, threads=1, gpu=False, precision="fp32"):
    return {"ranks": 1 if gpu else int(ranks), "threads": int(threads), "gpu": bool(gpu),
            "precision": precision}


def config_key(c):
    return (c["ranks"], c["threads"], c["gpu"], c["precision"])


def in_space(c, has_gpu=True):
    if c["precision"] not in PRECISIONS or c["threads"] not in THREADS:
        return False
    if c["gpu"]:
        return has_gpu and c["ranks"] == 1
    return c["ranks"] in RANKS and c["ranks"] * c["threads"] <= MAX_TOTAL_THREADS


def all_configs(has_gpu=True):
    out = []
    for prec in PRECISIONS:
        for r in RANKS:
            for t in THREADS:
                c = make_config(r, t, False, prec)
                if in_space(c, has_gpu):
                    out.append(c)
        if has_gpu:
            for t in THREADS:
                out.append(make_config(1, t, True, prec))
    return out


def apply_action(c, action):
    """Config after `action`, or None if the action doesn't apply here."""
    n = dict(c)
    if action == "ADD_RANKS":
        if c["gpu"]:
            return None
        n["ranks"] = c["ranks"] * 2
    elif action == "REMOVE_RANKS":
        if c["gpu"] or c["ranks"] == 1:
            return None
        n["ranks"] = c["ranks"] // 2
    elif action == "ADD_THREADS":
        n["threads"] = c["threads"] * 2
    elif action == "REMOVE_THREADS":
        if c["threads"] == 1:
            return None
        n["threads"] = c["threads"] // 2
    elif action == "USE_GPU":
        if c["gpu"]:
            return None
        n.update(gpu=True, ranks=1)
    elif action == "USE_CPU":
        if not c["gpu"]:
            return None
        n["gpu"] = False
    elif action == "USE_BF16":
        if c["precision"] == "bf16":
            return None
        n["precision"] = "bf16"
    elif action == "USE_FP32":
        if c["precision"] == "fp32":
            return None
        n["precision"] = "fp32"
    elif action == STOP:
        return dict(c)
    else:
        raise ValueError(f"unknown action {action!r}")
    return n


def valid_actions(c, has_gpu=True):
    """Tuning actions whose result is inside the configuration space."""
    out = []
    for a in TUNING_ACTIONS:
        n = apply_action(c, a)
        if n is not None and in_space(n, has_gpu):
            out.append(a)
    return out
