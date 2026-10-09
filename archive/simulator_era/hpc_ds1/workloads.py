"""
Multi-workload synthetic HPC performance model.

Replaces the old single-kernel N-body simulator, which had two bugs:
  1. A `max(2.0, ...)` runtime floor that every config fell below, so all
     36 baseline configs collapsed to an identical 2.00s runtime.
  2. Best-action selection tie-broke to whichever action was listed first
     (INCREASE_MPI), so every trajectory got the same label regardless of
     the actual configuration.

This module models 6 HPC kernels with genuinely different scaling laws so
that different tuning actions are optimal in different scenarios:

  nbody       - embarrassingly parallel, O(N^2) compute, huge GPU win, low comm
  gemm        - dense matmul, O(N^3) compute, huge GPU win, low comm
  montecarlo  - embarrassingly parallel, near-zero comm, moderate GPU win
  stencil     - PDE/halo-exchange, memory-bound, halo comm (grows ~sqrt(P))
  spmv        - sparse matvec, memory-bound, irregular comm (grows ~P)
  fft         - all-to-all transpose, communication-bound (grows ~P log P)

Runtime model (Amdahl's law + communication term + GPU term):

  W                = base_work_constant * problem_size ** work_exponent   # serial work, "seconds"
  parallel_eff     = serial_fraction + (1 - serial_fraction) / (mpi_ranks * omp_eff)
  compute_time     = W * parallel_eff  [/ gpu_speedup if GPU helps here]
  comm_time        = comm_model(mpi_ranks)
  total            = compute_time + comm_time  (+ small deterministic noise)

Because comm_time grows with mpi_ranks while compute_time shrinks, each
workload has a genuine sweet spot — over-provisioning MPI ranks can make
DECREASE_MPI the correct label, not just INCREASE_MPI.
"""

import math
import hashlib

ACTION_NAMES = ["INCREASE_MPI", "INCREASE_OMP", "ENABLE_GPU", "DISABLE_GPU",
                "DECREASE_MPI", "CHANGE_BLOCK_SIZE", "REDUCE_PRECISION", "PREFETCH"]

BOTTLENECK_NAMES = ["COMPUTE", "MEMORY", "COMMUNICATION", "IO"]


WORKLOADS = {
    "nbody": {
        "display_name": "N-body simulation",
        "work_exponent": 2.0,          # O(N^2) all-pairs
        "work_constant": 4.0e-10,
        "serial_fraction": 0.02,
        "comm_model": "none",          # only a final reduction
        "comm_base": 0.002,
        "memory_bound": False,
        "gpu_speedup_range": (14.0, 22.0),
        "gpu_min_problem": 5_000,
        "problem_sizes": [20_000, 50_000, 100_000, 200_000],
    },
    "gemm": {
        "display_name": "Dense matrix multiply",
        "work_exponent": 3.0,          # O(N^3)
        "work_constant": 2.0e-11,
        "serial_fraction": 0.01,
        "comm_model": "halo",          # blocked/tiled comm between ranks
        "comm_base": 0.01,
        "memory_bound": False,
        "gpu_speedup_range": (20.0, 38.0),
        "gpu_min_problem": 512,
        "problem_sizes": [512, 1024, 2048, 4096],
    },
    "montecarlo": {
        "display_name": "Monte Carlo integration",
        "work_exponent": 1.0,          # O(N) samples
        "work_constant": 6.0e-8,
        "serial_fraction": 0.01,
        "comm_model": "none",          # only a final reduction
        "comm_base": 0.001,
        "memory_bound": False,
        "gpu_speedup_range": (5.0, 9.0),
        "gpu_min_problem": 100_000,
        "problem_sizes": [500_000, 2_000_000, 8_000_000, 20_000_000],
    },
    "stencil": {
        "display_name": "3D stencil / PDE solver",
        "work_exponent": 1.0,          # O(N) grid points per timestep
        "work_constant": 3.0e-7,
        "serial_fraction": 0.02,
        "comm_model": "halo",          # nearest-neighbor halo exchange
        "comm_base": 0.03,
        "memory_bound": True,
        "gpu_speedup_range": (9.0, 16.0),
        "gpu_min_problem": 200_000,
        "problem_sizes": [500_000, 1_000_000, 4_000_000, 8_000_000],
    },
    "spmv": {
        "display_name": "Sparse matrix-vector multiply",
        "work_exponent": 1.05,         # ~O(nnz), slightly superlinear via irregular access
        "work_constant": 5.0e-8,
        "serial_fraction": 0.04,
        "comm_model": "alltoall",      # irregular partitioning needs wide exchange
        "comm_base": 0.045,
        "memory_bound": True,
        "gpu_speedup_range": (2.5, 4.5),
        "gpu_min_problem": 1_000_000,
        "problem_sizes": [1_000_000, 4_000_000, 10_000_000, 25_000_000],
    },
    "fft": {
        "display_name": "Distributed 3D FFT",
        "work_exponent": 1.15,         # ~O(N log N), approximated
        "work_constant": 1.2e-7,
        "serial_fraction": 0.03,
        "comm_model": "alltoall",      # global transpose
        "comm_base": 0.06,
        "memory_bound": False,
        "gpu_speedup_range": (7.0, 12.0),
        "gpu_min_problem": 500_000,
        "problem_sizes": [500_000, 2_000_000, 6_000_000, 16_000_000],
    },
}

WORKLOAD_NAMES = list(WORKLOADS.keys())


def _deterministic_noise(*key_parts, spread=0.02):
    """Small reproducible noise derived from a hash of the config, not a
    reset global seed — each distinct config gets its own stable jitter
    instead of every call replaying the same random draw."""
    key = "|".join(str(k) for k in key_parts)
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    frac = (h % 10_000) / 10_000.0  # in [0, 1)
    return 1.0 + spread * (2 * frac - 1.0)  # in [1-spread, 1+spread]


def _omp_efficiency(omp_threads, memory_bound):
    """Effective parallel threads after diminishing returns.

    Memory-bound kernels saturate memory bandwidth well before core count,
    so extra OMP threads beyond a small threshold barely help (and can
    even mildly hurt from contention) — this is what makes INCREASE_OMP
    a bad move for stencil/spmv once threads are already >= ~4.
    """
    if not memory_bound:
        return float(omp_threads)
    threshold = 4.0
    if omp_threads <= threshold:
        return float(omp_threads)
    # beyond threshold: strongly diminishing returns
    return threshold + (omp_threads - threshold) ** 0.35


def _comm_time(comm_model, comm_base, mpi_ranks):
    if mpi_ranks <= 1:
        return comm_base * 0.05
    if comm_model == "none":
        return comm_base * math.log2(mpi_ranks + 1)
    if comm_model == "halo":
        return comm_base * (mpi_ranks ** 0.5)
    if comm_model == "alltoall":
        return comm_base * mpi_ranks * math.log2(mpi_ranks + 1)
    return comm_base


def _gpu_factor(workload, gpu_enabled, problem_size, cuda_block_size):
    """Multiplicative factor applied to compute_time (< 1 = speedup)."""
    if not gpu_enabled:
        return 1.0

    spec = WORKLOADS[workload]
    if problem_size < spec["gpu_min_problem"]:
        # Not enough work to amortize kernel launch / transfer overhead.
        return 1.25

    lo, hi = spec["gpu_speedup_range"]
    # Bigger problems get closer to the high end of the speedup range.
    ratio = min(1.0, problem_size / (spec["gpu_min_problem"] * 20))
    speedup = lo + (hi - lo) * ratio

    # Block size matters: 128-256 is close to optimal, far from it costs ~10-20%.
    block_size = cuda_block_size or 128
    optimal = 128.0
    block_penalty = 1.0 + 0.15 * abs(math.log2(max(block_size, 1) / optimal)) / 3.0

    return block_penalty / speedup


def compute_runtime(workload, mpi_ranks, omp_threads, gpu_enabled, problem_size,
                     cuda_block_size=None, reduced_precision=False, prefetch=False,
                     hardware=None):
    """Compute a realistic synthetic runtime (seconds) for one configuration.

    `hardware=None` keeps the original single-laptop behaviour (no core cap).
    Passing a hardware profile dict adds clock scaling, a core-count cap with
    oversubscription contention, and GPU-speed scaling.
    """
    spec = WORKLOADS[workload]

    mpi_ranks = max(1, int(mpi_ranks))
    omp_threads = max(1, int(omp_threads))

    W = spec["work_constant"] * (problem_size ** spec["work_exponent"])

    omp_eff = _omp_efficiency(omp_threads, spec["memory_bound"])
    f = spec["serial_fraction"]
    parallelism = mpi_ranks * omp_eff
    contention = 1.0
    if hardware is not None:
        W *= 3.5 / hardware["cpu_freq_ghz"]
        cores = hardware["cpu_cores"]
        threads = mpi_ranks * omp_threads
        parallelism = min(parallelism, float(cores))
        # Oversubscribed ranks/threads time-slice and thrash caches.
        contention = 1.0 + 0.12 * max(0.0, threads / cores - 1.0)
    parallel_eff = f + (1 - f) / parallelism
    compute_time = W * parallel_eff * contention

    gpu_factor = _gpu_factor(workload, gpu_enabled, problem_size, cuda_block_size)
    if hardware is not None and gpu_enabled and gpu_factor < 1.0:
        gpu_factor /= max(hardware["gpu_relative_speed"], 0.1) ** 0.8
    compute_time *= gpu_factor

    if reduced_precision:
        compute_time *= 0.72  # fp16/mixed precision speedup, applied uniformly

    if prefetch and spec["memory_bound"]:
        compute_time *= 0.90  # prefetching helps memory-bound kernels specifically
    elif prefetch:
        compute_time *= 0.98  # negligible for compute-bound kernels

    comm_time = _comm_time(spec["comm_model"], spec["comm_base"], mpi_ranks)

    total = compute_time + comm_time
    noise = _deterministic_noise(workload, mpi_ranks, omp_threads, gpu_enabled,
                                  problem_size, cuda_block_size, reduced_precision, prefetch)
    total *= noise

    # A realistic floor (kernel launch / IO / measurement overhead), far
    # below anything actions need to differentiate against.
    total = max(0.005, min(3600.0, total))
    return total, compute_time * noise, comm_time * noise


def estimate_telemetry(workload, mpi_ranks, omp_threads, gpu_enabled, problem_size,
                        compute_time, comm_time, hardware=None):
    """Derive plausible utilization telemetry from the runtime breakdown."""
    spec = WORKLOADS[workload]
    total = max(compute_time + comm_time, 1e-9)

    comm_frac = comm_time / total
    cpu_util = 92.0 if not gpu_enabled else 45.0
    cpu_util = max(15.0, cpu_util - comm_frac * 40.0)
    cpu_util = min(99.0, cpu_util + (omp_threads - 1) * 1.0)
    if hardware is not None:
        # Whole-node utilization: 1 thread on a 64-core node reads ~1.5%.
        occupancy = min(1.0, mpi_ranks * omp_threads / hardware["cpu_cores"])
        cpu_util = min(99.0, max(1.0, cpu_util * occupancy))

    gpu_util = 0.0
    if gpu_enabled:
        gpu_util = min(97.0, 55.0 + 30.0 * min(1.0, problem_size / (spec["gpu_min_problem"] * 10 + 1)))

    mem_util = 40.0 + (25.0 if spec["memory_bound"] else 5.0) + min(30.0, omp_threads * 2.0)
    mem_util = min(96.0, mem_util)

    mpi_comm_pct = min(85.0, comm_frac * 100.0)

    return {
        "cpu_util_pct": round(cpu_util, 1),
        "gpu_util_pct": round(gpu_util, 1),
        "memory_util_pct": round(mem_util, 1),
        "mpi_comm_overhead_pct": round(mpi_comm_pct, 1),
    }


def classify_bottleneck(workload, mpi_ranks, comm_time, compute_time, memory_util_pct):
    """Ground-truth-ish bottleneck label used as an auxiliary training signal."""
    total = compute_time + comm_time
    comm_frac = comm_time / max(total, 1e-9)

    if comm_frac > 0.4:
        return "COMMUNICATION"
    if WORKLOADS[workload]["memory_bound"] and memory_util_pct > 70:
        return "MEMORY"
    return "COMPUTE"


def apply_action(config, action, workload):
    """Return a new config dict with `action` applied, respecting bounds."""
    spec = WORKLOADS[workload]
    c = dict(config)

    if action == "INCREASE_MPI":
        c["mpi_ranks"] = min(64, c["mpi_ranks"] * 2)
    elif action == "DECREASE_MPI":
        c["mpi_ranks"] = max(1, c["mpi_ranks"] // 2)
    elif action == "INCREASE_OMP":
        c["omp_threads"] = min(32, c["omp_threads"] * 2)
    elif action == "ENABLE_GPU":
        c["gpu_enabled"] = True
        if not c.get("cuda_block_size"):
            c["cuda_block_size"] = 128
    elif action == "DISABLE_GPU":
        c["gpu_enabled"] = False
    elif action == "CHANGE_BLOCK_SIZE":
        current = c.get("cuda_block_size") or 128
        # Move toward the optimum (128) if far from it, else try 256.
        c["cuda_block_size"] = 128 if current != 128 else 256
    elif action == "REDUCE_PRECISION":
        c["reduced_precision"] = True
    elif action == "PREFETCH":
        c["prefetch"] = True

    return c


def valid_actions_for(config, hardware=None):
    """Actions that make sense given the current config (avoid no-op actions)."""
    has_gpu = hardware is None or bool(hardware.get("gpu_model"))
    valid = []
    for a in ACTION_NAMES:
        if a == "ENABLE_GPU" and (config.get("gpu_enabled") or not has_gpu):
            continue
        if a == "INCREASE_MPI" and config.get("mpi_ranks", 1) >= 64:
            continue
        if a == "INCREASE_OMP" and config.get("omp_threads", 1) >= 32:
            continue
        if a == "DISABLE_GPU" and not config.get("gpu_enabled"):
            continue
        if a == "CHANGE_BLOCK_SIZE" and not config.get("gpu_enabled"):
            continue
        if a == "REDUCE_PRECISION" and config.get("reduced_precision"):
            continue
        if a == "PREFETCH" and config.get("prefetch"):
            continue
        if a == "DECREASE_MPI" and config.get("mpi_ranks", 1) <= 1:
            continue
        valid.append(a)
    return valid


# ============================================================================
# Hardware profiles + "run a job" simulation helpers
# ============================================================================

# gpu_relative_speed is a synthetic knob of this simulator (1.0 = the RTX 4050
# laptop this project was built on), not a published benchmark figure.
HARDWARE_PROFILES = {
    "laptop-rtx4050": {
        "cpu_cores": 8, "cpu_freq_ghz": 3.5, "l3_cache_mb": 24, "ram_gb": 24,
        "gpu_model": "RTX 4050 Laptop", "gpu_memory_gb": 6.0, "gpu_relative_speed": 1.0,
    },
    "workstation-rtx4090": {
        "cpu_cores": 24, "cpu_freq_ghz": 4.2, "l3_cache_mb": 36, "ram_gb": 128,
        "gpu_model": "RTX 4090", "gpu_memory_gb": 24.0, "gpu_relative_speed": 2.2,
    },
    "hpc-node-a100": {
        "cpu_cores": 64, "cpu_freq_ghz": 2.6, "l3_cache_mb": 256, "ram_gb": 512,
        "gpu_model": "A100 80GB", "gpu_memory_gb": 80.0, "gpu_relative_speed": 2.8,
    },
    "cpu-node-64c": {
        "cpu_cores": 64, "cpu_freq_ghz": 2.8, "l3_cache_mb": 256, "ram_gb": 256,
        "gpu_model": None, "gpu_memory_gb": 0.0, "gpu_relative_speed": 0.0,
    },
    "small-vm-4c": {
        "cpu_cores": 4, "cpu_freq_ghz": 3.0, "l3_cache_mb": 8, "ram_gb": 16,
        "gpu_model": None, "gpu_memory_gb": 0.0, "gpu_relative_speed": 0.0,
    },
}

HARDWARE_NAMES = list(HARDWARE_PROFILES.keys())


def resolve_hardware(hardware):
    """Accept a profile name or a hardware dict; return a complete dict."""
    if isinstance(hardware, str):
        if hardware not in HARDWARE_PROFILES:
            raise ValueError(f"unknown hardware profile {hardware!r}; known: {HARDWARE_NAMES}")
        return {"name": hardware, **HARDWARE_PROFILES[hardware]}
    hw = dict(hardware)
    base = HARDWARE_PROFILES.get(hw.get("name", ""), HARDWARE_PROFILES["laptop-rtx4050"])
    for k, v in base.items():
        hw.setdefault(k, v)
    hw.setdefault("name", "custom")
    return hw


def normalize_config(config):
    return {
        "mpi_ranks": int(config.get("mpi_ranks", 1)),
        "omp_threads": int(config.get("omp_threads", 1)),
        "gpu_enabled": bool(config.get("gpu_enabled", False)),
        "cuda_block_size": config.get("cuda_block_size") if config.get("gpu_enabled") else None,
        "reduced_precision": bool(config.get("reduced_precision", False)),
        "prefetch": bool(config.get("prefetch", False)),
    }


def measure(workload, problem_size, config, hardware):
    """Simulated job run: returns (telemetry dict, bottleneck label).

    Stands in for a real launcher (Slurm job, mpirun, ...) that runs the
    config and reports runtime + utilization counters.
    """
    hw = resolve_hardware(hardware)
    c = normalize_config(config)
    rt, comp, comm = compute_runtime(
        workload, c["mpi_ranks"], c["omp_threads"], c["gpu_enabled"], problem_size,
        c["cuda_block_size"], c["reduced_precision"], c["prefetch"], hardware=hw,
    )
    tel = estimate_telemetry(workload, c["mpi_ranks"], c["omp_threads"], c["gpu_enabled"],
                             problem_size, comp, comm, hardware=hw)
    bottleneck = classify_bottleneck(workload, c["mpi_ranks"], comm, comp, tel["memory_util_pct"])
    return {"runtime_sec": round(rt, 6), **tel}, bottleneck


def build_state(workload, problem_size, config, hardware, telemetry=None):
    """Assemble the state dict HPC-DS1 consumes. Telemetry defaults to a simulated run."""
    hw = resolve_hardware(hardware)
    c = normalize_config(config)
    if telemetry is None:
        telemetry, _ = measure(workload, problem_size, c, hw)
    spec = WORKLOADS[workload]
    return {
        "hardware": hw,
        "workload": {
            "name": workload,
            "problem_size": int(problem_size),
            "algorithm_type": "embarrassingly_parallel" if spec["serial_fraction"] < 0.03 else "moderate",
            "flops_estimate": float(problem_size) ** spec["work_exponent"],
        },
        "current_config": c,
        "telemetry": dict(telemetry),
    }


MPI_START_CHOICES = [1, 1, 2, 2, 4, 4, 8, 16, 32]
OMP_START_CHOICES = [1, 1, 2, 2, 4, 8, 16]
BLOCK_START_CHOICES = [32, 64, 128, 256, 512]


def sample_scenario(rng):
    """Random (workload, hardware name, starting config, problem size) — an
    untuned job as a user might first launch it."""
    workload = rng.choice(WORKLOAD_NAMES)
    hw_name = rng.choice(HARDWARE_NAMES)
    hw = HARDWARE_PROFILES[hw_name]
    spec = WORKLOADS[workload]
    gpu = bool(hw.get("gpu_model")) and rng.random() < 0.4
    config = {
        "mpi_ranks": rng.choice(MPI_START_CHOICES),
        "omp_threads": rng.choice(OMP_START_CHOICES),
        "gpu_enabled": gpu,
        "cuda_block_size": rng.choice(BLOCK_START_CHOICES) if gpu else None,
        "reduced_precision": rng.random() < 0.1,
        "prefetch": rng.random() < 0.1,
    }
    if rng.random() < 0.15:
        problem_size = max(1, spec["gpu_min_problem"] // 10)
    else:
        problem_size = rng.choice(spec["problem_sizes"])
    return workload, hw_name, config, problem_size
