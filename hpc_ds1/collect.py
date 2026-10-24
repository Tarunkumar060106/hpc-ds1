"""Measure configurations of the kernels on this machine (resumable).

For each (kernel, problem size) it first computes an fp32 reference result,
then runs every configuration in the space (space.py) for real and records
runtime, telemetry, and the relative error against the reference. Rows append
to <out_dir>/measurements.jsonl; anything already measured is skipped, so an
interrupted collection resumes. Used by scripts/collect.py and `hpc-ds1 setup`.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .kernels import KERNELS
from .space import all_configs, config_key


def rel_error(res, ref):
    a = np.asarray(res, dtype=np.complex128 if np.iscomplexobj(res) else np.float64)
    b = np.asarray(ref, dtype=np.complex128 if np.iscomplexobj(ref) else np.float64)
    if a.shape != b.shape:
        return float("inf")
    denom = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / denom) if denom else float(np.linalg.norm(a - b))


CPU_BUDGET_SEC = 60.0  # GPU-class workloads: CPU settings expected to run longer than this are skipped


def record(engine, workload, size, cfg, ref, min_total=0.3, skip=None):
    if skip:
        return {"workload": workload, "size": size, "config": cfg, "ok": False, "skipped": True, "error": skip,
                "hardware": engine.hardware["name"],
                "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    r = engine.measure(workload, size, cfg, min_total=min_total)
    row = {"workload": workload, "size": size, "config": cfg, "ok": r["ok"],
           "hardware": engine.hardware["name"],
           "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if r["ok"]:
        err = rel_error(r["result"], ref)
        row.update(telemetry=r["telemetry"], walls=r["walls"], capped=r["capped"],
                   rel_error=err, correct=bool(err <= KERNELS[workload].tolerance))
    else:
        row["error"] = r["error"]
        for k in ("skipped", "oom"):
            if r.get(k):
                row[k] = True
    return row


def _cpu_skip(engine, kern, size, cfg, probes):
    """For GPU-class workloads only: why to skip a CPU setting, or None to measure it.
    probes[precision] = (size, seconds) of a measured all-cores CPU run at this or a
    smaller size; a setting with fewer cores is expected to be proportionally slower."""
    if cfg["gpu"] or not getattr(kern, "gpu_class", False) or cfg["precision"] not in probes:
        return None
    psize, psec = probes[cfg["precision"]]
    cores = engine.hardware["cpu_threads"]
    est = psec * kern.work(size) / kern.work(psize) * cores / max(1, min(cores, cfg["ranks"] * cfg["threads"]))
    if est > CPU_BUDGET_SEC:
        return (f"skipped: expected ~{est:.0f}s on the CPU (all {cores} cores took {psec:.1f}s at size {psize}), "
                f"over the {CPU_BUDGET_SEC:.0f}s budget")
    return None


def quick_sizes(kernel):
    """Smallest, held-out test size, and largest: enough sizes to train on two
    and evaluate on the held-out one, in about half the time of a full run."""
    return sorted({kernel.sizes[0], kernel.test_size, kernel.sizes[-1]})


def collect(engine, out_dir, workloads=None, quick=False, log=print):
    """Measure every config of every (workload, size) not measured yet. Returns rows written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meas = out_dir / "measurements.jsonl"
    (out_dir / "hardware.json").write_text(json.dumps(engine.hardware, indent=2))
    configs = all_configs(engine.hardware["gpu_model"] is not None)
    min_total = 0.15 if quick else 0.3
    done = set()
    if meas.exists():
        with meas.open() as f:
            for line in f:
                r = json.loads(line)
                done.add((r["workload"], r["size"], config_key(r["config"])))

    written, t_all = 0, time.time()
    cores = engine.hardware["cpu_threads"]
    for workload in (workloads or list(KERNELS)):
        kern = KERNELS[workload]
        probes = {}  # precision -> (size, all-cores-equivalent CPU seconds), from real runs
        for size in (quick_sizes(kern) if quick else kern.sizes):
            # fastest CPU settings first, so the time budget is judged from real runs
            todo = sorted((c for c in configs if (workload, size, config_key(c)) not in done),
                          key=lambda c: (c["gpu"], -c["ranks"] * c["threads"]))
            if not todo:
                continue
            t0 = time.time()
            ref_run = engine.measure(workload, size, {"ranks": 1, "threads": cores}, min_total=0.0, max_reps=1)
            ref_note = ""
            if not ref_run["ok"] and engine.hardware["gpu_model"] is not None:
                # the CPU reference does not fit (e.g. not enough RAM): fp32 on the GPU is an equally exact reference
                ref_run = engine.measure(workload, size, {"ranks": 1, "threads": 1, "gpu": True, "precision": "fp32"},
                                         min_total=0.0, max_reps=1)
                ref_note = " (fp32 reference on the GPU)"
            if not ref_run["ok"]:
                log(f"reference failed for {workload} {size}: {ref_run['error']}")
                if ref_note and ref_run.get("oom"):
                    # a real attempt that ran out of memory: keep it, it is what the fleet needs to know
                    cfg = {"ranks": 1, "threads": 1, "gpu": True, "precision": "fp32"}
                    with meas.open("a") as f:
                        f.write(json.dumps({"workload": workload, "size": size, "config": cfg, "ok": False,
                                            "oom": True, "error": ref_run["error"],
                                            "hardware": engine.hardware["name"],
                                            "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                                + "\n")
                    written += 1
                continue
            if getattr(kern, "gpu_class", False) and "telemetry" in ref_run and ref_note == "":
                probes.setdefault("fp32", (size, ref_run["telemetry"]["runtime_sec"]))
            ok = skipped = 0
            with meas.open("a") as f:
                for cfg in todo:
                    row = record(engine, workload, size, cfg, ref_run["result"], min_total,
                                 skip=_cpu_skip(engine, kern, size, cfg, probes))
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    ok += row["ok"]
                    skipped += bool(row.get("skipped"))
                    written += 1
                    if row["ok"] and not cfg["gpu"] and getattr(kern, "gpu_class", False):
                        used = min(cores, cfg["ranks"] * cfg["threads"])
                        probes[cfg["precision"]] = (size, row["telemetry"]["runtime_sec"] * used / cores)
            log(f"{workload:10s} {size:>9}: {ok}/{len(todo)} ok" + (f", {skipped} skipped" if skipped else "")
                + f" in {time.time() - t0:5.1f}s{ref_note} (total {time.time() - t_all:6.0f}s)")
    return written
