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


def record(engine, workload, size, cfg, ref, min_total=0.3):
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
    return row


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
    for workload in (workloads or list(KERNELS)):
        kern = KERNELS[workload]
        for size in (quick_sizes(kern) if quick else kern.sizes):
            todo = [c for c in configs if (workload, size, config_key(c)) not in done]
            if not todo:
                continue
            t0 = time.time()
            ref_run = engine.measure(workload, size, {"ranks": 1, "threads": engine.hardware["cpu_threads"]},
                                     min_total=0.0, max_reps=1)
            if not ref_run["ok"]:
                log(f"reference failed for {workload} {size}: {ref_run['error']}")
                continue
            ok = 0
            with meas.open("a") as f:
                for cfg in todo:
                    row = record(engine, workload, size, cfg, ref_run["result"], min_total)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    ok += row["ok"]
                    written += 1
            log(f"{workload:10s} {size:>9}: {ok}/{len(todo)} ok in {time.time() - t0:5.1f}s "
                f"(total {time.time() - t_all:6.0f}s)")
    return written
