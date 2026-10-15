#!/usr/bin/env python3
"""Measure every configuration of every kernel on this machine.

For each (kernel, problem size) it first computes an fp32 reference result,
then runs every configuration in the space (hpc_ds1/space.py) for real and
records runtime, telemetry, and the relative error against the reference.
Results append to data/real/measurements.jsonl; re-running skips anything
already measured, so an interrupted collection resumes.

    python scripts/collect.py                 # full collection
    python scripts/collect.py --retest 80     # re-measure 80 random configs (repeatability)
"""

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hpc_ds1.collect import collect  # noqa: E402
from hpc_ds1.kernels import KERNELS  # noqa: E402
from hpc_ds1.runner import Engine  # noqa: E402

import os
from hpc_ds1.paths import DATA_DIR  # noqa: E402
OUT_DIR = Path(os.environ.get("KAIRO_DATA_DIR", DATA_DIR))
MEAS = OUT_DIR / "measurements.jsonl"
RETEST = OUT_DIR / "retest.jsonl"

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retest", type=int, default=0)
    ap.add_argument("--workloads", default=",".join(KERNELS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = Engine()
    print(f"{engine.hardware['cpu_model']} | {engine.hardware['cpu_cores']}C/{engine.hardware['cpu_threads']}T | "
          f"{engine.hardware['gpu_model']}", flush=True)

    if args.retest:
        rows = [json.loads(line) for line in MEAS.open()]
        rows = [r for r in rows if r["ok"] and not r.get("capped")]
        pick = random.Random(7).sample(rows, min(args.retest, len(rows)))
        with RETEST.open("a") as f:
            for i, r in enumerate(pick):
                again = engine.measure(r["workload"], r["size"], r["config"])
                if again["ok"]:
                    f.write(json.dumps({"workload": r["workload"], "size": r["size"], "config": r["config"],
                                        "first": r["telemetry"]["runtime_sec"],
                                        "second": again["telemetry"]["runtime_sec"]}) + "\n")
                print(f"retest {i + 1}/{len(pick)}", flush=True)
        engine.close()
        return

    t_all = time.time()
    collect(engine, OUT_DIR, args.workloads.split(","), log=lambda m: print(m, flush=True))
    engine.close()
    print(f"done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
