"""`hpc-ds1 learn`: System 2 teaches System 1.

Every real run made while HPC-DS1 is used on this machine (tuning sessions,
`plan`, fleet tasks, the playground) is logged to ~/.hpc_ds1/data/live_runs.jsonl
by the tuner (tuner.log_live_run). `learn` retrains System 1 on the collected
measurements plus those live runs, so jobs the facility actually runs, at the
sizes it actually uses, become familiar and better predicted over time.

Only runs from this machine are used (matched by hardware name). The
confidence calibration is kept from the last full evaluation
(scripts/train_system1.py); rerun that to re-measure accuracy.
"""

import json
from datetime import datetime, timezone

import torch

from .api import home_dir
from .dataset import data_dir, load_hardware, load_measurements
from .features import _hw
from .kernels import WORKLOAD_NAMES
from .system1 import (DEFAULT_HP, Calibrator, System1, rows_from_measurements, size_ranges, train_ensemble,
                      train_sizes_of)
from .tuner import live_log_path


def load_live_runs(hardware_name, path=None):
    path = path or live_log_path()
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("hardware") == hardware_name and r.get("workload") in WORKLOAD_NAMES:
                out.append(r)
    return out


def learn(previous=None, device=None, log=print):
    """Retrain System 1 on measurements + live runs. previous: the System1 in use (for its
    calibrator and recipe). Returns a report dict and saves ~/.hpc_ds1/model/system1.pt."""
    hardware = load_hardware(data_dir() / "hardware.json")
    base = load_measurements(data_dir() / "measurements.jsonl")
    live = load_live_runs(hardware.get("name"))
    base_problems = {(r["workload"], r["size"]) for r in base}
    live_problems = {(r["workload"], r["size"]) for r in live}
    log(f"{len(base)} collected runs + {len(live)} live runs from this machine "
        f"({len(live_problems)} problems, {len(live_problems - base_problems)} never collected before)")
    rows = base + live
    hp = {**DEFAULT_HP, **(getattr(previous, "hp", None) or {})}
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x, y, f, g = rows_from_measurements(rows, hardware)
    members = train_ensemble(x, y, f, g, device, hp)
    cal = previous.calibrator if previous is not None else Calibrator()
    s1 = System1([m.cpu() for m in members], WORKLOAD_NAMES, cal, size_ranges(rows), _hw(hardware),
                 hardware.get("name"), train_sizes=train_sizes_of(rows), hp=hp)
    out = home_dir() / "model" / "system1.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(s1.to_checkpoint(), out)
    report = {"learned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "machine": hardware.get("name"),
              "collected_runs": len(base), "live_runs": len(live),
              "live_problems": sorted(f"{w}:{s}" for w, s in live_problems),
              "new_problems": sorted(f"{w}:{s}" for w, s in live_problems - base_problems), "saved": str(out)}
    (home_dir() / "learn_report.json").write_text(json.dumps(report, indent=2))
    log(f"saved {out}; hpc_ds1.load() now uses it")
    return report
