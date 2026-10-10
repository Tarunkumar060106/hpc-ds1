#!/usr/bin/env python3
"""
Choose System 1's training recipe by cross-validation.

Every variant is scored on the out-of-fold picks of the problems that are NOT
held-out test sizes, so the test sizes stay untouched for the final report in
scripts/train_system1.py. Metrics: share of picks within 5% of the best config,
geometric-mean % of best speed, failing picks, and how well the (leave-one-out
calibrated) confidence separates good picks from bad ones (AUROC; 0.5 = no
better than chance).

    python scripts/tune_system1.py    -> results/system1_tuning.json
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_system1 import auroc, cross_validate, geo  # noqa: E402
from hpc_ds1.dataset import build_tables, load_hardware, load_measurements  # noqa: E402
from hpc_ds1.kernels import KERNELS, WORKLOAD_NAMES  # noqa: E402
from hpc_ds1.system1 import Calibrator, confidence_features  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

VARIANTS = {
    "previous (5 seeds, no bagging, no ranking)": {"bag": False, "rank_w": 0.0, "members": 5},
    "+ bagging": {"bag": True, "rank_w": 0.0},
    "+ ranking loss": {"bag": False, "rank_w": 1.0},
    "+ bagging + ranking": {"bag": True, "rank_w": 1.0},
    "+ both, small net (128x2)": {"bag": True, "rank_w": 1.0, "width": 128, "depth": 2},
    "+ both, wide net (512x3)": {"bag": True, "rank_w": 1.0, "width": 512},
    "+ both, more weight decay": {"bag": True, "rank_w": 1.0, "wd": 1e-2},
    "+ both, strong ranking (x3)": {"bag": True, "rank_w": 3.0},
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = load_hardware()
    rows = load_measurements()
    tables, _ = build_tables(rows)
    test = {(w, KERNELS[w].test_size) for w in WORKLOAD_NAMES}
    out = []
    for name, hp in VARIANTS.items():
        t0 = time.time()
        oof = [r for r in cross_validate(rows, hardware, tables, device, hp, quiet=True)
               if (r["workload"], r["size"]) not in test]
        feats = [confidence_features(r["signals"]) for r in oof]
        labels = [float(r["good"]) for r in oof]
        loo = [Calibrator().fit(feats[:i] + feats[i + 1:], labels[:i] + labels[i + 1:])(feats[i]) for i in range(len(oof))]
        r = {"variant": name, "hp": hp, "problems": len(oof),
             "within_5pct": round(float(np.mean(labels)), 4),
             "pct_of_best": round(100 * geo([x["pct_of_best"] or 1e-3 for x in oof]), 2),
             "failing_picks": sum(x["failed"] for x in oof),
             "confidence_auroc": auroc(loo, labels), "seconds": round(time.time() - t0)}
        out.append(r)
        print(f"{name:45s} within5 {100 * r['within_5pct']:5.1f}%  best {r['pct_of_best']:6.2f}%  "
              f"fail {r['failing_picks']}  AUROC {r['confidence_auroc']}  ({r['seconds']}s)", flush=True)
    (PROJECT_ROOT / "results" / "system1_tuning.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
