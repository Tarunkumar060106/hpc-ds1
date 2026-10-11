#!/usr/bin/env python3
"""
Out-of-fold System 1 runtime estimates for every measured problem and config.

For each fold (one size position of every workload), System 1 is trained on the
other problems and asked to estimate every config of the held-out ones. The
question encoder trains on these estimates, so the hints it learns to read are
as imperfect as the ones it will get for a new job in use.

    python scripts/system1_oof.py   -> results/system1_oof.json
        {"workload:size": {"rt": {"ranks,threads,gpu,precision": predicted runtime sec},
                           "fail": {same keys: predicted probability the config fails}}}
"""

import json
import sys
import warnings
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hpc_ds1.dataset import load_hardware, load_measurements  # noqa: E402
from hpc_ds1.kernels import KERNELS, WORKLOAD_NAMES  # noqa: E402
from hpc_ds1.system1 import DEFAULT_HP, rows_from_measurements, score_configs, train_ensemble  # noqa: E402

warnings.filterwarnings("ignore")


def key_str(c):
    return f"{c['ranks']},{c['threads']},{int(c['gpu'])},{c['precision']}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = load_hardware()
    rows = load_measurements()
    sizes = {w: sorted(KERNELS[w].sizes) for w in WORKLOAD_NAMES}
    out = {}
    for k in range(max(len(v) for v in sizes.values())):
        held = {(w, sizes[w][k]) for w in WORKLOAD_NAMES if k < len(sizes[w])}
        x, y, f, g = rows_from_measurements([r for r in rows if (r["workload"], r["size"]) not in held], hardware)
        members = train_ensemble(x, y, f, g, device, DEFAULT_HP)
        for w, s in sorted(held):
            configs, log_rt, p_fail = score_configs(members, hardware, w, s, WORKLOAD_NAMES, device)
            out[f"{w}:{s}"] = {"rt": {key_str(c): round(10 ** float(v), 7) for c, v in zip(configs, log_rt.mean(0))},
                               "fail": {key_str(c): round(float(v), 4) for c, v in zip(configs, p_fail.mean(0))}}
        print(f"fold {k}: {len(held)} problems estimated", flush=True)
    (PROJECT_ROOT / "results" / "system1_oof.json").write_text(json.dumps(out))
    print(f"saved results/system1_oof.json ({len(out)} problems)")


if __name__ == "__main__":
    main()
