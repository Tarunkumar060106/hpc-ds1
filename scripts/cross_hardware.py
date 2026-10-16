#!/usr/bin/env python3
"""
Cross-hardware check: a System 1 trained on one machine, asked about another
machine it never measured. For every problem measured on the target machine,
System 1 picks one configuration; the pick is scored with the target machine's
real measured runtimes.

    python scripts/cross_hardware.py data/lab/type_a_5090/.hpc_ds1 [system1.pt]
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hpc_ds1.dataset import build_tables  # noqa: E402
from hpc_ds1.paths import MODELS_DIR  # noqa: E402
from hpc_ds1.space import config_key, make_config  # noqa: E402
from hpc_ds1.system1 import System1  # noqa: E402


def main():
    home = Path(sys.argv[1])
    model = Path(sys.argv[2]) if len(sys.argv) > 2 else MODELS_DIR / "system1.pt"
    hardware = json.loads((home / "data" / "hardware.json").read_text())
    rows = [json.loads(l) for l in (home / "data" / "measurements.jsonl").read_text().splitlines() if l.strip()]
    tables, _ = build_tables(rows)
    s1 = System1.load(model)
    res, per = [], []
    for (w, s), table in sorted(tables.items()):
        if w not in s1.workloads:
            continue
        best = min(t["runtime_sec"] for t in table.values())
        default = table.get(config_key(make_config(1, 1, False, "fp32")))
        p = s1.predict(hardware, w, s)
        got = table.get(config_key(make_config(**p.config)))
        frac = best / got["runtime_sec"] if got else 0.0
        res.append(frac)
        per.append({"problem": f"{w}:{s}", "pick": p.config, "pct_of_best": round(100 * frac, 1),
                    "default_pct": round(100 * best / default["runtime_sec"], 1) if default else None,
                    "mode": p.mode, "confidence": p.confidence})
        print(f"{w:10s} {s:>9} pick {p.config} -> {100 * frac:5.1f}% of best  ({p.mode})")
    ok = [r for r in res if r > 0]
    out = {"target": hardware.get("gpu_model"), "model": str(model.name), "problems": len(res),
           "geo_pct_of_best": round(100 * math.exp(np.mean(np.log(ok))), 1) if ok else 0,
           "median_pct_of_best": round(100 * float(np.median(res)), 1),
           "within_5pct": round(float(np.mean([r >= 1 / 1.05 for r in res])), 3),
           "failed_picks": int(sum(r == 0 for r in res)), "per_problem": per}
    print({k: v for k, v in out.items() if k != "per_problem"})
    dst = PROJECT_ROOT / "results" / "experiments" / f"cross_hardware_{home.parent.name}.json"
    dst.write_text(json.dumps(out, indent=2))
    print("saved", dst)


if __name__ == "__main__":
    main()
