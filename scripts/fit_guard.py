#!/usr/bin/env python3
"""Fit the familiarity guard, attach it to the trained model, and measure it.

The network's weights are not touched: the guard is fitted on the same
train/validation split scripts/train.py used and stored next to them in
model/hpc_ds1.pt. Then, on real measurements:

  held-out sizes   how many familiar test decisions the guard escalates, and
                   what it does to auto-run share and accuracy (three tiers)
  unseen kernels   leave-one-kernel-out: how many states of the never-seen
                   kernel the guard flags, with and without the kernel-name check,
                   and how many confidently wrong auto-runs that prevents
                   (auto-run numbers from results/generalization.json)

Results -> results/guard.json and the visualizer export.

    python scripts/fit_guard.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hpc_ds1.api import HPCDS1  # noqa: E402
from hpc_ds1.dataset import build_states, build_tables, load_hardware, load_measurements  # noqa: E402
from hpc_ds1.guard import FamiliarityGuard  # noqa: E402
from hpc_ds1.kernels import WORKLOAD_NAMES  # noqa: E402
from hpc_ds1.training import fit_guard, is_good, split  # noqa: E402

CKPT = PROJECT_ROOT / "model" / "hpc_ds1.pt"


def tiers(model, rows, guarded):
    """Decision tiers on rows, with or without the guard."""
    guard, model.guard = model.guard, (model.guard if guarded else None)
    ds = model.decide_batch([r["state"] for r in rows])
    model.guard = guard
    modes = Counter(d.mode for d in ds)
    auto = [is_good(r, d.choice) for r, d in zip(rows, ds) if d.mode == "auto"]
    n = len(rows)
    return {"auto": round(modes["auto"] / n, 4), "confirm": round(modes["confirm"] / n, 4),
            "escalate": round(modes["escalate"] / n, 4),
            "auto_accuracy": round(sum(auto) / len(auto), 4) if auto else None,
            "wrong_auto_runs": len(auto) - sum(auto), "decisions": n}


def main():
    hardware = load_hardware()
    tables, _ = build_tables(load_measurements())
    states = build_states(tables, hardware)
    tr, va, te = split(states)

    guard = fit_guard(tr, va, hardware)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    ckpt["guard"] = guard.to_dict()
    torch.save(ckpt, CKPT)
    print(f"guard attached to {CKPT} (radius {guard.radius:.3f}, {len(guard.ref)} reference states)")

    model = HPCDS1.load(CKPT, device="cpu")
    flagged = [c for c in guard.check([r["state"] for r in te]) if not c["familiar"]]
    held_out = {"test_states": len(te), "flagged": len(flagged),
                "flag_reasons": sorted({r for c in flagged for r in c["reasons"]}),
                "without_guard": tiers(model, te, False), "with_guard": tiers(model, te, True)}
    print(f"held-out sizes: guard flags {len(flagged)}/{len(te)} familiar states")
    for k in ("without_guard", "with_guard"):
        t = held_out[k]
        print(f"  {k:14s} auto {t['auto']:.1%} confirm {t['confirm']:.1%} escalate {t['escalate']:.1%}  "
              f"auto accuracy {t['auto_accuracy']}  wrong auto-runs {t['wrong_auto_runs']}")

    gen_path = PROJECT_ROOT / "results" / "generalization.json"
    gen = {r["held_out"]: r for r in json.loads(gen_path.read_text())["leave_one_kernel_out"]} if gen_path.exists() else {}
    unseen = []
    for w in WORKLOAD_NAMES:
        rows = [s for s in states if s["workload"] == w]
        rest = [s for s in states if s["workload"] != w]
        random.Random(1).shuffle(rest)  # same split as scripts/generalization.py
        n_val = int(len(rest) * 0.15)
        g = FamiliarityGuard.fit([r["state"] for r in rest[n_val:]], [r["state"] for r in rest[:n_val]], hardware)
        checks = g.check([r["state"] for r in rows])
        r = {"held_out": w, "states": len(rows),
             "flagged_pct": round(100 * sum(not c["familiar"] for c in checks) / len(rows), 1),
             "flagged_by_distance_only_pct": round(100 * sum(c["novelty"] > 1 for c in checks) / len(rows), 1)}
        if w in gen and gen[w].get("auto_run_accuracy") is not None:
            auto_n = gen[w]["auto_run_at_085"] * gen[w]["test_states"]
            r["confident_wrong_before"] = round(auto_n * (1 - gen[w]["auto_run_accuracy"]))
            r["confident_wrong_after"] = round(r["confident_wrong_before"] * (1 - r["flagged_pct"] / 100))
        unseen.append(r)
        print(f"unseen {w:10s} flagged {r['flagged_pct']:5.1f}% (distance alone {r['flagged_by_distance_only_pct']:5.1f}%)"
              + (f"  confidently wrong auto-runs {r['confident_wrong_before']} -> {r['confident_wrong_after']}"
                 if "confident_wrong_before" in r else ""))

    results = {"radius": guard.radius, "quantile": 0.99, "held_out_sizes": held_out, "unseen_kernels": unseen}
    (PROJECT_ROOT / "results" / "guard.json").write_text(json.dumps(results, indent=2))
    viz = PROJECT_ROOT / "visualizer" / "model_internals.json"
    export = json.loads(viz.read_text())
    export["guard"] = results
    viz.write_text(json.dumps(export))
    print("saved results/guard.json")


if __name__ == "__main__":
    main()
