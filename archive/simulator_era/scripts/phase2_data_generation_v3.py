#!/usr/bin/env python3
"""
Phase 2 v3: rollout-based dataset across workloads AND hardware.

Changes vs v2:
  - 5 hardware profiles (laptop GPU, workstation GPU, A100 node, CPU-only
    64-core node, 4-core VM). The simulator now caps parallelism at the core
    count and penalizes oversubscription, so the right move depends on the
    machine, not just the kernel.
  - Tuning rollouts instead of isolated configs: from a random start we keep
    applying the best action (80%) or a random one (20%, for coverage) and
    record every state along the way. This produces the mid-tuning states a
    closed loop actually visits (precision already reduced, GPU already on,
    ...) and natural STOP_TUNING endpoints.
  - reduced_precision / prefetch are part of the state, so a model can tell
    they were already applied.
  - Splits are by episode, so correlated states from one rollout never leak
    across train/val/test.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hpc_ds1.workloads import (  # noqa: E402
    apply_action, build_state, measure, resolve_hardware, sample_scenario, valid_actions_for,
)

EPISODES = 3000
MAX_STEPS = 10
IMPROVEMENT_THRESHOLD = 0.005
EXPLORE_PROB = 0.2
SEED = 11

def label_state(workload, problem_size, config, hw):
    telemetry, bottleneck = measure(workload, problem_size, config, hw)
    base = telemetry["runtime_sec"]
    candidates = {}
    for action in valid_actions_for(config, hardware=hw):
        nxt = apply_action(config, action, workload)
        tel, _ = measure(workload, problem_size, nxt, hw)
        candidates[action] = tel["runtime_sec"]

    best = min(candidates, key=candidates.get) if candidates else None
    if best is None or candidates[best] >= base * (1 - IMPROVEMENT_THRESHOLD):
        label, improvement = "STOP_TUNING", 0.0
    else:
        label, improvement = best, (base - candidates[best]) / base
    return telemetry, bottleneck, candidates, label, improvement


def main():
    rng = random.Random(SEED)
    out_dir = PROJECT_ROOT / "data" / "phase2_v3_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    for ep in range(EPISODES):
        workload, hw_name, config, problem_size = sample_scenario(rng)
        hw = resolve_hardware(hw_name)

        steps = []
        for step in range(MAX_STEPS):
            telemetry, bottleneck, candidates, label, improvement = label_state(
                workload, problem_size, config, hw)
            state = build_state(workload, problem_size, config, hw, telemetry)
            steps.append({
                "episode": ep, "step": step, "workload": workload, "hardware": hw_name,
                "state": state, "candidates": candidates,
                "baseline_runtime": telemetry["runtime_sec"],
                "best_action": label, "best_improvement": improvement,
                "bottleneck": bottleneck,
            })
            if label == "STOP_TUNING":
                break
            valid = list(candidates)
            action = rng.choice(valid) if rng.random() < EXPLORE_PROB else label
            config = apply_action(config, action, workload)
        episodes.append(steps)

    rng.shuffle(episodes)
    n = len(episodes)
    splits = {
        "train": episodes[: int(n * 0.8)],
        "val": episodes[int(n * 0.8): int(n * 0.9)],
        "test": episodes[int(n * 0.9):],
    }

    total = 0
    for name, eps in splits.items():
        rows = [s for e in eps for s in e]
        total += len(rows)
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name:5s}: {len(eps):4d} episodes, {len(rows):5d} states")

    all_rows = [s for e in episodes for s in e]
    print(f"\ntotal states: {total}  (mean {total / n:.1f} per episode)")
    print("\nlabel distribution:")
    for k, v in Counter(r["best_action"] for r in all_rows).most_common():
        print(f"  {k:18s} {v:6d}  {100 * v / total:5.1f}%")
    print("\nbottleneck distribution:")
    for k, v in Counter(r["bottleneck"] for r in all_rows).most_common():
        print(f"  {k:18s} {v:6d}  {100 * v / total:5.1f}%")
    print("\nstates per hardware profile:")
    for k, v in Counter(r["hardware"] for r in all_rows).most_common():
        print(f"  {k:22s} {v:6d}")


if __name__ == "__main__":
    main()
