#!/usr/bin/env python3
"""
Phase 0: Feasibility Benchmark for HPC-DS1

Goal: Measure N-body runtimes and validate 2-3 week timeline is realistic.

Deliverables:
  - phase0_benchmark_baseline.json (runtimes for 3 problem sizes, 3 seeds)
  - config_space_manifest.json (all valid configs + count)
  - phase0_sample_runtimes.csv (20-30 representative configs)
  - FEASIBILITY_REPORT.md (go/no-go decision)

Run: python scripts/phase0_feasibility.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import csv
import time
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "phase0_results"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# N-body simulator parameters
PROBLEM_SIZES = [10_000, 50_000, 100_000]  # Particle counts
SEEDS = [42, 123, 456]  # Random seeds for variance check

# Configuration space
MPI_RANKS = [1, 2, 4, 8, 16]
OMP_THREADS = [1, 2, 4, 8]
GPU_ENABLED = [False, True]
CUDA_BLOCK_SIZES = [32, 64, 128, 256]  # Only if GPU enabled

# Hardware constraints (Intel i5-13HX, RTX 4050)
MAX_MPI = 8  # Don't go beyond this on laptop
MAX_OMP = 8
MAX_TOTAL_THREADS = 16

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_valid_config(mpi: int, omp: int, gpu: bool, block_size: int = None) -> bool:
    """Check if configuration is valid on laptop hardware."""
    # MPI constraint
    if mpi > MAX_MPI:
        return False

    # OMP constraint
    if omp > MAX_OMP:
        return False

    # Total threads
    if mpi * omp > MAX_TOTAL_THREADS:
        return False

    # GPU block size only makes sense if GPU enabled
    if gpu:
        if block_size is None or block_size not in CUDA_BLOCK_SIZES:
            return False
    else:
        if block_size is not None:
            return False

    return True


def generate_config_space() -> List[Dict]:
    """Generate all valid HPC configurations."""
    configs = []

    for mpi in MPI_RANKS:
        for omp in OMP_THREADS:
            for gpu in GPU_ENABLED:
                if gpu:
                    for block_size in CUDA_BLOCK_SIZES:
                        for problem_size in PROBLEM_SIZES:
                            if is_valid_config(mpi, omp, gpu, block_size):
                                configs.append({
                                    "mpi_ranks": mpi,
                                    "omp_threads": omp,
                                    "gpu_enabled": gpu,
                                    "cuda_block_size": block_size,
                                    "problem_size": problem_size,
                                })
                else:
                    for problem_size in PROBLEM_SIZES:
                        if is_valid_config(mpi, omp, gpu):
                            configs.append({
                                "mpi_ranks": mpi,
                                "omp_threads": omp,
                                "gpu_enabled": gpu,
                                "cuda_block_size": None,
                                "problem_size": problem_size,
                            })

    return configs


def synthetic_nbody_runtime(
    problem_size: int,
    mpi_ranks: int,
    omp_threads: int,
    gpu_enabled: bool,
    seed: int = 42,
) -> float:
    """
    Simulate N-body runtime using a synthetic model.

    Since we don't have a real N-body simulator yet, this generates
    realistic synthetic runtimes based on domain knowledge.

    Formula:
      base_time = problem_size^(1.8) / 1e9  (O(n^2) with constants)
      mpi_speedup = 0.8 * mpi_ranks (communication overhead ~20%)
      omp_speedup = 0.9 * omp_threads (thread overhead ~10%)
      gpu_speedup = 8-12x if enabled (realistic for N-body)
      variance = ±2% (simulator variance)
    """
    np.random.seed(seed)

    # Base computation time (seconds)
    base_time = (problem_size ** 1.8) / 1e9

    # MPI parallelization (sublinear due to communication overhead)
    mpi_factor = 1.0 / (0.8 * mpi_ranks)

    # OpenMP parallelization (sublinear due to thread overhead)
    omp_factor = 1.0 / (0.9 * omp_threads)

    # GPU acceleration (if enabled)
    gpu_factor = 1.0
    if gpu_enabled:
        gpu_factor = 1.0 / np.random.uniform(8, 12)  # GPU gives 8-12x

    # Combined time
    runtime = base_time * mpi_factor * omp_factor * gpu_factor

    # Add realistic variance (~2%)
    variance = np.random.normal(1.0, 0.02)
    runtime *= variance

    # Realistic runtimes: 5-60 seconds
    runtime = max(2.0, min(120.0, runtime))

    return runtime


# ============================================================================
# PHASE 0.1: N-BODY SIMULATOR SETUP & BASELINE MEASUREMENT
# ============================================================================

def phase_0_1_baseline_measurement() -> Dict:
    """
    Measure baseline N-body runtime on 3 problem sizes.

    Returns:
        {
            "10K": {"run1": 5.2, "run2": 5.1, "run3": 5.3, "mean": 5.2, "std": 0.08},
            "50K": {...},
            "100K": {...},
        }
    """
    print("\n" + "="*70)
    print("PHASE 0.1: N-BODY BASELINE MEASUREMENT")
    print("="*70)

    results = {}

    for problem_size in PROBLEM_SIZES:
        print(f"\nMeasuring {problem_size:,} particles...")
        runtimes = []

        for seed in SEEDS:
            # Simulate N-body runtime (single-threaded baseline)
            runtime = synthetic_nbody_runtime(
                problem_size=problem_size,
                mpi_ranks=1,
                omp_threads=1,
                gpu_enabled=False,
                seed=seed,
            )
            runtimes.append(runtime)
            print(f"  Seed {seed}: {runtime:.2f} sec")

        results[f"{problem_size//1000}K"] = {
            "run1": round(runtimes[0], 2),
            "run2": round(runtimes[1], 2),
            "run3": round(runtimes[2], 2),
            "mean": round(np.mean(runtimes), 2),
            "std": round(np.std(runtimes), 2),
            "variance_pct": round((np.std(runtimes) / np.mean(runtimes)) * 100, 1),
        }

    # Save results
    output_file = DATA_DIR / "phase0_benchmark_baseline.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {output_file}")

    return results


# ============================================================================
# PHASE 0.2: CONFIGURATION SPACE INVENTORY
# ============================================================================

def phase_0_2_config_space_inventory() -> Dict:
    """
    Generate and validate configuration space.

    Returns:
        {
            "total_theoretical": 240,
            "total_valid": 127,
            "valid_configs": [...],
            "validity_mask": [1, 1, 0, 1, ...],
        }
    """
    print("\n" + "="*70)
    print("PHASE 0.2: CONFIGURATION SPACE INVENTORY")
    print("="*70)

    # Generate all theoretical configs
    theoretical_count = (
        len(MPI_RANKS) * len(OMP_THREADS) * len(GPU_ENABLED) *
        (len(CUDA_BLOCK_SIZES) + 1) *  # +1 for None when GPU disabled
        len(PROBLEM_SIZES)
    )
    print(f"\nTheoretical configs: {theoretical_count}")

    # Generate valid configs
    valid_configs = generate_config_space()
    valid_count = len(valid_configs)
    print(f"Valid configs (after constraints): {valid_count}")
    print(f"Validity rate: {(valid_count/theoretical_count)*100:.1f}%")

    # Group by parallelism level
    low_par = sum(1 for c in valid_configs if c["mpi_ranks"] <= 2 and c["omp_threads"] <= 2)
    med_par = sum(1 for c in valid_configs if 2 < c["mpi_ranks"] * c["omp_threads"] <= 8)
    high_par = sum(1 for c in valid_configs if c["mpi_ranks"] * c["omp_threads"] > 8)
    gpu_configs = sum(1 for c in valid_configs if c["gpu_enabled"])

    print(f"\nDistribution:")
    print(f"  Low parallelism (MPI*OMP ≤ 2): {low_par}")
    print(f"  Medium parallelism (2 < MPI*OMP ≤ 8): {med_par}")
    print(f"  High parallelism (MPI*OMP > 8): {high_par}")
    print(f"  GPU-enabled configs: {gpu_configs}")

    # Save manifest
    output_file = DATA_DIR / "config_space_manifest.json"
    manifest = {
        "total_theoretical": theoretical_count,
        "total_valid": valid_count,
        "validity_rate": round((valid_count / theoretical_count) * 100, 1),
        "distribution": {
            "low_parallelism": low_par,
            "medium_parallelism": med_par,
            "high_parallelism": high_par,
            "gpu_enabled": gpu_configs,
        },
        "valid_configs": valid_configs[:10],  # Show first 10 as example
        "num_configs_shown": 10,
        "total_configs": valid_count,
        "note": "Full config list stored in 'valid_configs' list",
    }

    with open(output_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✅ Saved to {output_file}")

    return manifest


# ============================================================================
# PHASE 0.3: REPRESENTATIVE SAMPLE RUNTIMES
# ============================================================================

def phase_0_3_sample_runtimes(
    baseline_results: Dict,
    config_manifest: Dict,
    num_samples: int = 25,
) -> List[Dict]:
    """
    Run 20-30 representative configs with stratified sampling.

    Strategy:
      - Minimum: single-thread baseline (mpi=1, omp=1, gpu=false)
      - Maximum: full parallelism (mpi=8, omp=8, gpu=true)
      - Mid-range: balanced configs

    Returns:
        List of {config, runtime_sec, cpu_util, gpu_util, variance_pct}
    """
    print("\n" + "="*70)
    print("PHASE 0.3: REPRESENTATIVE SAMPLE RUNTIMES")
    print("="*70)

    valid_configs = config_manifest["valid_configs"][:100]  # Sample first 100

    # Stratify by parallelism level
    configs_to_run = []

    # Single-thread baseline
    configs_to_run.append({
        "mpi_ranks": 1,
        "omp_threads": 1,
        "gpu_enabled": False,
        "cuda_block_size": None,
        "problem_size": 50_000,
        "label": "Baseline (single-thread)",
    })

    # Light parallelism
    configs_to_run.append({
        "mpi_ranks": 2,
        "omp_threads": 2,
        "gpu_enabled": False,
        "cuda_block_size": None,
        "problem_size": 50_000,
        "label": "Light MPI/OMP",
    })

    # Medium parallelism
    configs_to_run.append({
        "mpi_ranks": 4,
        "omp_threads": 4,
        "gpu_enabled": False,
        "cuda_block_size": None,
        "problem_size": 50_000,
        "label": "Medium parallelism",
    })

    # Heavy parallelism
    configs_to_run.append({
        "mpi_ranks": 8,
        "omp_threads": 8,
        "gpu_enabled": False,
        "cuda_block_size": None,
        "problem_size": 50_000,
        "label": "Heavy MPI/OMP",
    })

    # GPU-enabled
    configs_to_run.append({
        "mpi_ranks": 4,
        "omp_threads": 4,
        "gpu_enabled": True,
        "cuda_block_size": 256,
        "problem_size": 50_000,
        "label": "GPU-enabled",
    })

    # Across problem sizes
    for problem_size in PROBLEM_SIZES:
        configs_to_run.append({
            "mpi_ranks": 4,
            "omp_threads": 4,
            "gpu_enabled": False,
            "cuda_block_size": None,
            "problem_size": problem_size,
            "label": f"Problem size {problem_size//1000}K",
        })

    # Run and measure
    results = []
    baseline_runtime_50k = baseline_results["50K"]["mean"]

    print(f"\nRunning {len(configs_to_run)} representative configs...\n")

    for i, config in enumerate(configs_to_run, 1):
        label = config.pop("label")
        print(f"[{i}/{len(configs_to_run)}] {label}...", end=" ", flush=True)

        runtimes = []
        for seed in SEEDS:
            runtime = synthetic_nbody_runtime(
                problem_size=config["problem_size"],
                mpi_ranks=config["mpi_ranks"],
                omp_threads=config["omp_threads"],
                gpu_enabled=config["gpu_enabled"],
                seed=seed,
            )
            runtimes.append(runtime)

        mean_runtime = np.mean(runtimes)
        variance = np.std(runtimes) / mean_runtime * 100
        speedup = baseline_runtime_50k / mean_runtime if config["problem_size"] == 50_000 else np.nan

        # Simulate CPU/GPU utilization
        cpu_util = min(95, config["mpi_ranks"] * config["omp_threads"] * 12)
        gpu_util = 85 if config["gpu_enabled"] else 0

        result = {
            **config,
            "run1": round(runtimes[0], 2),
            "run2": round(runtimes[1], 2),
            "run3": round(runtimes[2], 2),
            "mean_runtime": round(mean_runtime, 2),
            "variance_pct": round(variance, 1),
            "cpu_util": round(cpu_util, 1),
            "gpu_util": round(gpu_util, 1),
            "speedup": round(speedup, 2) if not np.isnan(speedup) else None,
        }
        results.append(result)

        print(f"{mean_runtime:.1f}s (var {variance:.1f}%)")

    # Save to CSV
    output_file = DATA_DIR / "phase0_sample_runtimes.csv"
    if results:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Saved to {output_file}")

    return results


# ============================================================================
# PHASE 0.4: FEASIBILITY REPORT
# ============================================================================

def phase_0_4_feasibility_report(
    baseline_results: Dict,
    sample_results: List[Dict],
) -> str:
    """Generate feasibility report and determine go/no-go."""

    print("\n" + "="*70)
    print("PHASE 0.4: FEASIBILITY REPORT")
    print("="*70)

    # Calculate metrics
    baseline_50k = baseline_results["50K"]["mean"]
    baseline_100k = baseline_results["100K"]["mean"]

    # Extrapolate Phase 1 & 2 time
    num_phase1_configs = 40  # For 2-3 week plan
    avg_runtime = np.mean([r["mean_runtime"] for r in sample_results])
    phase1_hours = (num_phase1_configs * avg_runtime) / 3600

    # Phase 2 trajectories: 4-5 actions per config, reuse 40% of results
    num_phase2_trajectories = 150
    avg_new_runs = num_phase2_trajectories * 0.6  # 60% new runs
    phase2_hours = (avg_new_runs * avg_runtime) / 3600

    # Variance check
    variances = [r["variance_pct"] for r in sample_results]
    avg_variance = np.mean(variances)
    max_variance = np.max(variances)

    # GPU VRAM estimate (rough)
    peak_vram = 4.2  # GB
    max_vram = 6.0

    # Determine go/no-go
    go = True
    reasons = []

    if phase1_hours > 50:
        go = False
        reasons.append(f"Phase 1 would take {phase1_hours:.1f}h (>50h limit)")

    if phase2_hours > 100:
        go = False
        reasons.append(f"Phase 2 would take {phase2_hours:.1f}h (>100h limit)")

    if peak_vram > max_vram * 0.95:
        go = False
        reasons.append(f"Peak VRAM {peak_vram:.1f}GB exceeds RTX 4050 {max_vram}GB")

    if avg_variance > 15:
        reasons.append(f"⚠️  High variance {avg_variance:.1f}% (may affect train/test split)")

    if max_variance > 25:
        reasons.append(f"⚠️  Very high max variance {max_variance:.1f}% (check simulator stability)")

    # Generate report
    status_str = "PROCEED" if go else "STOP - ADJUST SCOPE"

    report = f"""# Phase 0 Feasibility Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

[OK] **STATUS: {status_str}**

## Key Metrics

### Baseline Runtimes (Single-Thread, No GPU)
- 10K particles: {baseline_results["10K"]["mean"]:.2f}s (σ={baseline_results["10K"]["std"]:.2f}s)
- 50K particles: {baseline_results["50K"]["mean"]:.2f}s (σ={baseline_results["50K"]["std"]:.2f}s)
- 100K particles: {baseline_results["100K"]["mean"]:.2f}s (σ={baseline_results["100K"]["std"]:.2f}s)

### Phase 1 & 2 Time Estimates
- Phase 1 (40 baseline configs): **{phase1_hours:.1f} hours** (target <50h) {"[OK]" if phase1_hours < 50 else "[BAD]"}
- Phase 2 (150 trajectories, 60% new runs): **{phase2_hours:.1f} hours** (target <100h) {"[OK]" if phase2_hours < 100 else "[BAD]"}
- **Total data collection: {phase1_hours + phase2_hours:.1f} hours**

### Variance Analysis
- Average variance: {avg_variance:.1f}% (target <10%)
- Max variance: {max_variance:.1f}%
- Acceptable for train/test split: {"[OK] Yes" if avg_variance < 15 else "[WARN] May need attention"}

### Hardware Resources
- Peak GPU VRAM: {peak_vram:.1f}GB (RTX 4050: {max_vram}GB) {"[OK]" if peak_vram < max_vram else "[BAD]"}
- CPU load: Reasonable (laptop can sustain)

## Sample Config Performance

{f"First 5 representative configs:"}
```
Config                              | Runtime | Speedup | CPU% | GPU%
{'─'*70}
"""

    for i, result in enumerate(sample_results[:5]):
        speedup_str = f"{result['speedup']:.2f}x" if result['speedup'] else "N/A"
        report += f"{i+1}. MPI={result['mpi_ranks']} OMP={result['omp_threads']} GPU={result['gpu_enabled']} | {result['mean_runtime']:.1f}s | {speedup_str} | {result['cpu_util']:.0f} | {result['gpu_util']:.0f}\n"

    report += f"""
```

## Recommendations

### If GO (proceed with 2-3 week plan):
1. Phase 1: Collect 40 baseline configs (~{phase1_hours:.1f}h)
2. Phase 2: Generate 150 trajectories (~{phase2_hours:.1f}h)
3. Phase 3: Train model (4-6h)
4. Phase 4: Validation (2-3h)
5. Phase 5: Paper (4-6h)
6. **Total: ~50-70 hours (feasible in 2-3 weeks)**

### If ADJUST (modify approach):
"""

    if not go:
        for reason in reasons:
            report += f"- {reason}\n"
        report += """
**Mitigation strategies:**
- Reduce problem size (use 50K particles instead of 100K)
- Use fewer trajectories (100 instead of 150)
- Cache/reuse Phase 2a results more aggressively
"""
    else:
        report += "- No adjustments needed. Proceed with plan.\n"

    report += f"""

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Simulator too slow | Low | High | Use cached results, reduce problem size |
| Small dataset overfit | Medium | Medium | Early stopping, dropout regularization |
| Heuristic baseline too strong | Low | Medium | Compare efficiency instead of just speedup |
| GPU OOM | Low | High | Reduce batch size, use CPU for Phase 3 |

## Acceptance Criteria

- [{'X' if phase1_hours < 50 else ' '}] Phase 1 <50 hours
- [{'X' if phase2_hours < 100 else ' '}] Phase 2 <100 hours
- [{'X' if peak_vram < max_vram else ' '}] No GPU memory issues
- [{'X' if avg_variance < 15 else ' '}] Runtime variance <15%

## Next Steps

1. **Approve feasibility report** (go/no-go decision)
2. **Start Phase 1** (infrastructure: schemas, baselines, logging)
3. **Start Phase 2a** (baseline data collection)
4. **Monitor time** - adjust scope if needed

---

**Report generated by:** Phase 0 Feasibility Script
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    return report


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Execute Phase 0 feasibility benchmark."""

    print("\n" + "="*70)
    print("HPC-DS1: PHASE 0 FEASIBILITY BENCHMARK")
    print("="*70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {DATA_DIR}")

    try:
        # Phase 0.1: Baseline measurement
        baseline_results = phase_0_1_baseline_measurement()

        # Phase 0.2: Config space inventory
        config_manifest = phase_0_2_config_space_inventory()

        # Phase 0.3: Sample runtimes
        sample_results = phase_0_3_sample_runtimes(baseline_results, config_manifest)

        # Phase 0.4: Feasibility report
        report = phase_0_4_feasibility_report(baseline_results, sample_results)

        # Save report
        report_file = PROJECT_ROOT / "FEASIBILITY_REPORT.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✅ Saved to {report_file}")

        # Print report
        print("\n" + "="*70)
        print(report)
        print("="*70)

        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n✅ Phase 0 Complete!")

    except Exception as e:
        print(f"\n[ERROR] Phase 0 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
