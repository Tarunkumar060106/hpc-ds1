# HPC-DS1

A small, confidence-aware decision model that tunes HPC jobs, used the way Jev is: you call `decide(state)` and get back a choice, a confidence, and every option's probability. Confident decisions run automatically; uncertain ones go to an engineer.

**Everything is trained and evaluated on real measurements.** Every runtime in the dataset comes from actually running eight workloads on this machine (six HPC kernels, a neural-network training step, and backend request handling): an Intel i5 (10 cores / 16 threads, 24 GB) with an RTX 4050 Laptop GPU. Nothing is simulated.

```python
import hpc_ds1

model = hpc_ds1.load()
state = model.observe("stencil", 160, {"ranks": 1, "threads": 1})   # runs the job once, for real
d = model.decide(state)                     # d.choice, d.confidence, d.probabilities, d.should_escalate
result = model.tune("stencil", 160, {"ranks": 1, "threads": 1})   # closed loop, every step runs for real
```

## Results (held-out problem sizes the model never saw)

| | HPC-DS1 | Rules of thumb | Greedy search¹ | Random moves |
|---|---|---|---|---|
| Tuning: % of best config reached | **93.9%** | 90.1% | 94.8% | 82.3% |
| Real runs per tuning session | 2.7 | 2.4 | 10.7 | 8.4 |
| Single decision picks a best move² | **92.3%** | 70.3% | – | – |
| Decisions that made the job slower | **0.0%** | 5.6% | – | – |
| Decisions that picked a crashing / wrong-answer config | **0.0%** | 7.7% | – | – |

¹ Tries every neighbouring config at every step (what a careful engineer does by hand).
² Can still reach within 2% of the best runtime reachable from that state.

- **Confidence:** calibration error is 0.141 after temperature scaling, above the 0.12 target (it was 0.099 with the six HPC kernels only). At a 0.85 threshold, 58% of decisions run automatically and 95% of those are best moves; 35% ask to confirm and 7% escalate.
- **Live tuning sessions:** eight sessions, one per workload; every session ended within 0.5% of the best known config or beat it (99.5–104%), in 3–6 real runs. Earlier six-kernel run, for reference: six sessions, one per kernel, with every step executed on the hardware during evaluation. All six ended on the same device and precision as the best known configuration. Measured against that config re-run back-to-back under the same conditions, they reached 73–111%. The spread is run-to-run timing noise of about ±25% on millisecond GPU runs; one session ended on the *identical* config and still read 93%.
- **Never-seen workload** (train on 7, test on the 8th): accuracy ranges from 91% (training step) down to 18% (server), and calibration degrades (error 0.06–0.67). The familiarity guard escalates 100% of those decisions: 240 confidently wrong automatic moves become 0, while flagging 0 of 286 familiar test decisions (`results/guard.json`).

Full numbers: `results/metrics.json` and `results/generalization.json`; interactive: the layer explorer (below).

## Quick start

```bash
pip install -r requirements.txt     # install the CUDA build of torch for your driver
python -m pytest tests -q
python -m hpc_ds1 serve             # API + layer explorer at http://127.0.0.1:8765
```

```bash
python -m hpc_ds1 decide --workload gemm --size 2048 --ranks 2 --threads 4
python -m hpc_ds1 tune   --workload stencil --size 160       # asks before running uncertain moves
python -m hpc_ds1 info
```

HTTP: `POST /v1/decide`, `/v1/trace` (every layer's activations), `/v1/tune`, `/v1/decide_batch`; `GET /v1/health`, `/v1/schema`, `/v1/catalog`. Without `telemetry` in the request, the server runs the job once to measure it.

## How it works

**Workloads** (`hpc_ds1/kernels.py`). Eight real workloads, each splittable across ranks:

| Kernel | Character | Sizes (held-out in bold) |
|---|---|---|
| Dense matrix multiply | compute-bound, large result to gather | 1024–3072, **2048** |
| 3D FFT, slab decomposition | transpose / communication-heavy | 64–256, **192** |
| 3D Jacobi stencil | memory-bound, ghost-zone decomposition | 96–192, **160** |
| Sparse matrix-vector multiply | memory-bound, irregular | 200k–1.6M rows, **800k** |
| N-body forces | compute-bound, O(N²) | 2048–6144, **4096** |
| Monte Carlo option pricing | embarrassingly parallel, RNG-heavy | 2M–16M paths, **8M** |
| Neural-network training step | AI training: forward + backward, gradient all-reduce | batch 1024–8192, **4096** |
| Backend request handling | server: JSON parse, logic, compress, hash; GIL-bound; CPU fp32 only | 1000–8000 requests, **4000** |

**Your own task:** write a `Kernel` subclass with `@register` in a file in `~/.hpc_ds1/plugins/` (example: `examples/plugins/etl_batch.py`), then `hpc-ds1 setup --workloads <name>`. Until it is measured, decisions about it are escalated.

**Execution** (`hpc_ds1/runner.py`). Ranks are separate processes with their own memory, connected to rank 0 by pipes. Every rank computes its shard with T threads, and rank 0 gathers and combines the results. This is the distributed-memory pattern MPI uses, run on one machine without an MPI install. Each run is checked against an fp32 reference; a config that crashes or gives a wrong answer is recorded as failed. Telemetry is measured: runtime (median of repeats), CPU time across all ranks, GPU busy time, and communication/gather overhead.

**Configurations and actions** (`hpc_ds1/space.py`). A config is {ranks 1–8, threads 1–16, GPU on/off, fp32/bf16}, which gives 44 per problem. There are nine actions: ADD/REMOVE_RANKS, ADD/REMOVE_THREADS, USE_GPU/CPU, USE_BF16/FP32 and STOP_TUNING.

**Data** (`scripts/collect.py`, `hpc_ds1/dataset.py`).
- 2,112 real runs over 48 problems: 1,731 ran correctly and 381 failed or have no such version (server has no GPU or bf16 path).
- The failures are bf16 FFT and SpMV on this CPU, and GPU bf16 FFT at non-power-of-two sizes.
- Each correct config becomes a state. Its label is the first move of the best chain of measured, clearly faster (>5% and above noise) moves.
- Several moves are often equally good, so training uses soft targets that share credit among moves reaching within about 2% of the best.

**Model** (`hpc_ds1/net.py`, 2.8M parameters, size chosen on validation loss).
- **Encoders:** four modular encoders (hardware, workload, config, telemetry), each with pre-norm residual blocks.
- **Fusion:** attention over the four tokens.
- **Heads:** action, stop, and expected-improvement heads; invalid actions are masked.
- **Calibration:** temperature scaling fitted on validation.

**Tuning loop** (`hpc_ds1/tuner.py`).
1. The model decides; below the threshold, the decision is escalated.
2. The chosen config is run for real.
3. A faster result is kept. A slower or failing one is reverted and excluded, and the model is asked again.
4. The session ends on STOP or after two misses in a row.

## Reproduce everything on your machine

```bash
python scripts/collect.py          # ~45 min: measures every config of every kernel (resumable)
python scripts/train.py            # trains, calibrates, evaluates, runs live sessions, exports the UI data
python scripts/generalization.py   # leave-one-kernel-out + input ablations
```

`collect.py` measures whatever machine it runs on, so a different machine produces a model for that machine. Keep the machine plugged in and otherwise idle while collecting.

## Limitations

- **One machine.** The model has only seen this laptop. The hardware features are constant in the training data, so it can't transfer to other hardware without collecting there.
- **Ranks are processes, not MPI.** The communication pattern (shard, compute, gather to rank 0) is real, but it runs over pipes on one machine, not over a network.
- **Generalizing to new kinds of work is partial** (see results), and calibration degrades on them. That's exactly where escalating to an engineer matters most.
- **Evaluation replays real measurements** for the hundreds of tuning sessions, because running them all live would take hours. The live sessions show the same behaviour.
- **Telemetry:** memory use and run-to-run noise are recorded but deliberately not model inputs. Both depend on what the measuring process did earlier, not on the job, and they made identical jobs look different live versus during collection.

## Layout

```
hpc_ds1/      kernels, runner, space, features, net, dataset, tuner, api, server, CLI
scripts/      collect.py, train.py, generalization.py
data/real/    measurements.jsonl, hardware.json
model/        hpc_ds1.pt
results/      metrics.json, generalization.json, logs
visualizer/   layer explorer (index.html + model_internals.json)
tests/        test_project.py
docs/         DEMO.md (live demo script), research/ (Jev/Laya research, PRD)
archive/      the earlier simulator-based version, kept for history only
```
