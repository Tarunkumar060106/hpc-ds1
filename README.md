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
| Tuning: % of best config reached | **96.3%** | 90.1% | 94.8% | 82.3% |
| Real runs per tuning session | 2.9 | 2.4 | 10.7 | 8.4 |
| Single decision picks a best move² | **94.1%** | 70.3% | – | – |
| Decisions that made the job slower | **0.0%** | 5.6% | – | – |
| Decisions that picked a crashing / wrong-answer config | **0.0%** | 7.7% | – | – |

¹ Tries every neighbouring config at every step (what a careful engineer does by hand).
² Can still reach within 2% of the best runtime reachable from that state.

- **Confidence:** calibration error is 0.106 after temperature scaling, within the 0.12 target. At a 0.85 threshold, 59% of decisions run automatically and 97% of those are best moves; 38% ask to confirm and 2% escalate. A 5-seed ensemble was also trained; it was not better calibrated on validation, so the single network is used.
- **Live tuning sessions:** eight sessions, one per workload, every step executed on the hardware: 85–114% of the best known config re-run at the same time (the spread is run-to-run timing noise). Earlier six-kernel run, for reference: six sessions, one per kernel, with every step executed on the hardware during evaluation. All six ended on the same device and precision as the best known configuration. Measured against that config re-run back-to-back under the same conditions, they reached 73–111%. The spread is run-to-run timing noise of about ±25% on millisecond GPU runs; one session ended on the *identical* config and still read 93%.
- **Never-seen workload** (train on 7, test on the 8th): accuracy ranges from 86% (FFT, training step) down to 27% (server), and calibration degrades. The familiarity guard escalates 100% of those decisions: 240 confidently wrong automatic moves become 0, while flagging 0 of 286 familiar test decisions (`results/guard.json`).

Full numbers: `results/metrics.json` and `results/generalization.json`; interactive: the layer explorer (below).

## System 1: the whole configuration before anything runs

`hpc_ds1/system1.py` answers from the job description alone: workload, size and the machine. A performance model learned from every real measured run predicts each configuration's runtime and failure risk. An ensemble of eight, each trained on a different bootstrap resample of problems and with a loss that ranks the configs of each problem, scores all 44 configurations in one pass (about 9 ms on CPU); its disagreement becomes the confidence.

```bash
hpc-ds1 predict --workload mltrain --size 4096   # instant: GPU, bf16 — nothing is run
hpc-ds1 plan    --workload stencil --size 160    # dual process: System 1 picks, System 2 only if needed
```

**Dual process.** If System 1 is confident, the job is familiar, and the first real run is no slower than 1.5× the prediction, the configuration is kept. Otherwise System 2 (the step-by-step tuning loop) starts from System 1's pick, and unfamiliar jobs are escalated. Sessions (`"start": "system1"`) and fleet tasks use this by default.

| Held-out problem sizes, from a slow 1×1 CPU start | % of best speed | Real runs |
|---|---|---|
| **Dual process (System 1 → System 2 when unsure)** | **98.7%** | **1.0** |
| System 2 alone | 99.1% | 5.1 |
| Rules of thumb | 88.1% | 3.6 |

- Cross-validated over all 72 problems (9 folds by size): 83.3% of instant picks are within 5% of the best configuration, reaching 96.9% of best speed, with 0 crashing picks. The dual process gives up about 0.4 points of speed against System 2 alone for 5x fewer real runs.
- **Confidence:** calibrated (error 0.05), and about half of jobs are confident enough to skip System 2 (those reach 97.4% of best speed). It separated good from bad picks with AUROC 0.63 on 48 problems but only 0.52 on 72, so treat it as weak. The recipe was chosen by cross-validation without touching the test sizes (`scripts/tune_system1.py`, `results/system1_tuning.json`); it traded about one problem's worth of accuracy for this.
- **Limitations:** System 1's confidence barely separates good picks from bad, so System 2's check matters. Results on the 8 test sizes are a small sample.
- On a workload System 1 never saw, picks can be badly wrong (the server workload, held out, picks the GPU, which has no server version). The size and workload checks escalate those.

### Level 5: System 2 teaches System 1

Every real run made through HPC-DS1 on a machine (tuning sessions, `plan`, fleet tasks, the playground) is logged to `~/.hpc_ds1/data/live_runs.jsonl` in the same format as the collected measurements (turn off with `HPC_DS1_LOG_LIVE=0`). `hpc-ds1 learn` retrains System 1 on the collected measurements plus those live runs and saves `~/.hpc_ds1/model/system1.pt`, which `hpc_ds1.load()` then prefers. Problems the facility actually runs become familiar over time; rerun `scripts/train_system1.py` to re-measure accuracy.

Reproduce: `python scripts/train_system1.py` → `model/system1.pt`, `results/system1.json`.

## Ask it anything: a Laya-style question encoder

`hpc_ds1/encoder.py` answers free-text questions about a job, Jev-style: you give a question, a type (`noul` = yes/no/unclear, or `choice` with options), and the job; a fine-tuned **ModernBERT-base** reads question + options + the job described as text in one pass and scores the options (about 28 ms on the GPU, temperature-calibrated).

```bash
curl -X POST http://127.0.0.1:8765/v1/ask -H "Content-Type: application/json" -d '{"workload": "mltrain", "size": 4096,
  "config": {"ranks": 2, "threads": 4}, "questions": {"gpu": {"type": "noul", "question": "Would the graphics card speed this up?"}}}'
```

In the console playground, any question key that isn't one of the built-in five goes to the encoder.

Every training question is answered from the real measurements (e.g. "will bf16 be faster here?" is labelled by the two measured runtimes). Seven question families, several phrasings each, and synonyms (GPU / CUDA device / accelerator, bf16 / bfloat16 / half precision, ...); the last phrasing of every family and one synonym per term ("graphics card", "MPI ranks", ...) are never trained on.

| Questions (9,148 from 2,589 measured states) | Accuracy | Most-common-answer baseline | Calibration error |
|---|---|---|---|
| Unseen wording (incl. unseen synonyms) | **83.1%** | 61.6% | 0.024 |
| Unseen problem sizes | **83.4%** | 55.8% | 0.028 |
| Unseen wording and unseen problems | **82.8%** | 63.0% | 0.056 |

Strong on "CPU or GPU?", "fp32 or bf16?", "which of these two setups is faster?", "will this crash?"; weakest on "will this one change make it faster?" (about 60%). The familiarity guard marks answers about unfamiliar jobs. Weights are not in git: `HF_HOME=models/hf python scripts/train_encoder.py` (downloads ModernBERT-base, about 600 MB, then about 8 minutes on an RTX 4050).

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
| Dense matrix multiply | compute-bound, large result to gather | 1024–3072 (9 sizes), **2048** |
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
- 3,168 real runs over 72 problems (8 workloads x 9 sizes): 2,589 ran correctly and 579 failed or have no such version (server has no GPU or bf16 path).
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
