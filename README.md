# HPC-DS1

**HPC-DS1 picks the right settings for a computer job, and the right computer for it, before the job runs.**

```bash
pip install hpc-ds1
hpc-ds1 setup --quick        # once per machine: measures it for real (~30 min)
hpc-ds1 predict --workload llm_train --size 32
```

## In plain words

Every job a computer runs has settings: how many processes, how many threads, CPU or graphics card (GPU), full or half precision. Pick badly and the *same* job runs 10 to 400 times slower. On a lab RTX 5090 the untuned default was about 50× slower than the right settings, measured on 16 real tasks.

Today people either guess, or tune by trial and error, running the job again and again (10–50 runs) to find good settings. HPC-DS1 does it like this:

| | What it does | Like… |
|---|---|---|
| **System 1** | Looks at the job (what it is, how big, which machine) and answers instantly: "GPU, 8 threads, half precision". Nothing is run. | An experienced mechanic who hears the engine and knows the fix |
| **System 2** | Runs the job once, looks at what happened, and suggests one change at a time | Tuning a radio by turning the dial and listening |
| **Guard** | Notices when a job, size or machine is unlike anything it has seen, says "not sure", and asks a person | A new doctor who refers an unusual case |
| **Fleet** | Connects many machines over the network and sends each job to the one that can finish it first, and only to machines that *can* run it | A dispatcher who won't give a truck job to a bicycle |

(The names System 1 and System 2 come from psychology: Kahneman's fast intuition and slow reasoning.)

**Result on 90 real problems (10 workloads):** System 1's first answer reaches **99.7%** of the best possible speed (median; 97.8% geometric mean) and never picks a setting that crashes. Bayesian optimisation, the standard auto-tuner, needs about 5 trial runs to get there. Tuning costs about **39× less compute** (`results/benchmark.json`).

**Example: a job the laptop can't run.** A language-model training step at batch 96 needs 7.3 GB of GPU memory (measured). A laptop RTX 4050 has 6 GB on paper, but its measured runs slow down sharply from about 3 GB, because Windows and the display keep part of the card and the rest spills into system RAM: batch 96 took 8.6 s per step there, about 17× longer than batch 32 would suggest (0.17 s for a third of the work). At batch 128 it runs out of memory entirely. The fleet learns each machine's real limit from its own measurements and sends the job to the lab's RTX 5090 (32 GB), saying why:

```
✓ task t2 → lab-pc-01  redirected: main-01 can't run it
  needs ~7.3 GB of GPU memory
  ● lab-pc-01
  ✗ main-01    can't run it: needs 7.3 GB of GPU memory; NVIDIA GeForce RTX 4050 Laptop GPU has 6 GB
               but measured runs slow down sharply from 3.0 GB (the rest spills into system RAM)
```

Small jobs still run on the laptop when it finishes first. If a job fails on a machine anyway (out of memory, crash), it moves to the next machine that can run it automatically.

## How is this different from a normal classifier?

A normal classifier takes an input and outputs a label, e.g. "this photo is a cat". HPC-DS1 is different in five ways:

1. **It predicts what will happen, not a label.** System 1 predicts the *runtime and failure risk* of all 44 possible settings at once, then picks the fastest that won't crash. The answer changes with the machine: the same job gets "CPU, 8 processes" on one computer and "GPU, half precision" on another.
2. **It knows when not to trust itself.** A normal classifier is just as sure on things it has never seen. HPC-DS1 has a calibrated confidence and a familiarity guard: on workloads it never saw, it turned 240 confidently wrong automatic decisions into 0 by asking a person instead.
3. **Its training data is measured on your machine.** There's no fixed dataset from the internet: `hpc-ds1 setup` runs real jobs on each machine and learns from those (about 10 minutes of training). Every number in this project is a real measurement.
4. **It acts, and checks itself.** It doesn't stop at a prediction. The job runs, System 2 checks the result, and if a step made things slower it is undone. Every run it makes is saved, so `hpc-ds1 learn` makes System 1 better on the jobs the facility actually runs.
5. **It decides where, not only how.** Across a fleet it compares every machine's expected finish time and its memory limits, and moves work between machines.

## Technical overview

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
| Tuning: % of best config reached | **95.6%** | 91.5% | 95.5% | 83.5% |
| Real runs per tuning session | 2.9 | 2.4 | 10.7 | 8.3 |
| Single decision picks a best move² | **95.0%** | 74.8% | – | – |
| Decisions that made the job slower | **0.3%** | 4.7% | – | – |
| Decisions that picked a crashing / wrong-answer config | **0.0%** | 6.5% | – | – |

¹ Tries every neighbouring config at every step (what a careful engineer does by hand).
² Can still reach within 2% of the best runtime reachable from that state.

Ten workloads, 90 problems, 337 held-out test states (v1.1; v1.0 had eight workloads).

- **Confidence:** calibration error is 0.136 after temperature scaling (0.359 before). At a 0.85 threshold, 56% of decisions run automatically and 97% of those are best moves; 34% ask to confirm and 9% escalate. A 5-seed ensemble was also trained; it was not better calibrated on validation, so the single network is used.
- **Live tuning sessions** (v1.0, not re-run for v1.1): eight sessions, one per workload, every step executed on the hardware: 85–114% of the best known config re-run at the same time (the spread is run-to-run timing noise). Earlier six-kernel run, for reference: six sessions, one per kernel, with every step executed on the hardware during evaluation. All six ended on the same device and precision as the best known configuration. Measured against that config re-run back-to-back under the same conditions, they reached 73–111%. The spread is run-to-run timing noise of about ±25% on millisecond GPU runs; one session ended on the *identical* config and still read 93%.
- **Never-seen workload** (train on the others, test on one never seen): accuracy drops and calibration degrades. The familiarity guard escalates 100% of those decisions for all ten workloads, turning 298 confidently wrong automatic moves into 0, while flagging 0 of 337 familiar test decisions (`results/guard.json`).

Full numbers: `results/metrics.json` and `results/generalization.json`; interactive: the layer explorer (below).

## System 1: the whole configuration before anything runs

`hpc_ds1/system1.py` answers from the job description alone: workload, size and the machine. A performance model learned from every real measured run predicts each configuration's runtime and failure risk. An ensemble of eight, each trained on a different bootstrap resample of problems and with a loss that ranks the configs of each problem, scores all 44 configurations in one pass (about 4 ms on CPU); its disagreement becomes the confidence.

```bash
hpc-ds1 predict --workload mltrain --size 4096   # instant: GPU, bf16 — nothing is run
hpc-ds1 plan    --workload stencil --size 160    # dual process: System 1 picks, System 2 only if needed
```

**Dual process.** If System 1 is confident, the job is familiar, and the first real run is no slower than 1.5× the prediction, the configuration is kept. Otherwise System 2 (the step-by-step tuning loop) starts from System 1's pick, and unfamiliar jobs are escalated. Sessions (`"start": "system1"`) and fleet tasks use this by default.

| Held-out problem sizes, from a slow 1×1 CPU start | % of best speed | Real runs |
|---|---|---|
| **Dual process (System 1 → System 2 when unsure)** | **98.8%** | **1.0** |
| System 2 alone | 99.4% | 5.0 |
| Rules of thumb | 90.3% | 3.5 |

- Cross-validated over all 90 problems (9 folds by size, 10 workloads): 88.8% of instant picks are within 5% of the best configuration, reaching 97.7% of best speed, with 0 crashing picks (v1.0, 72 problems: 83.3% and 96.9%). The dual process gives up about 0.6 points of speed against System 2 alone for 5x fewer real runs.
- **Confidence:** calibrated (error 0.013), but it does not separate good picks from bad: AUROC 0.43 on 90 problems, and 0.41–0.55 across every signal set tried on 72 (`results/experiments/system1_confidence_*.json`), i.e. noise-dominated. More measured problems (level 5, in use) is the fix, not more tuning. About 40% of jobs are confident enough to skip System 2. The recipe was chosen by cross-validation without touching the test sizes (`scripts/tune_system1.py`, `results/system1_tuning.json`); it traded about one problem's worth of accuracy for this.
- **Limitations:** System 1's confidence barely separates good picks from bad, so System 2's check matters. Results on the 10 test sizes are a small sample.
- On a workload System 1 never saw, picks can be badly wrong (the server workload, held out, picks the GPU, which has no server version). The size and workload checks escalate those.

### Level 5: System 2 teaches System 1

Every real run made through HPC-DS1 on a machine (tuning sessions, `plan`, fleet tasks, the playground) is logged to `~/.hpc_ds1/data/live_runs.jsonl` in the same format as the collected measurements (turn off with `HPC_DS1_LOG_LIVE=0`). `hpc-ds1 learn` retrains System 1 on the collected measurements plus those live runs and saves `~/.hpc_ds1/model/system1.pt`, which `hpc_ds1.load()` then prefers. Problems the facility actually runs become familiar over time; rerun `scripts/train_system1.py` to re-measure accuracy.

Reproduce: `python scripts/train_system1.py` → `model/system1.pt`, `results/system1.json`.

## Ask it anything: a Laya-style question encoder

`hpc_ds1/encoder.py` answers free-text questions about a job, Jev-style: you give a question, a type (`noul` = yes/no/unclear, or `choice` with options), and the job; a fine-tuned **ModernBERT-base** reads question + options + the job described as text in one pass and scores the options (about 29 ms on the GPU, temperature-calibrated).

The job text includes **System 1's runtime estimate for every one-step change** ("moving the job to the GPU x0.12; switching to bf16 x0.75; ..."), so the encoder reads performance evidence instead of guessing it. During training these estimates come from System 1 models that never saw that problem (`scripts/system1_oof.py`), so they are as imperfect as the live ones.

```bash
curl -X POST http://127.0.0.1:8765/v1/ask -H "Content-Type: application/json" -d '{"workload": "mltrain", "size": 4096,
  "config": {"ranks": 2, "threads": 4}, "questions": {"gpu": {"type": "noul", "question": "Would the graphics card speed this up?"}}}'
```

In the console playground, any question key that isn't one of the built-in five goes to the encoder.

Every training question is answered from the real measurements (e.g. "will bf16 be faster here?" is labelled by the two measured runtimes). Seven question families, several phrasings each, and synonyms (GPU / CUDA device / accelerator, bf16 / bfloat16 / half precision, ...); the last phrasing of every family and one synonym per term ("graphics card", "MPI ranks", ...) are never trained on.

| Questions (9,148 from 2,589 measured states) | Accuracy | Most-common-answer baseline | Calibration error |
|---|---|---|---|
| Unseen wording (incl. unseen synonyms) | **85.8%** (83.1% without System 1 hints) | 61.6% | 0.036 |
| Unseen problem sizes | **87.4%** (83.4%) | 55.8% | 0.016 |
| Unseen wording and unseen problems | **87.2%** (82.8%) | 63.0% | 0.045 |

Yes/no questions about the effect of one change ("would the graphics card make this faster?") are answered from System 1's estimate for that change, found with a transparent lexicon: 83–85% on held-out questions versus 68–78% for the encoder (`scripts/eval_router.py`); every answer reports its `source`. Strong on "CPU or GPU?" and "fp32 or bf16?" (100%), "which of these two setups is faster?" and "is this near the best?" (85–92%); "will this one change make it faster?" improved from about 60% to 68–78% with the hints. Weak: "will this setup crash?" (50–78%). Adding System 1's failure predictions to the text was tried and was worse on validation, so it is off (`results/experiments/`). The familiarity guard marks answers about unfamiliar jobs. Weights are not in git: `HF_HOME=models/hf python scripts/train_encoder.py` (downloads ModernBERT-base, about 600 MB, then about 8 minutes on an RTX 4050).

## Quick start

Python 3.10+. For GPU moves, first install the CUDA build of PyTorch for your driver from pytorch.org; CPU-only works too (GPU moves are then unavailable).

```bash
pip install hpc-ds1                 # or: pip install "hpc-ds1[encoder]" for free-text questions
hpc-ds1 serve                       # console at http://127.0.0.1:8765
```

From source (to retrain or run the tests):

```bash
git clone https://github.com/Tarunkumar060106/hpc-ds1.git
cd hpc-ds1
pip install -e ".[test]"
python -m pytest tests -q
```

```bash
hpc-ds1 predict --workload mltrain --size 4096                # System 1: instant, nothing is run
hpc-ds1 plan    --workload stencil --size 160                 # System 1, then System 2 only if needed
hpc-ds1 decide  --workload gemm --size 2048 --ranks 2 --threads 4
hpc-ds1 tune    --workload stencil --size 160                 # asks before running uncertain moves
hpc-ds1 tasks                                                 # what the model knows
```

**Your own machine.** The shipped models were trained on one laptop (i5 + RTX 4050); on other hardware the familiarity guard escalates every move until you run `hpc-ds1 setup` (about 1 hour, or `--quick` for about 30 minutes), which measures your machine and trains on it.

**Free-text questions.** Install with the `encoder` extra, then download `hpc-ds1-encoder-v1.0.0.zip` from the [v1.0.0 release](https://github.com/Tarunkumar060106/hpc-ds1/releases/tag/v1.0.0) (about 600 MB, too big for PyPI) and unzip it into `~/.hpc_ds1` so that `~/.hpc_ds1/models/encoder/scorer.pt` exists. In a source checkout, `models/encoder/` also works, or set `HPC_DS1_ENCODER`.

HTTP: `POST /v1/decide`, `/v1/predict`, `/v1/ask`, `/v1/trace`, `/v1/tune`, `/v1/sessions`, `/v1/fleet/...`; `GET /v1/health`, `/v1/schema`, `/v1/queue`, `/v1/audit`. Without `telemetry` in a request, the server runs the job once to measure it.

> **Security:** `hpc-ds1 serve` has no HTTPS and no console login in v1. Run it on a trusted network or inside a private overlay network such as Tailscale or ZeroTier; do not expose it to the public internet.

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

**Model** (`hpc_ds1/net.py`, 0.59M parameters: the "small" size, chosen on validation loss among 0.1M, 0.59M and 2.85M).
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

- **One machine.** The model has only seen this laptop. The hardware features are constant in the training data, so it can't transfer to other hardware without collecting there (`hpc-ds1 setup`, which has so far only been tested on copies of this laptop's data).
- **Fleet = many independent tasks.** The fleet places whole tasks on machines (a high-throughput cluster); it does not yet split one job across machines.
- **No HTTPS or console login** in v1 (see Quick start).
- **Question encoder:** about 87% on unseen wording and problems, but it can be confidently wrong on "will this one change make it faster?"; use System 1 and System 2 for actual decisions.
- **Ranks are processes, not MPI.** The communication pattern (shard, compute, gather to rank 0) is real, but it runs over pipes on one machine, not over a network.
- **Generalizing to new kinds of work is partial** (see results), and calibration degrades on them. That's exactly where escalating to an engineer matters most.
- **Evaluation replays real measurements** for the hundreds of tuning sessions, because running them all live would take hours. The live sessions show the same behaviour.
- **Telemetry:** memory use and run-to-run noise are recorded but deliberately not model inputs. Both depend on what the measuring process did earlier, not on the job, and they made identical jobs look different live versus during collection.

## Layout

```
hpc_ds1/      workloads, runner, model, System 1, encoder, guard, tuner, API, server, fleet, CLI
scripts/      collect, train, generalization, fit_guard, train_system1, tune_system1, train_encoder, ...
hpc_ds1/ui/          console.html (facility console), index.html (layer explorer)
hpc_ds1/models/      hpc_ds1.pt (step model + guard), system1.pt
hpc_ds1/data/real/   measurements.jsonl (3,168 real runs), hardware.json
hpc_ds1/results/     metrics, generalization, guard, system1, encoder (shown on the Evaluation page)
results/             training logs, out-of-fold estimates, experiments/ (every variant tried)
examples/     plugins/etl_batch.py (bring your own task)
tests/        test_project.py
docs/         DEMO.md, architecture/ (diagram)
archive/      the earlier simulator-based prototype (code only), kept for history
```
