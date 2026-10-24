# Changelog

## v1.1.1 — console fix

- Fixed: the web console was blank in v1.1.0. An apostrophe in the new "can't run it" label broke the page's JavaScript. A test now syntax-checks the console's scripts.

## v1.1.0 — machines that can't run a job, real AI workloads, readable output

- **The fleet knows what each machine can run.** Before comparing finish times, the coordinator sets aside devices that can't run a job, and says why in plain words: the job needs more GPU memory than the device can really use, every setting failed at that size, or the job already failed there. The memory need is the real peak GPU memory measured for that job on any device (or fitted from measured sizes). Each device learns its *usable* GPU memory from its own runs: the laptop RTX 4050 has 6 GB, but measured runs slow down sharply from 3.0 GB (Windows and the display keep part of the card; the rest spills into system RAM). A job that fails on a device anyway (out of memory, crash) is re-queued automatically on the next device that can run it. The CLI and console show every redirect.
- **Two real AI training workloads whose memory grows with the batch:** `llm_train` (one training step of a GPT-style language model, 63M parameters, 256-token sequences, batch 4–128) and `cnn_train` (ResNet-18-style image model on 112×112 images, batch 64–2048). Measured on the RTX 4050: up to 2.5 GB they run at full speed, above that they spill and slow down (language model batch 96: 8.6 s per step), and batch 128 runs out of memory.
- **Peak GPU memory and out-of-memory failures are recorded** in every measurement (`gpu_peak_gb`, `oom`). Data collection protects the machine: CPU settings of GPU-class workloads expected to exceed 60 s (judged from a real all-cores run) and runs that wouldn't fit in free RAM are skipped and recorded as skipped, never made up. If the CPU fp32 reference doesn't fit, the fp32 GPU run is the reference.
- **Benchmark dataset:** `scripts/export_dataset.py` writes every real run on every machine to `dataset/hpc_ds1_runs.csv` (with `machines.json` and a README), with neutral machine names.
- **Retrained on 10 workloads, 90 problems.** Step model: best move 95.0% (rules 74.8%), tuning 95.6% of best in 2.9 runs. System 1: 88.8% of instant picks within 5% of best, 97.7% of best speed, 0 crashing picks; its confidence is calibrated (0.013) but does not rank picks (AUROC 0.43). Guard: 100% of decisions on every never-seen workload escalated, 298 confident mistakes → 0, 0 of 337 familiar decisions flagged. Benchmark: System 1's first pick 99.7% of best (median), Bayesian optimisation needs about 5 runs; System 1 → System 2 uses about 39× less tuning compute than TPE.
- **Colour-coded terminal output** for every command (AUTO / CONFIRM / ESCALATE badges, readable settings, ✓ ! ✗), off when piped or with `NO_COLOR`.
- **Plain-language README** (what it does, System 1 vs System 2, how it differs from a normal classifier) and three new slides.
- Fixed: a rank process that died at start-up (e.g. Windows out of memory) made measurement wait forever; it now fails that run and starts a fresh worker group.
- Not re-run for this release: the live tuning sessions and the leave-one-workload-out study (v1.0 numbers kept); the question encoder is unchanged.

## v1.0.2 — new machines get System 1, clearer fleet results

- **`hpc-ds1 setup` now trains System 1 too.** Before, setup on a new machine trained only the step model, so System 1 kept answering with the bundled laptop model (e.g. on an RTX 5090 it picked the CPU for Monte Carlo and reached 21% of the best speed; after training on that machine it picks the GPU, 92% sure). Setup also uses the small step model chosen on validation (0.59M parameters) instead of the 2.85M one.
- **Fleet task results** show the speed-up against the untuned default (1 rank × 1 thread, CPU, fp32) measured on that device, and the actual runtime next to System 1's prediction, instead of "×1.00", which was relative to System 1's own pick.
- **Agents do one unrecorded warm-up run** before each task, so the first job on a cold GPU is not timed with start-up costs (seen on the lab RTX 5090: 68% of best cold, 98% warm).
- **Agents print a short hint** when they can't reach the coordinator (IP, same network, port 8765) or the token is wrong, instead of a traceback.
- **Grounded change questions:** yes/no questions about the effect of one change ("would the graphics card make this faster?") are now answered from System 1's estimate for that change, found with a transparent lexicon, instead of by the encoder. On held-out questions: 83.8% vs 68.1% (unseen wording), 82.8% vs 76.6% (unseen sizes), 85.2% vs 77.8% (both). Answers report their `source`. Caveat: the lexicon was written knowing the held-out synonyms, so its coverage still needs a blind test with human-written questions (`scripts/eval_router.py`).
- Fixed: the v1.0.0 notes said the step model has 2.85M parameters; the model chosen on validation has 0.59M.

## v1.0.1 — installable from PyPI

- `pip install hpc-ds1` now gives a complete install: the console, the trained models, the measurements and the evaluation results ship inside the package (`hpc_ds1/ui`, `models`, `data`, `results`; one place for all paths in `hpc_ds1/paths.py`).
- The question-encoder weights stay a GitHub release download (PyPI's file limit); unzip them into `~/.hpc_ds1`, or set `HPC_DS1_ENCODER`.
- `[encoder]` extra installs `transformers`; PyPI metadata (licence, classifiers, project URL).
- Publishing is automated with PyPI Trusted Publishing (`.github/workflows/publish.yml`) on each GitHub release.

## v1.0.0 — first public release

HPC-DS1 is a confidence-aware decision model that tunes compute jobs (ranks, threads, CPU or GPU, fp32 or bf16) on the machine it runs on, trained only on real measurements.

### Models
- **System 2, the step model:** picks the next change for a measured job, one move at a time, with calibrated confidence (0.59M parameters, size chosen on validation). On held-out problem sizes: a best move 94.1% of the time (rules of thumb 70.3%), 96.3% of the best configuration's speed in 2.9 real runs, 0% moves that made a job slower or crashed, calibration error 0.106.
- **System 1, instant configuration:** an 8-network ensemble performance model that picks the whole configuration from the job description alone, in about 10 ms. 83% of picks within 5% of the best, 96.9% of best speed, no crashing picks (cross-validated over 72 problems).
- **Dual process:** System 1 picks, the job runs once, and System 2 checks only when needed; 98.7% of best speed in 1.0 real run on held-out sizes (System 2 alone: 99.1% in 5.1 runs).
- **Familiarity guard:** escalates jobs, sizes and machines unlike the training data; on never-seen workloads 240 confidently wrong automatic moves became 0, with no familiar decisions flagged.
- **Question encoder:** a fine-tuned ModernBERT-base answers free-text questions about a job, Jev-style (yes/no/unclear or a choice), reading System 1's estimates; 87% on unseen wording and problem sizes. Weights are a release asset.

### Data
- 3,168 real timed runs over 72 problems: 8 workloads (6 HPC kernels, a neural-network training step, a backend request batch) × 9 sizes × 44 configurations, on one laptop (Intel i5, RTX 4050 Laptop GPU). Every result is checked against an fp32 reference.

### Tools
- `hpc-ds1` CLI: `setup`, `serve`, `predict`, `plan`, `decide`, `tune`, `learn`, `agent`, `submit`, `tasks`, `info`.
- Facility console: playground (typed and free-text questions), tuning sessions, approval queue, audit log, fleet, machine, evaluation, and a layer explorer.
- Fleet: machines join over HTTP (agents poll, no inbound ports); each task goes to the machine with the earliest expected finish, and that machine's own model tunes it.
- Bring your own task: `@register` a workload in `~/.hpc_ds1/plugins/`; level 5 logs every live run and `hpc-ds1 learn` retrains System 1 on them.

### Known limitations
- Trained on one laptop; on other hardware run `hpc-ds1 setup` first (not yet tested end to end on other hardware).
- No HTTPS or console login: use a trusted network or an overlay such as Tailscale.
- The fleet runs whole tasks per machine; it does not split one job across machines (no MPI yet).
- System 1's confidence barely separates good picks from bad (AUROC about 0.55); System 2's check covers it.
- The question encoder can be confidently wrong on "will this one change make it faster?".
