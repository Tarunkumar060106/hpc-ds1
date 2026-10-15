# Live demo script (review, 9 Oct)

About 10–12 minutes. Everything runs on this laptop; nothing is simulated.

## Before the review (10 minutes ahead)

1. Plug the laptop in and set Windows power mode to **Best performance**. Battery mode throttles the CPU and GPU, and timings drift from the training data.
2. Close heavy apps (browser tabs, games, IDE indexing). Don't run `pytest` while the server and agent are up: they compete for the GPU and the tests stall.
3. Terminal 1, the coordinator (keep it open; closing it changes the token):
   ```
   cd D:\Radiants\Projects\kairo
   hpc-ds1 serve --host 0.0.0.0
   ```
   Note the **fleet join token** it prints.
4. Terminal 2, connect this laptop as a device:
   ```
   hpc-ds1 agent --server http://127.0.0.1:8765 --token <TOKEN> --name laptop
   ```
5. Open **http://localhost:8765** and press **Ctrl+F5**. Check the sidebar: "trained on <this machine> · guard on", coordinator dot green. On the Fleet page, `laptop` should be **online** with 8 measured kernels.
6. Type your name into the box at the top of the **Approvals** page.

If the page looks stale or wrong: another server may be on port 8765. Check with `netstat -ano | findstr 8765`.

## The story (slides 1–15, then switch to the console)

### 1. Playground: "ask it like Jev" (Playground page, ~2 min)
- Preset **Stencil 160 · slow start** → **Run**. The job runs for real (telemetry strip), then one forward pass answers five typed questions: `next_move` (Choice with probability rings), `route` (**auto**), `familiar` (true, distance 0.3×), `should_stop`, `expected_gain`.
- Point out the model latency in the header (a few ms; Jev is ~33 ms).

### 2. "It knows when it doesn't know" (still Playground, ~1 min)
- Preset **Unfamiliar · matmul 512** → **Run**. The model is ~99–100% sure it should move to the GPU, yet `route` says **escalate**: "size 512 is outside the measured range 1024–3072", "unlike any measured training run".
- Line: *on never-seen workloads this guard turned 240 confidently wrong automatic moves into 0, without flagging any of 286 familiar decisions.*

### 3. Fleet: "the model decides where it runs" (Fleet page, ~3 min)
- Submit **3D Jacobi stencil, 160, 1 rank, 1 thread**. The task row shows where it was placed and why ("earliest expected finish: queue + run"), then *running*, then *done*: about 20–35× faster in about 6 real runs, ending on GPU · bf16 with "model stop".
- Submit **Neural network training step, 4096, 1×1** (AI training): expect GPU, thousands of times faster in about 3 runs.
- Submit **Backend request handling, 4000, 1×1** (server): it adds worker processes, never uses the GPU (there's no GPU version); about 5× faster.
- If a second machine is available: on it, run `hpc-ds1 agent --server http://<this-laptop-ip>:8765 --token <TOKEN>` (find it with `ipconfig` or `ip addr`). Submit again and show both devices' estimates on the task.

### 4. Approvals + audit: "a human stays in charge" (~2 min)
- Submit **Monte Carlo option pricing, 8,000,000, 1×1**. When the **Approvals** badge lights up, open it: the move, its confidence, the tier (*confirm*), and why. **Approve** it.
- Optionally, from a third terminal, submit a size it never measured (the console form only offers measured sizes):
  ```
  hpc-ds1 submit --server http://127.0.0.1:8765 --token <TOKEN> --workload gemm --size 512 --threads 2
  ```
  It lands in Approvals as **escalate** with the reasons. **Reject** it: nothing runs, and the task stops at its starting config.
- **Audit log** page: every decision, its tier, and *approved by <your name>*.

### 5. Evaluation + machine (~1 min)
- **Evaluation**: 92.3% best moves vs 70.3% for rules; 0% slower moves; tuning-session table; the guard table on never-seen workloads.
- **Machine & setup**: what was measured here, and the `pip install` / `hpc-ds1 setup` commands for a new machine.
- Optional: **Layer explorer** in the sidebar, if they ask how it decides inside.

Close on the roadmap: *next level is a true System 1. It answers instantly from the job description with zero runs; System 2 (the measured tuning loop) checks only when it's unsure, and every System 2 session teaches System 1.*

## Likely questions

- **Is this MPI?** No. Ranks are separate processes with their own memory that exchange results over pipes, the same distributed-memory pattern MPI uses, on one machine. Across machines, the fleet places whole tasks over HTTP.
- **Can it handle any task?** Any task once it has been measured on that machine. A facility adds its own as a plug-in (`~/.hpc_ds1/plugins/`, example `examples/plugins/etl_batch.py`), runs `hpc-ds1 setup`, and the model learns it. Until then the guard escalates it. Held out of training, server work was decided right only 18% of the time, which is exactly why it asks.
- **What got worse when you added AI and server work?** Calibration error is 0.141, above the 0.12 target (it was 0.099 with six kernels). Auto-run accuracy is still 95%.
- **Why replay measurements in evaluation?** Hundreds of sessions × several runs would take hours live. Replay uses the same real measurements, and all 8 live sessions ended within 0.5% of the best known config or beat it.
- **What if a suggested config crashes?** The tuner runs it, sees the failure or the wrong answer (checked against an fp32 reference), reverts, and asks again; it never picked a crashing config in evaluation.
- **Is the network secure?** Remote agents and clients need the join token printed by `serve`; agents poll the coordinator, so devices need no open ports. It's a lab setup, not hardened for the internet.
