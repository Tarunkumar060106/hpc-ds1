"""Command line: hpc-ds1 (or python -m hpc_ds1) {setup,serve,agent,submit,decide,tune,tasks,info}"""

import argparse
import json
import warnings

from .api import DEFAULT_ESCALATE_BELOW, DEFAULT_THRESHOLD, HPCDS1
from .kernels import KERNELS

warnings.filterwarnings("ignore", category=UserWarning)


def _job_args(p):
    p.add_argument("--workload", required=True, choices=list(KERNELS))
    p.add_argument("--size", type=int, required=True, help="problem size (see `info` for each kernel's unit)")
    p.add_argument("--ranks", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="auto-run at or above this confidence")
    p.add_argument("--escalate-below", type=float, default=DEFAULT_ESCALATE_BELOW,
                   help="below this confidence, escalate; in between, ask to confirm")
    p.add_argument("--checkpoint")


def _config(a):
    return {"ranks": a.ranks, "threads": a.threads, "gpu": a.gpu, "precision": "bf16" if a.bf16 else "fp32"}


def main():
    ap = argparse.ArgumentParser(prog="hpc-ds1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("setup", help="measure this machine and train HPC-DS1 on it (~1 h, or ~30 min with --quick)")
    u.add_argument("--quick", action="store_true", help="3 sizes per kernel instead of 6")
    u.add_argument("--workloads", help=f"comma-separated subset of {','.join(KERNELS)}")
    u.add_argument("--home", help="where to keep measurements and the model (default ~/.hpc_ds1)")
    u.add_argument("--skip-measure", action="store_true", help="retrain on measurements already taken")
    s = sub.add_parser("serve", help="HTTP API + live layer explorer")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--checkpoint")
    g = sub.add_parser("agent", help="connect this machine to a coordinator; it runs the tasks placed on it")
    g.add_argument("--server", required=True, help="coordinator URL, e.g. http://192.168.1.20:8765")
    g.add_argument("--token", required=True, help="join token printed by `hpc-ds1 serve`")
    g.add_argument("--name", help="device name (default: host name)")
    g.add_argument("--checkpoint")
    m = sub.add_parser("submit", help="send a task to the fleet; the coordinator picks the device")
    m.add_argument("--server", required=True)
    m.add_argument("--token", required=True)
    _job_args(m)
    d = sub.add_parser("decide", help="run the job once for real, then ask the model what to change")
    _job_args(d)
    d.add_argument("--candidates", help="comma-separated subset of actions")
    t = sub.add_parser("tune", help="closed-loop tuning; every step runs on this machine")
    _job_args(t)
    t.add_argument("--max-steps", type=int, default=8)
    k = sub.add_parser("tasks", help="workloads HPC-DS1 knows, which ones the model was trained on, and plugins")
    k.add_argument("--checkpoint")
    i = sub.add_parser("info", help="model, hardware and kernels")
    i.add_argument("--checkpoint")
    args = ap.parse_args()

    if args.cmd == "setup":
        from .onboard import setup
        setup(args.home, quick=args.quick, workloads=args.workloads.split(",") if args.workloads else None,
              collect_data=not args.skip_measure, log=lambda m: print(m, flush=True))
        return

    if args.cmd == "agent":
        from .fleet import Agent
        try:
            Agent(args.server, args.token, args.name, log=lambda x: print(x, flush=True)).run(
                HPCDS1.load(args.checkpoint))
        except KeyboardInterrupt:
            pass
        return

    if args.cmd == "submit":
        from .fleet import submit
        t = submit(args.server, args.token, args.workload, args.size, _config(args),
                   threshold=args.threshold, escalate_below=args.escalate_below)
        print(f"task {t['id']} -> {next(c['name'] for c in t['placement']['candidates'] if c['device'] == t['device'])}: "
              f"{t['placement']['reason']}")
        for c in t["placement"]["candidates"]:
            est = f"{c['finish_sec']:.4g}s" if c["finish_sec"] is not None else c["note"]
            print(f"  {c['name']:20s} {est}")
        return

    if args.cmd == "tasks":
        from .kernels import PLUGINS
        trained = []
        try:
            trained = HPCDS1.load(args.checkpoint, device="cpu").net.workloads
        except (FileNotFoundError, ValueError) as e:
            print(f"(no usable model: {e})")
        for w, k in KERNELS.items():
            mark = "trained" if w in trained else "NOT TRAINED - run `hpc-ds1 setup --workloads " + w + "`"
            print(f"{w:12s} {k.display_name:36s} sizes {k.sizes[0]:,}-{k.sizes[-1]:,} {k.size_unit}   [{mark}]")
        print("\nplugins loaded: " + (", ".join(PLUGINS) or "none (see examples/plugins/)"))
        return

    if args.cmd == "serve":
        from .server import serve
        return serve(args.host, args.port, args.checkpoint)

    try:
        model = HPCDS1.load(args.checkpoint)
    except FileNotFoundError as e:
        raise SystemExit(f"hpc-ds1: {e}")
    try:
        if args.cmd == "info":
            print(json.dumps({**model.info(), "hardware": model.engine.hardware,
                              "kernels": {w: {"sizes": k.sizes, "unit": k.size_unit, "character": k.character}
                                          for w, k in KERNELS.items()}}, indent=2))
        elif args.cmd == "decide":
            state = model.observe(args.workload, args.size, _config(args))
            t = state["telemetry"]
            print(f"measured: {t['runtime_sec']:.4f}s  cpu {t['cpu_util_pct']}%  gpu {t['gpu_busy_pct']}%  "
                  f"comm {t['comm_overhead_pct']}%\n")
            cands = args.candidates.split(",") if args.candidates else None
            d = model.decide(state, cands, args.threshold, args.escalate_below)
            print(json.dumps(d.to_dict(), indent=2))
            print(f"\n{d.mode.upper()}: {d.choice} ({d.confidence:.0%} sure)"
                  + "".join(f"\n  - {r}" for r in d.reasons))
        elif args.cmd == "tune":
            def ask(action, confidence, state):
                d = model.last_decision
                why = "".join(f"\n    - {r}" for r in d.reasons) if d and d.reasons else ""
                tier = "ESCALATED" if d and d.mode == "escalate" else "please confirm"
                answer = input(f"  [{tier}] model is {confidence:.0%} sure about {action}{why}\n  run it? [y/N] ")
                return answer.strip().lower() == "y"

            r = model.tune(args.workload, args.size, _config(args), max_steps=args.max_steps, threshold=args.threshold,
                           escalate_below=args.escalate_below, on_escalate=ask)
            for st in r.steps:
                after = f"{st.runtime_after:.4f}s" if st.runtime_after is not None else "-"
                print(f"step {st.step + 1}: {st.action:15s} conf {st.confidence:.3f}  {st.runtime_before:.4f}s -> {after}"
                      f"  {'kept' if st.accepted and st.action != 'STOP_TUNING' else st.note or ('done' if st.action == 'STOP_TUNING' else 'rejected')}")
            print(f"\nx{r.speedup:.2f} faster in {r.experiments} real runs (stopped: {r.stopped_reason}); "
                  f"final config {r.final_config}")
    finally:
        model.close()


if __name__ == "__main__":
    main()
