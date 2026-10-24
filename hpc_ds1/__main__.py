"""Command line: hpc-ds1 (or python -m hpc_ds1) {setup,serve,agent,submit,predict,plan,learn,decide,tune,tasks,info}"""

import argparse
import json
import warnings

from . import style as S
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
    m.add_argument("--start", choices=["system1", "given"], default="system1",
                   help="start from System 1's instant pick on the chosen device (default) or from the given config")
    p = sub.add_parser("predict", help="System 1: the full configuration for a job, instantly, without running it")
    p.add_argument("--workload", required=True, choices=list(KERNELS))
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--checkpoint")
    n = sub.add_parser("plan", help="dual process: System 1 picks, System 2 tunes only if System 1 is unsure")
    n.add_argument("--workload", required=True, choices=list(KERNELS))
    n.add_argument("--size", type=int, required=True)
    n.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    n.add_argument("--checkpoint")
    sub.add_parser("learn", help="System 2 teaches System 1: retrain the instant model on every real run made here")
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
              collect_data=not args.skip_measure, log=S.log)
        return

    if args.cmd == "agent":
        from .fleet import Agent
        try:
            Agent(args.server, args.token, args.name, log=S.log).run(
                HPCDS1.load(args.checkpoint))
        except KeyboardInterrupt:
            pass
        return

    if args.cmd in ("submit", "agent") and not str(args.server).startswith(("http://", "https://")):
        raise SystemExit(f"hpc-ds1: --server must look like http://<coordinator-ip>:8765 (got {args.server!r})")

    if args.cmd == "submit":
        from .fleet import submit
        t = submit(args.server, args.token, args.workload, args.size, _config(args),
                   threshold=args.threshold, escalate_below=args.escalate_below, start=args.start)
        chosen = next(c["name"] for c in t["placement"]["candidates"] if c["device"] == t["device"])
        print(f"{S.good(S.OK)} {S.bold('task ' + t['id'])} {S.dim(S.ARROW)} {S.paint(chosen, 'bold', 'magenta')}  "
              + (S.warn("redirected: " + ", ".join(c["name"] for c in t["placement"]["candidates"] if not c.get("can_run", True))
                        + " can't run it") if any(not c.get("can_run", True) for c in t["placement"]["candidates"])
                 else S.dim("earliest expected finish")), flush=True)
        need = t["placement"].get("memory_need_gb")
        if need is not None:
            print(f"  {S.dim('needs')} ~{S.num(f'{need:.1f} GB')} {S.dim('of GPU memory')}")
        for c in sorted(t["placement"]["candidates"], key=lambda c: (not c.get("can_run", True), c["finish_sec"] is None,
                                                                     c["finish_sec"] or 0)):
            mark = S.good(S.DOT) if c["device"] == t["device"] else S.dim(S.DOT)
            if not c.get("can_run", True):
                print(f"  {S.bad(S.BAD)} {c['name']:18s} {S.bad("can't run it:")} "
                      f"{c['why_not']}")
                continue
            if c["finish_sec"] is None:
                print(f"  {mark} {c['name']:18s} {S.dim(c['note'])}")
                continue
            q, run = c.get("queue_sec") or 0.0, c.get("expected_runtime_sec")
            detail = S.dim(f"queue {S.secs(q)} + run {S.secs(run)}") if run is not None else ""
            name = S.bold(c["name"].ljust(18)) if c["device"] == t["device"] else c["name"].ljust(18)
            print(f"  {mark} {name} {S.num(S.secs(c['finish_sec'])):>10}  {detail}")
        return

    if args.cmd == "learn":
        from .learn import learn
        try:
            prev = HPCDS1.load(device="cpu").system1
        except (FileNotFoundError, ValueError):
            prev = None
        learn(prev, log=S.log)
        return

    if args.cmd == "tasks":
        from .kernels import PLUGINS
        trained = []
        try:
            trained = HPCDS1.load(args.checkpoint, device="cpu").net.workloads
        except (FileNotFoundError, ValueError) as e:
            S.item(f"no usable model: {e}", "warn")
        S.header("Workloads", f"{len(KERNELS)} known")
        for w, k in KERNELS.items():
            mark = S.good(S.OK + " trained") if w in trained else S.warn(f"{S.WARN} not trained") + S.dim(f"  hpc-ds1 setup --workloads {w}")
            print(f"  {S.accent(w.ljust(12))} {k.display_name:36s} {S.dim(f'{k.sizes[0]:,}-{k.sizes[-1]:,} {k.size_unit}'.ljust(34))} {mark}")
        print(f"\n  {S.dim('plugins:')} " + (", ".join(PLUGINS) or S.dim("none (see examples/plugins/)")))
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
        elif args.cmd == "predict":
            p = model.predict(args.workload, args.size, threshold=args.threshold)
            S.header(f"System 1 {S.ARROW} {args.workload} {args.size:,}", f"{p.latency_ms:.1f} ms, nothing run")
            S.kv("configuration", S.cfg(p.config))
            S.kv("expected runtime", S.num(S.secs(p.expected_runtime_sec)))
            S.kv("confidence", f"{S.num(S.pct(p.confidence))} sure it is within 5% of the best  {S.mode(p.mode)}")
            for r in p.reasons:
                S.item(S.warn(r), "warn")
            if p.alternatives:
                S.kv("alternatives", "")
                for a in p.alternatives:
                    print(f"    {S.dim(S.DOT)} {S.cfg(a['config'])}  {S.dim('~' + S.secs(a['expected_runtime_sec']))}")
            print()
        elif args.cmd == "plan":
            def ask(action, confidence, state):
                d = model.last_decision
                why = "".join(f"\n    - {r}" for r in (d.reasons if d else state.get("prediction", {}).get("reasons", [])))
                return input(f"  {S.mode('confirm')} {S.bold(action)} ({S.pct(confidence)} sure){why}\n  run it? [y/N] ").strip().lower() == "y"

            pred, r, used2 = model.plan(args.workload, args.size, threshold=args.threshold, on_escalate=ask)
            S.header(f"Plan {S.ARROW} {args.workload} {args.size:,}")
            S.kv("System 1 picked", f"{S.cfg(pred.config)}  {S.num(S.pct(pred.confidence))} sure  {S.mode(pred.mode)}  "
                                    f"{S.dim(f'{pred.latency_ms:.1f} ms')}")
            if r is None:
                S.item(S.warn("not run: unfamiliar job and no approval"), "warn")
            else:
                S.kv("predicted", S.secs(pred.expected_runtime_sec))
                S.kv("ran", f"{S.num(S.secs(r.final_runtime))} in {r.experiments} real run{'s' if r.experiments != 1 else ''}")
                S.kv("final", S.cfg(r.final_config))
                S.item("System 2 tuned from there" if used2 else
                       "kept as is: System 1 was confident and the run confirmed it", "ok")
            print()
        elif args.cmd == "decide":
            state = model.observe(args.workload, args.size, _config(args))
            t = state["telemetry"]
            S.header(f"Measured {S.ARROW} {args.workload} {args.size:,}", S.cfg(_config(args)))
            S.kv("runtime", S.num(S.secs(t["runtime_sec"])))
            S.kv("cpu / gpu / comm", f"{t['cpu_util_pct']}% / {t['gpu_busy_pct']}% / {t['comm_overhead_pct']}%")
            print()
            cands = args.candidates.split(",") if args.candidates else None
            d = model.decide(state, cands, args.threshold, args.escalate_below)
            print(json.dumps(d.to_dict(), indent=2))
            print(f"\n{S.mode(d.mode)} {S.bold(d.choice)}  {S.num(S.pct(d.confidence))} sure")
            for r in d.reasons:
                S.item(S.warn(r), "warn")
        elif args.cmd == "tune":
            def ask(action, confidence, state):
                d = model.last_decision
                why = "".join(f"\n    - {r}" for r in d.reasons) if d and d.reasons else ""
                tier = S.mode("escalate" if d and d.mode == "escalate" else "confirm")
                answer = input(f"  {tier} model is {S.pct(confidence)} sure about {S.bold(action)}{why}\n  run it? [y/N] ")
                return answer.strip().lower() == "y"

            r = model.tune(args.workload, args.size, _config(args), max_steps=args.max_steps, threshold=args.threshold,
                           escalate_below=args.escalate_below, on_escalate=ask)
            S.header(f"Tuning {S.ARROW} {args.workload} {args.size:,}", f"from {S.cfg(_config(args))}")
            for st in r.steps:
                if st.action == "STOP_TUNING":
                    print(f"  {S.dim(f'{st.step + 1:>2}')} {S.good(S.OK)} {S.bold('stop')}  {S.dim(f'{st.confidence:.0%} sure')}")
                    continue
                kept = st.accepted
                after = S.secs(st.runtime_after) if st.runtime_after is not None else "-"
                verdict = S.good("kept") if kept else S.bad(st.note or "rejected")
                print(f"  {S.dim(f'{st.step + 1:>2}')} {(S.good(S.OK) if kept else S.bad(S.BAD))} {st.action:15s} "
                      f"{S.dim(f'{st.confidence:.0%}'):>4}  {S.secs(st.runtime_before)} {S.dim(S.ARROW)} {after}  {verdict}")
            print(f"\n  {S.good(S.bold(f'x{r.speedup:.2f} faster'))} in {r.experiments} real runs  "
                  f"{S.dim('(' + r.stopped_reason.replace('_', ' ') + ')')}")
            S.kv("final", S.cfg(r.final_config))
            print()
    finally:
        model.close()


if __name__ == "__main__":
    main()
