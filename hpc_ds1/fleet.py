"""Fleet: many machines, one console, the model decides where each task runs.

Devices join over the network; no cables, shared filesystem or MPI between them.

    coordinator   hpc-ds1 serve --host 0.0.0.0          (prints a join token)
    each device   hpc-ds1 agent --server http://<coordinator>:8765 --token <token>
                  (or hpc_ds1.fleet.Agent(server, token).run() in Python)
    submit work   hpc-ds1 submit --server ... --token ... --workload gemm --size 2048
                  or the console's Fleet page, or POST /v1/fleet/tasks

An agent registers its hardware and a profile of what it has measured on
itself (`hpc-ds1 setup`): the best runtime of each kernel and size. It then
polls the coordinator for work over plain HTTP, so it works from behind NAT
and needs no inbound port.

Placement. For each task the coordinator estimates, on every online device,

    finish = time its queue still needs + expected runtime of this task there

where the expected runtime is the device's own best measured runtime at the
nearest measured size, scaled by the kernel's work ratio. The task goes to the
device with the earliest finish; every device's estimate is kept on the task,
so the choice can be audited. A device with no measurements of that kernel
gets no estimate and is only used if no device has one. On the chosen device
its own model tunes the job, and moves that aren't "auto" wait for approval
in the console like any other.
"""

import itertools
import json
import math
import secrets
import socket
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from .kernels import KERNELS
from .tuner import LiveRunner, run_tuning

OFFLINE_AFTER_SEC = 30
POLL_SEC = 2.0


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# what a device knows about itself
# ---------------------------------------------------------------------------

def device_profile(hardware):
    """{workload: {size: best measured runtime}} from this machine's own
    measurements. Empty if the measurements on disk came from another machine."""
    try:
        from .dataset import build_tables, data_dir, load_hardware, load_measurements
        measured_on = load_hardware(data_dir() / "hardware.json")
        if measured_on.get("name") != hardware.get("name"):
            return {}
        tables, _ = build_tables(load_measurements(data_dir() / "measurements.jsonl"))
    except (OSError, ValueError):
        return {}
    prof = {}
    for (w, size), table in tables.items():
        prof.setdefault(w, {})[str(size)] = min(t["runtime_sec"] for t in table.values())
    return prof


def estimate_runtime(profile, workload, size):
    """Best runtime this device should reach on (workload, size), from its own
    measurements at the nearest size scaled by work; None if it never ran it."""
    runs = profile.get(workload)
    if not runs or workload not in KERNELS:
        return None
    k = KERNELS[workload]
    near = min(runs, key=lambda s: abs(math.log(int(s)) - math.log(size)))
    return runs[near] * k.work(size) / k.work(int(near))


# ---------------------------------------------------------------------------
# coordinator side
# ---------------------------------------------------------------------------

class Fleet:
    def __init__(self, token=None, audit=None):
        self.token = token or secrets.token_urlsafe(12)
        self.audit = audit
        self.devices, self.tasks = {}, {}
        self.lock = threading.Lock()
        self._dev_ids, self._task_ids = itertools.count(1), itertools.count(1)

    # devices
    def register(self, name, hardware, profile, model):
        with self.lock:
            for d in self.devices.values():  # a device reconnecting keeps its id
                if d["name"] == name and d["hardware"].get("name") == hardware.get("name"):
                    d.update(hardware=hardware, profile=profile, model=model, last_seen=time.time())
                    return d
            d = {"id": f"d{next(self._dev_ids)}", "name": name, "hardware": hardware, "profile": profile,
                 "model": model, "joined_at": _now(), "last_seen": time.time(), "current": None}
            self.devices[d["id"]] = d
            return d

    def online(self, d):
        return time.time() - d["last_seen"] < OFFLINE_AFTER_SEC

    def _queue_sec(self, dev_id):
        return sum(t["placement"]["chosen_estimate_sec"] or 0 for t in self.tasks.values()
                   if t["device"] == dev_id and t["status"] in ("queued", "running", "awaiting_approval"))

    # tasks
    def submit(self, workload, size, config=None, threshold=0.85, escalate_below=0.5, max_steps=8, start=None):
        if workload not in KERNELS:
            raise ValueError(f"unknown workload {workload!r}; choose from {list(KERNELS)}")
        with self.lock:
            candidates = []
            for d in self.devices.values():
                if not self.online(d):
                    continue
                est = estimate_runtime(d["profile"], workload, size)
                wait = self._queue_sec(d["id"])
                candidates.append({"device": d["id"], "name": d["name"], "expected_runtime_sec": est,
                                   "queue_sec": round(wait, 6),
                                   "finish_sec": None if est is None else round(wait + est, 6),
                                   "note": "" if est is not None else f"has not measured {workload} yet"})
            if not candidates:
                raise ValueError("no device is online; connect one with `hpc-ds1 agent`")
            known = [c for c in candidates if c["finish_sec"] is not None]
            if known:
                best = min(known, key=lambda c: c["finish_sec"])
                why = (f"earliest expected finish: {best['finish_sec']:.4g}s "
                       f"(queue {best['queue_sec']:.3g}s + run {best['expected_runtime_sec']:.3g}s)")
            else:
                best = min(candidates, key=lambda c: c["queue_sec"])
                why = "no device has measured this kernel; sent to the least busy one"
            t = {"id": f"t{next(self._task_ids)}", "workload": workload, "size": int(size),
                 "config": config or {"ranks": 1, "threads": 1, "gpu": False, "precision": "fp32"},
                 "start": start or ("given" if config else "system1"),
                 "threshold": threshold, "escalate_below": escalate_below, "max_steps": max_steps,
                 "device": best["device"], "status": "queued", "submitted_at": _now(),
                 "placement": {"candidates": candidates, "reason": why,
                               "chosen_estimate_sec": best["expected_runtime_sec"]},
                 "pending": None, "answer": None, "result": None, "error": None, "events": []}
            self.tasks[t["id"]] = t
            return t

    def poll(self, dev_id):
        with self.lock:
            d = self.devices.get(dev_id)
            if d is None:
                raise KeyError(f"unknown device {dev_id}; register again")
            d["last_seen"] = time.time()
            if d["current"] and self.tasks[d["current"]]["status"] in ("running", "awaiting_approval"):
                return None
            for t in self.tasks.values():
                if t["device"] == dev_id and t["status"] == "queued":
                    t["status"], t["started_at"] = "running", _now()
                    d["current"] = t["id"]
                    return t
            return None

    def update(self, task_id, dev_id, body):
        with self.lock:
            t = self.tasks.get(task_id)
            if t is None or t["device"] != dev_id:
                raise KeyError(f"task {task_id} is not assigned to {dev_id}")
            self.devices[dev_id]["last_seen"] = time.time()
            if "event" in body:
                ev = body["event"]
                if "system1" in ev:
                    t["system1"] = ev["system1"]
                if "decision" in ev and self.audit is not None:
                    row = self.audit.add(f"fleet {task_id} @ {self.devices[dev_id]['name']}", ev["state"], ev["decision"])
                    t["last_audit"] = row["id"]
                    ev = {"decision": {k: ev["decision"].get(k) for k in ("choice", "confidence", "mode", "reasons")},
                          "config": ev["state"]["config"], "runtime_sec": ev["state"]["telemetry"].get("runtime_sec")}
                t["events"].append({"at": _now(), **ev})
            if body.get("pending"):
                t["pending"], t["answer"], t["status"] = {**body["pending"], "since": _now()}, None, "awaiting_approval"
            if body.get("status") in ("done", "failed"):
                t["status"], t["finished_at"] = body["status"], _now()
                t["result"], t["error"] = body.get("result"), body.get("error")
                self.devices[dev_id]["current"] = None
            return {"answer": t["answer"]}

    def approve(self, task_id, approve, by="engineer"):
        with self.lock:
            t = self.tasks.get(task_id)
            if t is None or t["status"] != "awaiting_approval":
                raise ValueError(f"task {task_id} is not waiting for approval")
            t["answer"] = {"approve": bool(approve), "by": by or "engineer"}
            if self.audit is not None and t.get("last_audit"):
                self.audit.resolve(t["last_audit"], approved=bool(approve), approved_by=t["answer"]["by"])
            t["events"].append({"at": _now(), "approval": t["answer"], "action": t["pending"]["action"]})
            t["pending"], t["status"] = None, "running"
            return t

    def queue(self):
        return [{"task": t["id"], "device": self.devices[t["device"]]["name"], "workload": t["workload"],
                 "size": t["size"], **t["pending"]} for t in self.tasks.values() if t["status"] == "awaiting_approval"]

    def snapshot(self):
        with self.lock:
            devs = [{**{k: v for k, v in d.items() if k != "profile"}, "online": self.online(d),
                     "measured": {w: sorted(int(s) for s in sizes) for w, sizes in d["profile"].items()},
                     "seconds_since_seen": round(time.time() - d["last_seen"], 1)} for d in self.devices.values()]
            tasks = sorted(self.tasks.values(), key=lambda t: t["submitted_at"], reverse=True)
            return {"devices": devs, "tasks": tasks}


# ---------------------------------------------------------------------------
# device side
# ---------------------------------------------------------------------------

def _post(server, path, body, token, timeout=30):
    req = urllib.request.Request(server.rstrip("/") + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-HPC-Token": token})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def submit(server, token, workload, size, config=None, **kw):
    """Send a task to the fleet; returns it with the placement decision."""
    return _post(server, "/v1/fleet/tasks", {"workload": workload, "size": size, "config": config, **kw}, token)


class Agent:
    """Runs on each device: registers, pulls tasks, tunes them with this device's
    own model, and reports every run back. Blocks in run()."""

    def __init__(self, server, token, name=None, log=print):
        self.server, self.token, self.log = server.rstrip("/"), token, log
        self.name = name or socket.gethostname()
        self.id = None
        self._defaults = None

    def default_runtime(self, workload, size):
        """Measured runtime of the untuned default (1 rank x 1 thread, CPU, fp32) on this
        machine, from its setup measurements; None if it was never measured."""
        if self._defaults is None:
            self._defaults = {}
            try:
                from .dataset import build_tables, data_dir, load_measurements
                from .space import config_key, make_config
                tables, _ = build_tables(load_measurements(data_dir() / "measurements.jsonl"))
                key = config_key(make_config(1, 1, False, "fp32"))
                self._defaults = {p: t[key]["runtime_sec"] for p, t in tables.items() if key in t}
            except (OSError, ValueError, KeyError):
                pass
        return self._defaults.get((workload, size))

    def _post(self, path, body):
        return _post(self.server, path, body, self.token)

    def register(self, model):
        hw = model.engine.hardware
        prof = device_profile(hw)
        info = model.info()
        d = self._post("/v1/fleet/register", {"name": self.name, "hardware": hw, "profile": prof,
                                              "model": {"trained_on": info.get("trained_on"),
                                                        "parameters": info["parameters"]}})
        self.id = d["id"]
        self.log(f"joined {self.server} as {self.name} ({self.id}); measured kernels: "
                 f"{', '.join(sorted(prof)) or 'none - run `hpc-ds1 setup` so the fleet can place work here'}")
        if info.get("trained_on") != hw.get("name"):
            self.log(f"note: this model was trained on {info.get('trained_on')}, not this machine; "
                     "its familiarity guard will send every move for approval until you run `hpc-ds1 setup`")

    def run_task(self, model, t):
        tid = t["id"]
        self.log(f"task {tid}: {t['workload']} {t['size']} from {t['config']}")

        def report(**body):
            return self._post(f"/v1/fleet/tasks/{tid}/status", {"device": self.id, **body})

        pol = model.policy(t["threshold"], t["escalate_below"])

        def policy(state, exclude=()):
            out = pol(state, exclude)
            report(event={"decision": model.last_decision.to_dict(), "state": state})
            return out

        def on_escalate(action, confidence, state):
            d = model.last_decision
            report(pending={"action": action, "confidence": confidence, "mode": d.mode if d else "confirm",
                            "reasons": d.reasons if d else [], "config": state["config"],
                            "runtime_sec": state["telemetry"]["runtime_sec"]})
            while True:
                time.sleep(POLL_SEC)
                ans = report().get("answer")
                if ans:
                    return bool(ans["approve"])

        try:
            from .system1 import dual_start, gated
            start, gate, pred = t["config"], None, None
            if t.get("start") == "system1":
                start, pred, gate = dual_start(model, t["workload"], t["size"], t["threshold"], t["escalate_below"],
                                               t["config"])
                if pred is not None:
                    report(event={"system1": pred.to_dict()})
                    self.log(f"task {tid}: System 1 picks {pred.config} ({pred.confidence:.0%} sure, {pred.mode})")
            # one unrecorded warm-up run, so the first task on a cold GPU is not timed with start-up costs
            model.engine.measure(t["workload"], t["size"], start, min_total=0.0, max_reps=1)
            runner = LiveRunner(t["workload"], t["size"], model.engine)
            r = run_tuning(gated(policy, gate), t["workload"], t["size"], start, runner, max_steps=t["max_steps"],
                           threshold=t["threshold"], on_escalate=on_escalate)
            result = r.to_dict()
            if t.get("start") == "system1" and pred is not None:
                result["predicted_runtime_sec"] = pred.expected_runtime_sec
            base = self.default_runtime(t["workload"], t["size"])
            if base:
                result["default_runtime_sec"] = base
                result["vs_default"] = round(base / max(r.final_runtime, 1e-12), 2)
            report(status="done", result=result)
            self.log(f"task {tid}: ran {r.final_runtime * 1e3:.2f} ms"
                     + (f", x{result['vs_default']:.1f} vs untuned" if base else "")
                     + f", {r.experiments} run(s) -> {r.final_config}")
        except Exception as e:  # report and keep serving
            report(status="failed", error=str(e))
            self.log(f"task {tid} failed: {e}")

    def run(self, model=None):
        from .api import HPCDS1
        model = model or HPCDS1.load()
        try:
            try:
                self.register(model)
            except urllib.error.URLError as e:
                if isinstance(e, urllib.error.HTTPError) and e.code == 401:
                    raise SystemExit("the coordinator rejected the token: copy it again from `hpc-ds1 serve`")
                raise SystemExit(f"can't reach the coordinator at {self.server} ({getattr(e, 'reason', e)}).\n"
                                 "  check: is `hpc-ds1 serve --host 0.0.0.0` running there, is the IP right "
                                 "(same network?), and is TCP port 8765 allowed through its firewall?")
            while True:
                try:
                    t = self._post("/v1/fleet/poll", {"device": self.id}).get("task")
                except urllib.error.HTTPError as e:
                    if e.code == 404:  # coordinator restarted: join again
                        self.register(model)
                        continue
                    raise
                if t:
                    self.run_task(model, t)
                else:
                    time.sleep(POLL_SEC)
        finally:
            model.close()
