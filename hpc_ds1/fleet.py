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
so the choice can be audited.

Can the device run it at all? Before comparing finish times, devices that
cannot run the job are set aside, with the reason in plain words:
  - the job needs more GPU memory than the device has. The need is the real
    peak GPU memory measured for that job and size on any device of the fleet
    (or fitted from the sizes measured, or the kernel's own estimate); above
    its memory a GPU spills into system RAM and runs many times slower;
  - every setting failed at that size when the device measured it;
  - the job already failed on that device (out of memory, crash): the task is
    then re-queued automatically on the next device that can run it. A device with no measurements of that kernel
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


def device_limits(hardware):
    """What this machine's own measurements say it can and cannot run:
    {"gpu_peak_gb": {workload: {size: GB}}, "failed": {workload: [sizes]}} where
    gpu_peak_gb is the smallest real peak GPU memory of a correct GPU run and
    failed lists sizes at which no setting ran correctly."""
    try:
        from .dataset import data_dir, load_hardware, load_measurements
        measured_on = load_hardware(data_dir() / "hardware.json")
        if measured_on.get("name") != hardware.get("name"):
            return {}
        rows = load_measurements(data_dir() / "measurements.jsonl")
    except (OSError, ValueError):
        return {}
    peak, ok, seen = {}, set(), set()
    for r in rows:
        key = (r["workload"], int(r["size"]))
        seen.add(key)
        if r.get("ok") and r.get("correct"):
            ok.add(key)
            g = (r.get("telemetry") or {}).get("gpu_peak_gb")
            if r["config"].get("gpu") and g:
                w = peak.setdefault(r["workload"], {})
                w[str(key[1])] = min(g, w.get(str(key[1]), g))
    failed = {}
    for w, s in sorted(seen - ok):
        failed.setdefault(w, []).append(s)
    return {"gpu_peak_gb": peak, "failed": failed, "gpu_usable_gb": usable_gpu_memory(rows)}


SPILL_SLOWDOWN = 2.0  # a size this much slower per unit of work than the device's best is spilling


def usable_gpu_memory(rows):
    """GPU memory this device can really use at full speed, from its own runs: the
    smallest peak memory at which a GPU-class job got SPILL_SLOWDOWN times slower per
    unit of work than at its smaller sizes (Windows and the display keep part of
    the card; above the rest the driver spills into system RAM). None if no run
    has slowed down that way."""
    best = {}
    for r in rows:
        k = KERNELS.get(r["workload"])
        t = r.get("telemetry") or {}
        if not (getattr(k, "gpu_class", False) and r["config"].get("gpu") and r.get("ok") and r.get("correct")
                and t.get("gpu_peak_gb")):
            continue
        key = (r["workload"], int(r["size"]))
        if key not in best or t["runtime_sec"] < best[key][0]:
            best[key] = (t["runtime_sec"], t["gpu_peak_gb"])
    spill = []
    for w in {w for w, _ in best}:
        sizes = sorted(s for x, s in best if x == w)
        per = {s: best[(w, s)][0] / KERNELS[w].work(s) for s in sizes}
        for i, s in enumerate(sizes[1:], start=1):
            if per[s] > SPILL_SLOWDOWN * min(per[x] for x in sizes[:i]):
                spill.append(best[(w, s)][1])
                break
    return round(min(spill), 2) if spill else None


def memory_need_gb(devices, workload, size):
    """GPU memory a job needs, and where that number comes from: measured on a
    device at this size, fitted from the sizes measured (power law), or the
    kernel's own estimate. None if nothing is known."""
    pts = {}
    for d in devices:
        for s, g in ((d.get("limits") or {}).get("gpu_peak_gb") or {}).get(workload, {}).items():
            pts[int(s)] = min(g, pts.get(int(s), g))
    if size in pts:
        src = [d["name"] for d in devices
               if str(size) in ((d.get("limits") or {}).get("gpu_peak_gb") or {}).get(workload, {})]
        return pts[size], f"measured on {', '.join(src)}"
    if len(pts) >= 2:
        xs = sorted(pts)
        lx = [math.log(x) for x in xs]
        ly = [math.log(pts[x]) for x in xs]
        mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
        b = sum((a - mx) * (c - my) for a, c in zip(lx, ly)) / max(1e-12, sum((a - mx) ** 2 for a in lx))
        return math.exp(my + b * (math.log(size) - mx)), f"fitted from {len(xs)} measured sizes"
    k = KERNELS.get(workload)
    if k is not None and hasattr(k, "memory_gb"):
        return k.memory_gb(size, "bf16"), "the kernel's own estimate"
    return None, None


def can_run(device, workload, size, need):
    """(True, None) or (False, plain-language reason)."""
    lim = device.get("limits") or {}
    key = f"{workload}:{size}"
    if key in device.get("cant_run", {}):
        return False, f"it already failed there: {device['cant_run'][key]}"
    if size in (lim.get("failed") or {}).get(workload, []):
        return False, "every setting failed at this size when it was measured"
    hw = device.get("hardware") or {}
    mem = hw.get("gpu_memory_gb") or 0
    k = KERNELS.get(workload)
    if need is not None and getattr(k, "gpu_class", False):
        if not mem:
            return False, f"needs a GPU with {need:.1f} GB; this machine has none"
        usable = lim.get("gpu_usable_gb")
        gpu = hw.get("gpu_model") or "its GPU"
        if usable and need >= usable:
            return False, (f"needs {need:.1f} GB of GPU memory; {gpu} has {mem:.0f} GB but measured runs slow down "
                           f"sharply from {usable:.1f} GB (the rest spills into system RAM)")
        if need > 0.95 * mem:
            return False, (f"needs {need:.1f} GB of GPU memory, {gpu} has {mem:.0f} GB: "
                           "it would spill into system RAM and run many times slower")
    return True, None


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
    def register(self, name, hardware, profile, model, limits=None):
        with self.lock:
            for d in self.devices.values():  # a device reconnecting keeps its id
                if d["name"] == name and d["hardware"].get("name") == hardware.get("name"):
                    d.update(hardware=hardware, profile=profile, model=model, limits=limits or {},
                             last_seen=time.time())
                    return d
            d = {"id": f"d{next(self._dev_ids)}", "name": name, "hardware": hardware, "profile": profile,
                 "model": model, "limits": limits or {}, "cant_run": {}, "joined_at": _now(),
                 "last_seen": time.time(), "current": None}
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
            best, candidates, why, need = self._place(workload, int(size))
            t = {"id": f"t{next(self._task_ids)}", "workload": workload, "size": int(size),
                 "config": config or {"ranks": 1, "threads": 1, "gpu": False, "precision": "fp32"},
                 "start": start or ("given" if config else "system1"),
                 "threshold": threshold, "escalate_below": escalate_below, "max_steps": max_steps,
                 "device": best["device"], "status": "queued", "submitted_at": _now(),
                 "placement": {"candidates": candidates, "reason": why, "memory_need_gb": need,
                               "chosen_estimate_sec": best["expected_runtime_sec"]},
                 "pending": None, "answer": None, "result": None, "error": None, "events": [], "redirects": []}
            self.tasks[t["id"]] = t
            return t

    def _place(self, workload, size, exclude=()):
        """Pick the device for a job: drop devices that can't run it, then the earliest
        expected finish. Returns (chosen candidate, all candidates, reason, memory need)."""
        online = [d for d in self.devices.values() if self.online(d) and d["id"] not in exclude]
        if not online:
            raise ValueError("no device is online; connect one with `hpc-ds1 agent`")
        need, need_src = memory_need_gb(list(self.devices.values()), workload, size)
        candidates = []
        for d in online:
            est = estimate_runtime(d["profile"], workload, size)
            wait = self._queue_sec(d["id"])
            ok, why_not = can_run(d, workload, size, need)
            candidates.append({"device": d["id"], "name": d["name"], "expected_runtime_sec": est,
                               "queue_sec": round(wait, 6),
                               "finish_sec": None if est is None else round(wait + est, 6),
                               "can_run": ok, "why_not": why_not,
                               "note": "" if est is not None else f"has not measured {workload} yet"})
        able = [c for c in candidates if c["can_run"]]
        pool = able or candidates
        known = [c for c in pool if c["finish_sec"] is not None]
        if known:
            best = min(known, key=lambda c: c["finish_sec"])
            why = (f"earliest expected finish: {best['finish_sec']:.4g}s "
                   f"(queue {best['queue_sec']:.3g}s + run {best['expected_runtime_sec']:.3g}s)")
        else:
            best = min(pool, key=lambda c: c["queue_sec"])
            why = "no device that can run it has measured this kernel; sent to the least busy one"
        if not able:
            why = "no device can comfortably run it; " + why
        rejected = [c for c in candidates if not c["can_run"]]
        if rejected and able:
            why = "; ".join(f"{c['name']} can't: {c['why_not']}" for c in rejected) + f" -> {best['name']}, " + why
        if need is not None:
            why += f" [needs ~{need:.1f} GB GPU memory, {need_src}]"
        return best, candidates, why, None if need is None else round(need, 3)

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
                dev = self.devices[dev_id]
                dev["current"] = None
                if body["status"] == "failed" and self._requeue(t, dev, body.get("error") or "failed"):
                    return {"answer": None}
                t["status"], t["finished_at"] = body["status"], _now()
                t["result"], t["error"] = body.get("result"), body.get("error")
            return {"answer": t["answer"]}

    def _requeue(self, t, dev, error):
        """A task that failed on a device moves to the next device that can run it."""
        short = error if len(error) < 160 else error[:157] + "..."
        dev.setdefault("cant_run", {})[f"{t['workload']}:{t['size']}"] = short
        tried = {r["device"] for r in t["redirects"]} | {dev["id"]}
        try:
            best, candidates, why, need = self._place(t["workload"], t["size"], exclude=tried)
        except ValueError:
            return False
        if not best["can_run"]:
            return False
        t["redirects"].append({"device": dev["id"], "from": dev["name"], "to": best["name"], "reason": short,
                               "at": _now()})
        t["device"], t["status"] = best["device"], "queued"
        t["placement"] = {"candidates": candidates, "memory_need_gb": need,
                          "reason": f"failed on {dev['name']} ({short}); re-queued -> {best['name']}, {why}",
                          "chosen_estimate_sec": best["expected_runtime_sec"]}
        t["pending"], t["answer"] = None, None
        return True

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
            devs = [{**{k: v for k, v in d.items() if k not in ("profile", "limits")}, "online": self.online(d),
                     "gpu_peak_gb": (d.get("limits") or {}).get("gpu_peak_gb", {}),
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
                                              "limits": device_limits(hw),
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
        except Exception as e:  # report and keep serving (the coordinator re-queues it elsewhere)
            cause = getattr(locals().get("runner"), "last_error", None)
            msg = f"{e}" + (f" ({cause})" if cause else "")
            report(status="failed", error=msg)
            self.log(f"task {tid} failed: {msg}")

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
