"""What a shared computing facility needs around the model.

  AuditLog   every decision the service makes (job, choice, confidence, mode,
             reasons, who approved it), in memory and appended to
             ~/.hpc_ds1/audit.jsonl, so automated changes can be reviewed later
  Sessions   closed-loop tuning sessions that run in the background and pause
             whenever a decision isn't "auto": the pending move waits in an
             approval queue until an engineer approves or rejects it

The model and the rank engine are shared, so every decision and every real run
takes the server's lock; waiting for approval does not.
"""

import itertools
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone

from .api import DEFAULT_ESCALATE_BELOW, DEFAULT_THRESHOLD, home_dir
from .space import ACTIONS
from .tuner import LiveRunner, run_tuning

APPROVAL_TIMEOUT_SEC = 3600


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AuditLog:
    def __init__(self, path=None, keep=2000):
        self.path = path or home_dir() / "audit.jsonl"
        self.rows = deque(maxlen=keep)
        self.lock = threading.Lock()
        self._ids = itertools.count(1)
        try:
            with open(self.path) as f:
                for line in f:
                    self.rows.append(json.loads(line))
            if self.rows:
                self._ids = itertools.count(int(self.rows[-1].get("id", 0)) + 1)
        except (OSError, ValueError):
            pass

    def add(self, source, state, decision, **extra):
        """decision: a Decision, or its to_dict() (as reported by a fleet agent)."""
        d = decision if isinstance(decision, dict) else decision.to_dict()
        row = {"id": next(self._ids), "at": _now(), "source": source,
               "workload": state["workload"]["name"], "size": state["workload"]["size"],
               "config": state["config"], "runtime_sec": state["telemetry"].get("runtime_sec"),
               "choice": d["choice"], "confidence": d["confidence"], "mode": d["mode"],
               "familiar": d.get("familiar", True), "reasons": d.get("reasons", []), **extra}
        with self.lock:
            self.rows.append(row)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps(row) + "\n")
            except OSError:
                pass
        return row

    def resolve(self, row_id, **fields):
        with self.lock:
            for r in reversed(self.rows):
                if r["id"] == row_id:
                    r.update(fields)
                    break

    def recent(self, limit=200):
        with self.lock:
            return list(self.rows)[-limit:][::-1]


class Session:
    def __init__(self, sid, workload, size, config, threshold, escalate_below, max_steps):
        self.id, self.workload, self.size, self.config = sid, workload, size, config
        self.threshold, self.escalate_below, self.max_steps = threshold, escalate_below, max_steps
        self.status = "running"          # running | awaiting_approval | done | failed
        self.started_at, self.finished_at = _now(), None
        self.runs = []                   # every real run: config, runtime or failure
        self.decisions = []              # every model decision in this session
        self.pending = None              # the decision waiting for a human
        self.result, self.error = None, None
        self._answer = None
        self._event = threading.Event()

    def to_dict(self):
        return {"id": self.id, "workload": self.workload, "size": self.size, "initial_config": self.config,
                "threshold": self.threshold, "escalate_below": self.escalate_below, "status": self.status,
                "started_at": self.started_at, "finished_at": self.finished_at, "runs": self.runs,
                "decisions": self.decisions, "pending": self.pending, "result": self.result, "error": self.error}


class Sessions:
    def __init__(self, model, lock, audit):
        self.model, self.lock, self.audit = model, lock, audit
        self.items = {}
        self._ids = itertools.count(1)

    def start(self, workload, size, config, threshold=DEFAULT_THRESHOLD,
              escalate_below=DEFAULT_ESCALATE_BELOW, max_steps=8):
        s = Session(f"s{next(self._ids)}", workload, size, config, threshold, escalate_below, max_steps)
        self.items[s.id] = s
        threading.Thread(target=self._run, args=(s,), daemon=True).start()
        return s

    def _run(self, s):
        model, lock = self.model, self.lock
        live = LiveRunner(s.workload, s.size, model.engine)

        def runner(config):
            with lock:
                tel = live(config)
            s.runs.append({"at": _now(), "config": dict(config),
                           "runtime_sec": tel["runtime_sec"] if tel else None,
                           "ok": tel is not None})
            return tel

        runner.hardware = live.hardware
        last = {}

        def policy(state, exclude=()):
            cands = [a for a in ACTIONS if a not in exclude] if exclude else None
            with lock:
                d = model.decide(state, cands, s.threshold, s.escalate_below)
            row = self.audit.add("session " + s.id, state, d)
            last.update(decision=d, row=row["id"])
            s.decisions.append({"at": row["at"], "config": state["config"], "choice": d.choice,
                                "confidence": d.confidence, "mode": d.mode, "reasons": d.reasons,
                                "audit_id": row["id"]})
            return d.choice, d.confidence, 0, s.threshold > 0 and d.should_escalate

        def on_escalate(action, confidence, state):
            d = last["decision"]
            s.pending = {"action": action, "confidence": confidence, "mode": d.mode, "reasons": d.reasons,
                         "config": state["config"], "runtime_sec": state["telemetry"]["runtime_sec"],
                         "alternatives": d.alternatives, "since": _now(), "audit_id": last["row"]}
            s.status = "awaiting_approval"
            s._event.clear()
            answered = s._event.wait(APPROVAL_TIMEOUT_SEC)
            ok, by = (s._answer if answered else (False, "timeout"))
            self.audit.resolve(last["row"], approved=ok, approved_by=by)
            s.decisions[-1].update(approved=ok, approved_by=by)
            s.pending, s.status = None, "running"
            return ok

        try:
            r = run_tuning(policy, s.workload, s.size, s.config, runner, max_steps=s.max_steps,
                           threshold=s.threshold, on_escalate=on_escalate)
            s.result = r.to_dict()
            s.status = "done"
        except Exception as e:  # a failing job must not take the server down
            s.error, s.status = str(e), "failed"
        s.finished_at = _now()

    def answer(self, sid, approve, by="engineer"):
        s = self.items.get(sid)
        if s is None:
            raise KeyError(f"no session {sid}")
        if s.status != "awaiting_approval":
            raise ValueError(f"session {sid} is not waiting for approval ({s.status})")
        s._answer = (bool(approve), by or "engineer")
        s._event.set()
        deadline = time.time() + 2
        while s.status == "awaiting_approval" and time.time() < deadline:
            time.sleep(0.02)
        return s

    def queue(self):
        return [{"session": s.id, "workload": s.workload, "size": s.size, **s.pending}
                for s in self.items.values() if s.status == "awaiting_approval" and s.pending]

    def list(self):
        return [{k: v for k, v in s.to_dict().items() if k not in ("runs", "decisions")}
                for s in sorted(self.items.values(), key=lambda s: s.started_at, reverse=True)]
