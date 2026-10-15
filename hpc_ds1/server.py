"""Local HTTP API + UI host. Standard library only.

    python -m hpc_ds1 serve            # http://127.0.0.1:8765

Endpoints
    GET  /                    facility console: playground, tuning sessions, approvals, audit, machine
    GET  /explorer            layer explorer (model internals)
    GET  /v1/health           model info
    GET  /v1/schema           actions, kernels and sizes, this machine's hardware
    GET  /v1/catalog          every layer: id, op, shape, parameter count
    POST /v1/decide           {workload, size, config, telemetry?, candidates?, threshold?, escalate_below?}
                              no telemetry -> the job is run once on this machine first
                              -> mode auto | confirm | escalate, plus familiarity reasons
    POST /v1/decide_batch     {requests: [{state, ...}]}  (states must carry telemetry)
    POST /v1/trace            same body as /v1/decide -> decision + every layer's activations
    POST /v1/tune             {workload, size, config, threshold?, max_steps?}
                              closed-loop tuning in one call; every step runs on this machine
    POST /v1/predict          {workload, size} -> System 1: full config from the job description, nothing run
    POST /v1/ask              {job or state, questions: {key: {type: noul|choice, question, options?}}}
                              -> free-text questions answered by the fine-tuned encoder (models/encoder)
    POST /v1/sessions         same body (+ start: "system1"|"given") -> background tuning session that
                              pauses for approval; "system1" starts from System 1's pick (dual process)
    GET  /v1/sessions[/<id>]  session list / one session with every run and decision
    POST /v1/sessions/<id>/approve   {approve: true|false, by?}
    GET  /v1/queue            moves waiting for an engineer, across all sessions
    GET  /v1/audit?limit=     every decision the service made (also ~/.hpc_ds1/audit.jsonl)
    GET  /v1/machine          hardware, what was measured, guard, model file
    GET  /v1/metrics          evaluation results shipped with the model

Fleet (hpc_ds1/fleet.py); agents and remote clients send the join token as X-HPC-Token
    GET  /v1/fleet                     devices, tasks, placement decisions
    POST /v1/fleet/register            agent joins {name, hardware, profile, model}
    POST /v1/fleet/poll                agent asks for work {device}
    POST /v1/fleet/tasks               submit {workload, size, config?} -> placed on a device
    POST /v1/fleet/tasks/<id>/status   agent reports runs, pending approvals, result
    POST /v1/fleet/tasks/<id>/approve  {approve, by?}
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .api import DEFAULT_ESCALATE_BELOW, DEFAULT_THRESHOLD, HPCDS1, home_dir
from .facility import AuditLog, Sessions
from .fleet import Fleet
from .kernels import KERNELS
from .space import ACTION_HELP, ACTIONS, apply_action, config_key, in_space, make_config, valid_actions
from .tuner import build_state

from .paths import RESULTS_DIR, UI_DIR as VISUALIZER_DIR, encoder_dir  # noqa: E402
STATIC = {"/": ("console.html", "text/html; charset=utf-8"),
          "/console": ("console.html", "text/html; charset=utf-8"),
          "/playground": ("console.html", "text/html; charset=utf-8"),
          "/explorer": ("index.html", "text/html; charset=utf-8"),
          "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/model_internals.json": ("model_internals.json", "application/json")}
MAX_BODY = 1 << 20


def _measured_tables():
    try:
        from .dataset import build_tables, load_measurements
        return build_tables(load_measurements())[0]
    except Exception:
        return {}


class App:
    def __init__(self, model: HPCDS1):
        self.model = model
        self.lock = threading.Lock()
        self.tables = _measured_tables()
        self._states = {}
        self.audit = AuditLog()
        self.sessions = Sessions(model, self.lock, self.audit)
        self.fleet = Fleet(audit=self.audit)
        self._answerer = None

    def answerer(self):
        if self._answerer is None:
            path = encoder_dir()
            if not (path / "scorer.pt").exists():
                raise FileNotFoundError("no question encoder: download hpc-ds1-encoder-*.zip from the GitHub release "
                                        "and unzip it into ~/.hpc_ds1 (or set HPC_DS1_ENCODER)")
            from .encoder import Answerer
            self._answerer = Answerer(path)
        return self._answerer

    def ask(self, body):
        qs = body.get("questions") or {}
        if not qs:
            raise ValueError("send at least one question: {key: {type, question, options?}}")
        with self.lock:
            state, source = self._state(body)
            out = [a.to_dict() for a in self.answerer().ask(state, qs, self.model.guard, self.model.system1)]
        return {"answers": out, "state": state, "telemetry_source": source}

    def _parse_job(self, body):
        w = body.get("workload")
        if w not in KERNELS:
            raise ValueError(f"unknown workload {w!r}; choose from {list(KERNELS)}")
        size = int(body.get("size", 0))
        lo, hi = KERNELS[w].sizes[0] // 4, KERNELS[w].sizes[-1]
        if not lo <= size <= hi:
            raise ValueError(f"size for {w} must be between {lo} and {hi} ({KERNELS[w].size_unit})")
        cfg = make_config(**{k: v for k, v in body.get("config", {}).items()
                             if k in ("ranks", "threads", "gpu", "precision")})
        if not in_space(cfg, bool(self.model.engine.hardware.get("gpu_model"))):
            raise ValueError(f"config {cfg} is outside the supported space")
        return w, size, cfg

    def _state(self, body):
        if "state" in body:
            return body["state"], "provided"
        w, size, cfg = self._parse_job(body)
        if body.get("telemetry"):
            return build_state(self.model.engine.hardware, w, size, cfg, body["telemetry"]), "provided"
        return self.model.observe(w, size, cfg), "measured now"

    def outcomes(self, state):
        """Measured ground truth for this exact job, if it was in the collection:
        next-run runtime of every move, best runtime reachable after it, and
        which moves are best (same definitions as training)."""
        inst = (state["workload"]["name"], state["workload"]["size"])
        table = self.tables.get(inst)
        if not table:
            return None
        if inst not in self._states:
            from .dataset import build_states
            self._states[inst] = {config_key(r["state"]["config"]): r
                                  for r in build_states({inst: table}, state["hardware"])}
        row = self._states[inst].get(config_key(make_config(**state["config"])))
        if row is None:
            return None
        reach = row["action_reach"]
        best = min(v for v in reach.values() if v is not None)
        return {"outcomes": row["outcomes"], "reachable": row["reachable"],
                "good_moves": [a for a in ACTIONS if reach.get(a) is not None and reach[a] <= best * 1.02],
                "label": row["best_action"]}

    def decide(self, body, trace=False):
        threshold = float(body.get("threshold", DEFAULT_THRESHOLD))
        below = float(body.get("escalate_below", DEFAULT_ESCALATE_BELOW))
        with self.lock:
            state, source = self._state(body)
            if trace:
                out = self.model.trace(state, body.get("candidates"), threshold, below)
                out["state"] = state
                truth = self.outcomes(state)
                if truth:
                    out.update(truth)
            else:
                d = self.model.decide(state, body.get("candidates"), threshold, below)
                self.audit.add(str(body.get("source", "api"))[:40], state, d)
                out = d.to_dict()
        out.setdefault("state", state)
        out["telemetry_source"] = source
        return out

    def tune(self, body):
        w, size, cfg = self._parse_job(body)
        with self.lock:
            r = self.model.tune(w, size, cfg, max_steps=min(int(body.get("max_steps", 8)), 12),
                                threshold=float(body.get("threshold", 0.0)))
        return r.to_dict()

    def start_session(self, body):
        w, size, cfg = self._parse_job(body)
        s = self.sessions.start(w, size, cfg, float(body.get("threshold", DEFAULT_THRESHOLD)),
                                float(body.get("escalate_below", DEFAULT_ESCALATE_BELOW)),
                                min(int(body.get("max_steps", 8)), 12), str(body.get("start", "given")))
        return s.to_dict()

    def predict(self, body):
        w = body.get("workload")
        if w not in KERNELS:
            raise ValueError(f"unknown workload {w!r}; choose from {list(KERNELS)}")
        size = int(body.get("size", 0))
        with self.lock:
            p = self.model.predict(w, size, threshold=float(body.get("threshold", DEFAULT_THRESHOLD)),
                                   escalate_below=float(body.get("escalate_below", DEFAULT_ESCALATE_BELOW)))
        return p.to_dict()

    def machine(self):
        from .dataset import data_dir
        coverage = {}
        for (w, size), table in sorted(self.tables.items()):
            c = coverage.setdefault(w, {"sizes": [], "correct_runs": 0})
            c["sizes"].append(size)
            c["correct_runs"] += len(table)
        report = home_dir() / "setup_report.json"
        g = self.model.guard
        hw_note = None
        if g:
            probe = {"hardware": self.model.engine.hardware, "workload": {"name": "_", "size": 1},
                     "config": {"ranks": 1, "threads": 1, "gpu": False, "precision": "fp32"},
                     "telemetry": {"runtime_sec": 1.0}}
            hw_note = [r for r in g.check([probe])[0]["reasons"] if r.startswith("different machine")]
        return {"hardware": self.model.engine.hardware, "coverage": coverage, "data_dir": str(data_dir()),
                "checkpoint": str(getattr(self.model, "path", "")), "home": str(home_dir()),
                "trained_on": self.model.meta.get("trained_on"), "trained_at": self.model.meta.get("trained_at"),
                "guard": ({"workloads": g.workloads, "size_range": g.size_range, "trained_on": g.hardware_name,
                           "reference_states": len(g.ref), "radius": round(g.radius, 4)} if g else None),
                "model_matches_this_machine": (not hw_note) if g else None,
                "setup_report": json.loads(report.read_text()) if report.exists() else None}

    def metrics(self):
        out = {}
        for name in ("metrics", "guard", "generalization", "system1"):
            f = RESULTS_DIR / f"{name}.json"
            if f.exists():
                out[name] = json.loads(f.read_text())
        m = out.get("metrics", {})
        return {"model": m.get("model"), "dataset": m.get("dataset"), "metrics": m.get("metrics"),
                "replay": m.get("replay"), "live": m.get("live"), "guard": out.get("guard"),
                "generalization": out.get("generalization"), "system1": out.get("system1")}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "hpc-ds1/1.0"

        def _send(self, code, payload, ctype="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(data)

        def _local(self):
            return self.client_address[0] in ("127.0.0.1", "::1")

        def _authorized(self, body):
            token = self.headers.get("X-HPC-Token") or body.get("token")
            return self._local() or token == app.fleet.token

        def do_OPTIONS(self):
            self._send(204, b"")

        def do_GET(self):
            path = urlparse(self.path).path
            if path in STATIC:
                name, ctype = STATIC[path]
                f = VISUALIZER_DIR / name
                return self._send(200, f.read_bytes(), ctype) if f.exists() else self._send(404, {"error": f"{name} missing"})
            if path == "/v1/health":
                return self._send(200, {"status": "ok", **app.model.info()})
            if path == "/v1/schema":
                return self._send(200, {
                    "actions": ACTIONS, "action_help": ACTION_HELP, "hardware": app.model.engine.hardware,
                    "workloads": {w: {"display_name": k.display_name, "character": k.character, "sizes": k.sizes,
                                      "size_unit": k.size_unit} for w, k in KERNELS.items()},
                    "example_request": {"workload": "gemm", "size": 2048,
                                        "config": {"ranks": 2, "threads": 4, "gpu": False, "precision": "fp32"}}})
            if path == "/v1/catalog":
                return self._send(200, app.model.layer_catalog())
            if path == "/v1/sessions":
                return self._send(200, app.sessions.list())
            if path.startswith("/v1/sessions/"):
                s = app.sessions.items.get(path.split("/")[3])
                return self._send(200, s.to_dict()) if s else self._send(404, {"error": "no such session"})
            if path == "/v1/queue":
                return self._send(200, app.sessions.queue() + app.fleet.queue())
            if path == "/v1/fleet":
                snap = app.fleet.snapshot()
                if self._local():
                    snap["token"] = app.fleet.token
                return self._send(200, snap)
            if path == "/v1/audit":
                q = parse_qs(urlparse(self.path).query)
                return self._send(200, app.audit.recent(min(int(q.get("limit", ["200"])[0]), 2000)))
            if path == "/v1/machine":
                return self._send(200, app.machine())
            if path == "/v1/metrics":
                return self._send(200, app.metrics())
            self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                return self._send(413, {"error": "request body too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if path == "/v1/decide":
                    return self._send(200, app.decide(body))
                if path == "/v1/trace":
                    return self._send(200, app.decide(body, trace=True))
                if path == "/v1/decide_batch":
                    reqs = body.get("requests", [])
                    if any("state" not in r and not r.get("telemetry") for r in reqs):
                        raise ValueError("decide_batch needs a state or telemetry on every request")
                    return self._send(200, [app.decide(r) for r in reqs])
                if path == "/v1/tune":
                    return self._send(200, app.tune(body))
                if path.startswith("/v1/fleet/"):
                    if not self._authorized(body):
                        return self._send(401, {"error": "missing or wrong fleet token (X-HPC-Token)"})
                    parts = path.split("/")
                    if path == "/v1/fleet/register":
                        return self._send(200, app.fleet.register(str(body.get("name", "device"))[:60],
                                                                  body.get("hardware", {}), body.get("profile", {}),
                                                                  body.get("model", {})))
                    if path == "/v1/fleet/poll":
                        return self._send(200, {"task": app.fleet.poll(body.get("device"))})
                    if path == "/v1/fleet/tasks":
                        return self._send(200, app.fleet.submit(
                            body.get("workload"), int(body.get("size", 0)), body.get("config"),
                            float(body.get("threshold", DEFAULT_THRESHOLD)),
                            float(body.get("escalate_below", DEFAULT_ESCALATE_BELOW)),
                            min(int(body.get("max_steps", 8)), 12), body.get("start")))
                    if len(parts) == 6 and parts[5] == "status":
                        return self._send(200, app.fleet.update(parts[4], body.get("device"), body))
                    if len(parts) == 6 and parts[5] == "approve":
                        return self._send(200, app.fleet.approve(parts[4], body.get("approve", False), body.get("by")))
                    return self._send(404, {"error": "not found"})
                if path == "/v1/sessions":
                    return self._send(200, app.start_session(body))
                if path == "/v1/predict":
                    return self._send(200, app.predict(body))
                if path == "/v1/ask":
                    return self._send(200, app.ask(body))
                if path.startswith("/v1/sessions/") and path.endswith("/approve"):
                    s = app.sessions.answer(path.split("/")[3], body.get("approve", False), body.get("by"))
                    return self._send(200, s.to_dict())
                return self._send(404, {"error": "not found"})
            except (ValueError, KeyError, TypeError, FileNotFoundError) as e:
                return self._send(400, {"error": str(e)})
            except RuntimeError as e:
                return self._send(500, {"error": f"run failed: {e}"})

        def log_message(self, fmt, *args):
            print(f"[hpc-ds1] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def serve(host="127.0.0.1", port=8765, checkpoint=None, device=None):
    model = HPCDS1.load(checkpoint, device)
    app = App(model)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    info = model.info()
    print(f"HPC-DS1 ({info['parameters']:,} params, {info['device']}) on http://{host}:{port}", flush=True)
    print(f"  console: http://{host}:{port}/   API: POST http://{host}:{port}/v1/decide", flush=True)
    print(f"  fleet join token: {app.fleet.token}", flush=True)
    print(f"  connect a device: hpc-ds1 agent --server http://<this-machine-ip>:{port} --token {app.fleet.token}"
          + ("" if host not in ("127.0.0.1", "localhost") else "   (serve with --host 0.0.0.0 to accept other machines)"),
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        model.close()
