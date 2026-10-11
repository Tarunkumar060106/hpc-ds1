"""Jev-style decision API for HPC-DS1.

    import hpc_ds1
    model = hpc_ds1.load()
    state = model.observe("gemm", 2048, {"ranks": 2, "threads": 4})   # runs it once, for real
    d = model.decide(state)                                           # all valid actions + STOP_TUNING
    d = model.decide(state, ["USE_GPU", "ADD_THREADS", "STOP_TUNING"])
    if d.mode == "auto":
        apply(d.choice)
    elif d.mode == "confirm":
        ask_to_confirm(d)          # likely right: a one-click yes/no
    else:
        escalate(d, d.reasons)     # unsure, or a job unlike anything it was trained on

A decision is a function call: options in, (choice, confidence,
probabilities) out, deterministic for the same input. Confidence is the
temperature-calibrated probability of the chosen option.

Three tiers, like Jev:
  auto      confidence >= threshold (0.85) and the job is familiar
  confirm   escalate_below (0.5) <= confidence < threshold
  escalate  confidence < escalate_below, or the familiarity guard (guard.py)
            says the job or machine is unlike the training data
should_escalate is True for both confirm and escalate: a human is involved.
"""

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .features import FEATURE_VERSION, STOP_INDEX, featurize
from .guard import FamiliarityGuard
from .net import HPCDS1Net, summarize_trace
from .space import ACTIONS
from .tuner import LiveRunner, build_state, run_tuning

DEFAULT_THRESHOLD = 0.85
DEFAULT_ESCALATE_BELOW = 0.5
MODES = ("auto", "confirm", "escalate")


def home_dir():
    """Where `hpc-ds1 setup` keeps this machine's measurements and model."""
    return Path(os.environ.get("HPC_DS1_HOME", Path.home() / ".hpc_ds1"))


def default_checkpoint():
    """$HPC_DS1_CHECKPOINT, else this machine's own model from `hpc-ds1 setup`,
    else the model shipped with the source tree."""
    if os.environ.get("HPC_DS1_CHECKPOINT"):
        return Path(os.environ["HPC_DS1_CHECKPOINT"])
    own = home_dir() / "model" / "hpc_ds1.pt"
    if own.exists():
        return own
    return Path(__file__).resolve().parent.parent / "model" / "hpc_ds1.pt"


def decision_mode(confidence, familiar, threshold, escalate_below):
    if not familiar or confidence < escalate_below:
        return "escalate"
    return "auto" if confidence >= threshold else "confirm"


@dataclass
class Decision:
    choice: str
    confidence: float
    probabilities: Dict[str, float]
    should_escalate: bool
    threshold: float
    alternatives: List[Dict[str, float]]
    stop_probability: float
    expected_improvement: float
    valid_actions: List[str]
    ignored_candidates: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    mode: str = "auto"
    escalate_below: float = DEFAULT_ESCALATE_BELOW
    familiar: bool = True
    novelty: Optional[float] = None
    reasons: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class HPCDS1:
    def __init__(self, net: HPCDS1Net, device, meta: Optional[dict] = None,
                 guard: Optional[FamiliarityGuard] = None):
        self.net = net.to(device).eval()
        self.device = device
        self.meta = meta or {}
        self.guard = guard
        self._engine = None
        self.last_decision: Optional[Decision] = None
        self.system1 = None  # System1, loaded from system1.pt next to the checkpoint if present

    @classmethod
    def load(cls, path=None, device=None):
        path = Path(path or default_checkpoint())
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `hpc-ds1 setup` to measure this machine and train on it")
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if ckpt.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"checkpoint feature_version {ckpt.get('feature_version')} != {FEATURE_VERSION}; retrain")
        if ckpt.get("ensemble"):
            from .net import EnsembleNet
            net = EnsembleNet([HPCDS1Net(**ckpt["hparams"]) for _ in range(ckpt["ensemble"])])
        else:
            net = HPCDS1Net(**ckpt["hparams"])
        net.load_state_dict(ckpt["state_dict"])
        guard = FamiliarityGuard.from_dict(ckpt["guard"]) if ckpt.get("guard") else None
        model = cls(net, device, ckpt.get("meta"), guard)
        model.path = path
        own = home_dir() / "model" / "system1.pt"  # written by `hpc-ds1 learn`
        s1 = own if own.exists() else path.parent / "system1.pt"
        if s1.exists():
            from .system1 import System1
            model.system1 = System1.load(s1, device="cpu")
            model.system1_path = s1
        return model

    # ------------------------------------------------------------------ engine

    @property
    def engine(self):
        if self._engine is None:
            from .runner import Engine
            self._engine = Engine()
        return self._engine

    def close(self):
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def observe(self, workload, size, config):
        """Run the job once on this machine and return the state to decide on."""
        runner = LiveRunner(workload, size, self.engine)
        tel = runner(config)
        if tel is None:
            raise ValueError(f"{workload} {size} with {config} failed or gave a wrong result on this machine")
        return build_state(self.engine.hardware, workload, size, config, tel)

    # --------------------------------------------------------------- decisions

    def _allowed(self, mask, candidates):
        allowed = mask.clone()
        ignored = [[] for _ in range(allowed.shape[0])]
        if candidates is not None:
            unknown = [c for c in candidates if c not in ACTIONS]
            if unknown:
                raise ValueError(f"unknown candidate(s) {unknown}; choose from {ACTIONS}")
            wanted = torch.tensor([a in candidates for a in ACTIONS], device=allowed.device)
            for i in range(allowed.shape[0]):
                ignored[i] = [a for j, a in enumerate(ACTIONS) if wanted[j] and not allowed[i, j]]
            allowed = allowed & wanted
        if not allowed.any(dim=1).all():
            raise ValueError("no candidate is valid for this state")
        return allowed, ignored

    @torch.no_grad()
    def _run(self, states, candidates, threshold, trace=False, escalate_below=DEFAULT_ESCALATE_BELOW):
        t0 = time.perf_counter()
        feats = featurize(states, device=self.device, workloads=self.net.workloads)
        valid = feats["mask"].clone()
        feats["mask"], ignored = self._allowed(feats["mask"], candidates)
        out = self.net(feats, trace=trace)
        checks = (self.guard.check(states) if self.guard
                  else [{"familiar": True, "novelty": None, "reasons": []} for _ in states])
        latency = (time.perf_counter() - t0) * 1000 / len(states)
        probs = out["probs"].cpu()
        stop_p = torch.sigmoid(out["stop_logit"]).cpu()
        imp = out["improvement"].clamp(0, 1).cpu()
        decisions = []
        for i in range(len(states)):
            p = {a: round(probs[i, j].item(), 5) for j, a in enumerate(ACTIONS) if feats["mask"][i, j]}
            ranked = sorted(p.items(), key=lambda kv: -kv[1])
            choice, conf = ranked[0]
            mode = decision_mode(conf, checks[i]["familiar"], threshold, escalate_below)
            decisions.append(Decision(
                choice=choice, confidence=round(conf, 5), probabilities=p,
                should_escalate=mode != "auto", threshold=threshold,
                alternatives=[{"action": a, "probability": v} for a, v in ranked[1:4]],
                stop_probability=round(stop_p[i].item(), 5), expected_improvement=round(imp[i].item(), 4),
                valid_actions=[a for j, a in enumerate(ACTIONS) if valid[i, j] or j == STOP_INDEX],
                ignored_candidates=ignored[i], latency_ms=round(latency, 3),
                mode=mode, escalate_below=escalate_below, **checks[i]))
        return decisions, out

    def decide(self, state: dict, candidates: Optional[Sequence[str]] = None,
               threshold: float = DEFAULT_THRESHOLD, escalate_below: float = DEFAULT_ESCALATE_BELOW) -> Decision:
        return self._run([state], candidates, threshold, escalate_below=escalate_below)[0][0]

    def decide_batch(self, states: List[dict], candidates: Optional[Sequence[str]] = None,
                     threshold: float = DEFAULT_THRESHOLD,
                     escalate_below: float = DEFAULT_ESCALATE_BELOW) -> List[Decision]:
        return self._run(states, candidates, threshold, escalate_below=escalate_below)[0]

    def trace(self, state: dict, candidates: Optional[Sequence[str]] = None,
              threshold: float = DEFAULT_THRESHOLD, escalate_below: float = DEFAULT_ESCALATE_BELOW) -> dict:
        """Decision plus every layer's activations (powers the layer explorer)."""
        decisions, out = self._run([state], candidates, threshold, trace=True, escalate_below=escalate_below)
        return {"decision": decisions[0].to_dict(), "layers": summarize_trace(out, 0)}

    def policy(self, threshold=DEFAULT_THRESHOLD, escalate_below=DEFAULT_ESCALATE_BELOW):
        """Policy for run_tuning. threshold <= 0 is fully autonomous: nothing is
        escalated, the familiarity guard included (evaluation replays use this)."""
        def fn(state, exclude=()):
            cands = [a for a in ACTIONS if a not in exclude] if exclude else None
            d = self.decide(state, cands, threshold=threshold, escalate_below=escalate_below)
            self.last_decision = d
            return d.choice, d.confidence, 0, threshold > 0 and d.should_escalate
        return fn

    def tune(self, workload, size, initial_config, runner=None, max_steps=8,
             threshold=DEFAULT_THRESHOLD, on_escalate=None, escalate_below=DEFAULT_ESCALATE_BELOW):
        """Closed-loop tuning, running each suggested config for real by default.
        Decisions that aren't "auto" go to on_escalate(action, confidence, state);
        self.last_decision holds the full decision (mode, reasons) for the prompt."""
        runner = runner or LiveRunner(workload, size, self.engine)
        return run_tuning(self.policy(threshold, escalate_below), workload, size, initial_config, runner,
                          max_steps=max_steps, threshold=threshold, on_escalate=on_escalate)

    # ------------------------------------------------------- System 1 / dual process

    def predict(self, workload, size, hardware=None, threshold=DEFAULT_THRESHOLD,
                escalate_below=DEFAULT_ESCALATE_BELOW):
        """System 1: the full configuration from the job description alone. Nothing is run."""
        if self.system1 is None:
            raise FileNotFoundError("no System 1 model (model/system1.pt); train it with scripts/train_system1.py")
        return self.system1.predict(hardware or self.engine.hardware, workload, int(size), threshold, escalate_below)

    def plan(self, workload, size, threshold=DEFAULT_THRESHOLD, escalate_below=DEFAULT_ESCALATE_BELOW,
             on_escalate=None, max_steps=8, runner=None):
        """Dual process. System 1 picks a configuration instantly. If it is confident and
        the job is familiar, that configuration is run once and kept. Otherwise System 2
        (the measured tuning loop) starts from System 1's pick; unfamiliar jobs are escalated
        through on_escalate before anything runs. Returns (prediction, TuneResult, used_system2)."""
        from .system1 import GATE_SLACK
        pred = self.predict(workload, size, threshold=threshold, escalate_below=escalate_below)
        runner = runner or LiveRunner(workload, size, self.engine)
        if not pred.familiar:
            if on_escalate is None or not on_escalate("RUN_SYSTEM1_PICK", pred.confidence, {"prediction": pred.to_dict()}):
                return pred, None, False
        if pred.mode == "auto":
            result = run_tuning(lambda state, exclude=(): ("STOP_TUNING", 1.0, 0), workload, size, pred.config, runner,
                                max_steps=1)
            if result.final_runtime <= pred.expected_runtime_sec * GATE_SLACK:  # ran as predicted
                return pred, result, False
        result = run_tuning(self.policy(threshold, escalate_below), workload, size, pred.config, runner,
                            max_steps=max_steps, threshold=threshold, on_escalate=on_escalate)
        return pred, result, True

    def info(self):
        return {"model": "HPC-DS1", "parameters": self.net.count_parameters(), "hparams": self.net.hparams,
                "temperature": round(self.net.temperature.item(), 4), "actions": ACTIONS,
                "system1": bool(self.system1),
                "device": str(self.device), "checkpoint": str(getattr(self, "path", "")), "escalate_below": DEFAULT_ESCALATE_BELOW,
                "familiarity_guard": ({"trained_workloads": self.guard.workloads, "trained_on": self.guard.hardware_name,
                                       "radius": round(self.guard.radius, 4)} if self.guard else None),
                **{k: v for k, v in self.meta.items() if k in ("test_metrics", "trained_at", "trained_on")}}

    def layer_catalog(self):
        return self.net.layer_catalog()


def load(path=None, device=None) -> HPCDS1:
    return HPCDS1.load(path, device)
