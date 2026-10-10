"""System 1: the whole configuration in one glance, before the job ever runs.

The step-by-step model (net.py, tuner.py) is System 2: it needs one real run
to see the job, then improves it a move at a time, measuring each move.
System 1 answers from the job description alone:

    predict("mltrain", 4096)  ->  GPU, 1 thread, bf16   (no run needed)

How. A small performance model learns, from every real measured run, the
runtime of a (job, configuration) pair and whether that configuration fails.
To answer, it scores every configuration in the space in one batched forward
pass and picks the fastest one predicted to work. Five independently trained
networks form an ensemble; how much they agree is turned into a calibrated
confidence that the pick is within 5% of the best configuration.

Dual process (plan()):
    familiar and confident  -> run System 1's pick, done (one run)
    otherwise               -> System 2 tunes, starting from System 1's pick
    unfamiliar              -> escalate (familiarity guard)
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import _cfg, _hw, _wl
from .kernels import WORKLOAD_NAMES
from .space import all_configs, config_key

GOOD_WITHIN = 1.05  # a pick is "right" if its measured runtime is within 5% of the best
GATE_SLACK = 1.5   # a confident pick is kept only if its first real run is at most 1.5x the predicted runtime
SEEDS = (0, 1, 2, 3, 4)


def _largest_prime_factor(n):
    n, f, best = int(n), 2, 1
    while f * f <= n:
        while n % f == 0:
            best, n = f, n // f
        f += 1
    return max(best, n) if n > 1 else best


def _job_vec(hardware, workload, size, workloads):
    # Largest prime factor of the problem size: FFT-style libraries (cuFFT, MKL) have fast
    # radix-2/3/5 paths, and some precisions only work for powers of two on some hardware.
    lpf = _largest_prime_factor(size)
    return _hw(hardware) + _wl({"name": workload, "size": size}, workloads) + [
        1.0 if lpf == 2 else 0.0, math.log2(lpf) / 10.0]


def _cfg_vec(config, hardware):
    return _cfg(config, hardware)


class PerfNet(nn.Module):
    """(job, config) -> predicted log10 runtime, and logit that the config fails."""

    def __init__(self, in_dim, width=256, depth=3, dropout=0.1):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU(), nn.Dropout(dropout)]
            d = width
        self.body = nn.Sequential(*layers)
        self.runtime = nn.Linear(width, 1)
        self.fails = nn.Linear(width, 1)

    def forward(self, x):
        h = self.body(x)
        return self.runtime(h).squeeze(-1), self.fails(h).squeeze(-1)


# ---------------------------------------------------------------------------
# data: one row per real measured run
# ---------------------------------------------------------------------------

def rows_from_measurements(rows, hardware, workloads=None, keep=None):
    """X, log10 runtime (nan for failures), failed flag. keep(workload, size) filters problems."""
    workloads = list(workloads or WORKLOAD_NAMES)
    xs, ys, fs = [], [], []
    for r in rows:
        if r["workload"] not in workloads or (keep and not keep(r["workload"], r["size"])):
            continue
        ok = r["ok"] and r.get("correct", False)
        xs.append(_job_vec(hardware, r["workload"], r["size"], workloads) + _cfg_vec(r["config"], hardware))
        ys.append(math.log10(max(r["telemetry"]["runtime_sec"], 1e-6)) if ok else float("nan"))
        fs.append(0.0 if ok else 1.0)
    return (torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32),
            torch.tensor(fs, dtype=torch.float32))


def train_member(x, y, failed, seed, device, epochs=600, lr=2e-3, wd=1e-3):
    torch.manual_seed(seed)
    net = PerfNet(x.shape[1]).to(device)
    x, y, failed = x.to(device), y.to(device), failed.to(device)
    ok = ~torch.isnan(y)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 0.05)
    for _ in range(epochs):
        net.train()
        pred, fail_logit = net(x)
        loss = F.smooth_l1_loss(pred[ok], y[ok], beta=0.05) + 0.5 * F.binary_cross_entropy_with_logits(fail_logit, failed)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


def train_ensemble(x, y, failed, device, seeds=SEEDS):
    return [train_member(x, y, failed, s, device) for s in seeds]


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_configs(members, hardware, workload, size, workloads, device):
    """For every config: ensemble predictions [members, configs] of log10 runtime and fail probability."""
    configs = all_configs(bool(hardware.get("gpu_model")))
    job = _job_vec(hardware, workload, size, workloads)
    x = torch.tensor([job + _cfg_vec(c, hardware) for c in configs], dtype=torch.float32, device=device)
    rt, fl = zip(*(m(x) for m in members))
    return configs, torch.stack(rt).cpu(), torch.sigmoid(torch.stack(fl)).cpu()


def pick(configs, log_rt, p_fail):
    """Ensemble pick + the raw signals confidence is built from."""
    mean_rt, mean_fail = log_rt.mean(0), p_fail.mean(0)
    usable = p_fail.max(0).values < 0.5  # cautious: avoid a config if any member expects it to fail
    if not usable.any():
        usable = torch.ones_like(usable)
    score = torch.where(usable, mean_rt, torch.full_like(mean_rt, float("inf")))
    best = int(score.argmin())
    # agreement: share of members whose own pick is predicted (by the ensemble) within 5% of ours
    tol = math.log10(GOOD_WITHIN)
    member_picks = torch.where(p_fail < 0.5, log_rt, torch.full_like(log_rt, float("inf"))).argmin(1)
    agree = float((mean_rt[member_picks] <= mean_rt[best] + tol).float().mean())
    order = score.argsort()
    runner_up = next((int(i) for i in order[1:] if config_key(configs[int(i)]) != config_key(configs[best])), best)
    margin = float(score[runner_up] - score[best]) if math.isfinite(float(score[runner_up])) else 1.0
    spread = float(log_rt[:, best].std())
    return best, {"agreement": agree, "margin": margin, "spread": spread,
                  "fail_prob": float(mean_fail[best]), "pred_log_runtime": float(mean_rt[best])}


def confidence_features(sig):
    return [sig["agreement"], min(sig["margin"], 0.5) * 4, min(sig["spread"], 0.5) * 4, sig["fail_prob"]]


class Calibrator:
    """Logistic map from ensemble signals to P(pick within 5% of best), fitted on out-of-fold picks."""

    def __init__(self, w=None, b=0.0):
        self.w = list(w) if w is not None else [3.0, 1.0, -1.0, -2.0]
        self.b = b

    def fit(self, feats, labels, l2=0.3, steps=2000):
        x, y = torch.tensor(feats, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=steps)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(x @ w + b, y) + l2 * (w ** 2).sum() / len(y)
            loss.backward()
            return loss
        opt.step(closure)
        self.w, self.b = w.detach().tolist(), float(b.detach())
        return self

    def __call__(self, feats):
        z = sum(a * c for a, c in zip(self.w, feats)) + self.b
        return 1.0 / (1.0 + math.exp(-z))

    def to_dict(self):
        return {"w": self.w, "b": self.b}


@dataclass
class Prediction:
    config: dict
    confidence: float
    expected_runtime_sec: float
    fail_probability: float
    mode: str                      # auto | confirm | escalate
    familiar: bool
    reasons: List[str] = field(default_factory=list)
    alternatives: List[Dict] = field(default_factory=list)
    signals: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


class System1:
    def __init__(self, members, workloads, calibrator, size_range, hardware_ref, hardware_name, device="cpu"):
        self.members = [m.to(device).eval() for m in members]
        self.workloads = list(workloads)
        self.calibrator = calibrator
        self.size_range = {w: list(r) for w, r in size_range.items()}
        self.hardware_ref = list(hardware_ref)
        self.hardware_name = hardware_name
        self.device = device

    def familiarity(self, hardware, workload, size):
        reasons = []
        hw = _hw(hardware)
        if any(abs(a - b) > 0.05 * max(abs(b), 1e-9) for a, b in zip(hw, self.hardware_ref)):
            reasons.append(f"different machine from the one System 1 learned ({self.hardware_name}); "
                           "run `hpc-ds1 setup` to measure this one")
        if workload not in self.workloads:
            reasons.append(f"never trained on workload '{workload}'")
        elif workload in self.size_range:
            lo, hi = self.size_range[workload]
            if not lo <= size <= hi:
                reasons.append(f"size {size} is outside the measured range {lo}-{hi} for {workload}")
        return reasons

    def predict(self, hardware, workload, size, threshold=0.85, escalate_below=0.5):
        import time
        t0 = time.perf_counter()
        configs, log_rt, p_fail = score_configs(self.members, hardware, workload, size, self.workloads, self.device)
        best, sig = pick(configs, log_rt, p_fail)
        conf = self.calibrator(confidence_features(sig))
        reasons = self.familiarity(hardware, workload, size)
        mode = "escalate" if reasons or conf < escalate_below else ("auto" if conf >= threshold else "confirm")
        mean = log_rt.mean(0)
        order = [int(i) for i in mean.argsort() if p_fail.mean(0)[int(i)] < 0.5][:4]
        alts = [{"config": configs[i], "expected_runtime_sec": round(10 ** float(mean[i]), 6)}
                for i in order if i != best][:3]
        return Prediction(config=configs[best], confidence=round(conf, 4),
                          expected_runtime_sec=round(10 ** sig["pred_log_runtime"], 6),
                          fail_probability=round(sig["fail_prob"], 4), mode=mode, familiar=not reasons,
                          reasons=reasons, alternatives=alts,
                          signals={k: round(v, 4) for k, v in sig.items()},
                          latency_ms=round((time.perf_counter() - t0) * 1000, 3))

    # -------- persistence
    def to_checkpoint(self):
        return {"kind": "system1", "version": 1, "workloads": self.workloads, "in_dim": self.members[0].body[0].in_features,
                "members": [m.state_dict() for m in self.members], "calibrator": self.calibrator.to_dict(),
                "size_range": self.size_range, "hardware_ref": self.hardware_ref, "hardware_name": self.hardware_name}

    @classmethod
    def from_checkpoint(cls, ck, device="cpu"):
        members = []
        for sd in ck["members"]:
            m = PerfNet(ck["in_dim"])
            m.load_state_dict(sd)
            members.append(m)
        return cls(members, ck["workloads"], Calibrator(**ck["calibrator"]), ck["size_range"],
                   ck["hardware_ref"], ck["hardware_name"], device)

    @classmethod
    def load(cls, path, device="cpu"):
        return cls.from_checkpoint(torch.load(path, map_location=device, weights_only=False), device)


def size_ranges(rows):
    out = {}
    for r in rows:
        lo, hi = out.get(r["workload"], (r["size"], r["size"]))
        out[r["workload"]] = (min(lo, r["size"]), max(hi, r["size"]))
    return out


def dual_start(model, workload, size, threshold, escalate_below, fallback_config):
    """Where a tuning session should start, and whether it may stop right away.

    Returns (start_config, prediction or None, gate). With a familiar job, the
    session starts from System 1's pick. If System 1 was confident ("auto"),
    gate(state) says whether the first real run went as predicted (within GATE_SLACK x of
    the expected runtime); if so the session keeps that configuration without
    asking System 2 anything. Unfamiliar jobs start from fallback_config and are
    handled by System 2 and its own guard."""
    if getattr(model, "system1", None) is None:
        return fallback_config, None, None
    pred = model.predict(workload, size, threshold=threshold, escalate_below=escalate_below)
    if not pred.familiar:
        return fallback_config, pred, None
    gate = None
    if pred.mode == "auto":
        limit = pred.expected_runtime_sec * GATE_SLACK

        def gate(state):
            return state["telemetry"]["runtime_sec"] <= limit
    return pred.config, pred, gate


def gated(policy, gate):
    """Wrap a System 2 policy so the first decision is STOP when System 1's pick held up."""
    first = [True]

    def fn(state, exclude=()):
        if first[0]:
            first[0] = False
            if gate is not None and gate(state):
                return "STOP_TUNING", 1.0, 0, False
        return policy(state, exclude)
    return fn
