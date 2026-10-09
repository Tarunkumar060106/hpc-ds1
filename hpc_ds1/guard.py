"""Familiarity guard: knowing when the model doesn't know.

Calibrated confidence is only trustworthy on jobs like the ones the model was
trained on. On a kernel it has never seen (results/generalization.json) it can
be confidently wrong. The guard checks every state against the training data
before a decision may run automatically:

  machine   the hardware must match the machine the measurements came from;
            on another computer, measure it first (`hpc-ds1 setup`)
  workload  the kernel must be one the model was trained on
  size      the problem size must lie inside the measured range for that
            kernel (no extrapolation)
  job       the job (size, config, measured telemetry) must lie near a
            training state: nearest-neighbour distance in standardised
            feature space, compared to the distance that covers 99% of
            held-out familiar states

Any failed check makes the decision unfamiliar, and unfamiliar decisions are
always escalated, however confident the network is. The guard is fitted from
the same measurements as the model and stored in the checkpoint.
"""

import math

import numpy as np

from .features import CFG_FEATURES, TEL_FEATURES, _cfg, _hw, _tel

JOB_FEATURES = ["log_ops", "log_size"] + CFG_FEATURES + TEL_FEATURES
HW_TOLERANCE = 0.05   # relative difference in any hardware feature that means "another machine"
QUANTILE = 0.99       # held-out familiar states inside the radius
MIN_SCALE = 0.02      # floor on per-feature spread, so a constant feature can't blow up distances


def _job_vector(state):
    from .kernels import KERNELS
    wl, hw = state["workload"], state["hardware"]
    ops = KERNELS[wl["name"]].work(wl["size"]) if wl["name"] in KERNELS else 1.0
    return [math.log10(ops) / 12.0, math.log10(wl["size"]) / 7.0] + _cfg(state["config"], hw) + _tel(state["telemetry"], hw)


def job_matrix(states):
    return np.asarray([_job_vector(s) for s in states], dtype=np.float64)


def _nearest(x, ref):
    """Distance from every row of x to its nearest row of ref."""
    out = np.empty(len(x))
    for i in range(0, len(x), 256):
        d = ((x[i:i + 256, None, :] - ref[None, :, :]) ** 2).sum(-1)
        out[i:i + 256] = np.sqrt(d.min(1))
    return out


class FamiliarityGuard:
    def __init__(self, workloads, hardware, hardware_name, ref, scale, radius, size_range=None):
        self.workloads = sorted(workloads)
        self.size_range = {w: list(r) for w, r in (size_range or {}).items()}
        self.hardware = list(hardware)
        self.hardware_name = hardware_name
        self.ref = np.asarray(ref, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.radius = float(radius)

    @classmethod
    def fit(cls, train_states, calib_states, hardware):
        """train_states / calib_states: lists of model states (state dicts).
        The radius is set on calib_states, which must not overlap train_states."""
        ref = job_matrix(train_states)
        scale = np.maximum(ref.std(0), MIN_SCALE)
        d = _nearest(job_matrix(calib_states) / scale, ref / scale)
        radius = float(np.quantile(d, QUANTILE))
        sizes = {}
        for s in list(train_states) + list(calib_states):
            w, n = s["workload"]["name"], s["workload"]["size"]
            lo, hi = sizes.get(w, (n, n))
            sizes[w] = (min(lo, n), max(hi, n))
        return cls({s["workload"]["name"] for s in train_states}, _hw(hardware),
                   hardware.get("name"), ref, scale, radius, sizes)

    def check(self, states):
        """[{familiar, novelty, reasons}] for each state. novelty is the nearest-neighbour
        distance as a multiple of the familiar radius (<= 1 is familiar)."""
        x = job_matrix(states) / self.scale
        dist = _nearest(x, self.ref / self.scale)
        out = []
        for s, d in zip(states, dist):
            reasons = []
            hw = _hw(s["hardware"])
            if any(abs(a - b) > HW_TOLERANCE * max(abs(b), 1e-9) for a, b in zip(hw, self.hardware)):
                reasons.append(f"different machine from the one it was trained on ({self.hardware_name}); "
                               "run `hpc-ds1 setup` to measure this one")
            name = s["workload"]["name"]
            if name not in self.workloads:
                reasons.append(f"never trained on workload '{name}'")
            elif name in self.size_range:
                lo, hi = self.size_range[name]
                if not lo <= s["workload"]["size"] <= hi:
                    reasons.append(f"size {s['workload']['size']} is outside the measured range {lo}-{hi} for {name}")
            novelty = d / max(self.radius, 1e-9)
            if novelty > 1.0:
                reasons.append(f"job is unlike any measured training run ({novelty:.1f}x the familiar range)")
            out.append({"familiar": not reasons, "novelty": round(float(novelty), 3), "reasons": reasons})
        return out

    def to_dict(self):
        return {"workloads": self.workloads, "hardware": self.hardware, "hardware_name": self.hardware_name,
                "ref": self.ref.tolist(), "scale": self.scale.tolist(), "radius": self.radius,
                "size_range": self.size_range}

    @classmethod
    def from_dict(cls, d):
        return cls(d["workloads"], d["hardware"], d.get("hardware_name"), d["ref"], d["scale"], d["radius"],
                   d.get("size_range"))
