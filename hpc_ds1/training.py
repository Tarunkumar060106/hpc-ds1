"""Training core shared by scripts/train.py, scripts/generalization.py and
`hpc-ds1 setup`: data split, targets, the training loop, temperature
calibration, and saving a checkpoint (with its familiarity guard).
"""

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .features import FEATURE_VERSION, STOP_INDEX, featurize
from .guard import FamiliarityGuard
from .kernels import KERNELS, WORKLOAD_NAMES
from .net import HPCDS1Net
from .space import ACTIONS

SIZES = {
    "tiny": dict(dim=64, encoder_blocks=1, fusion_layers=1, heads=4, ffn_dim=256, dropout=0.1),
    "small": dict(dim=128, encoder_blocks=1, fusion_layers=2, heads=4, ffn_dim=512, dropout=0.1),
    "medium": dict(dim=256, encoder_blocks=2, fusion_layers=2, heads=8, ffn_dim=1024, dropout=0.15),
}
TAU = 0.02       # moves whose eventual runtime is within ~2% share the credit
GOOD_TOL = 1.02  # a choice is "a best move" if it can reach within 2% of the best reachable runtime


def soft_target(row):
    reach = row["action_reach"]
    best = min(v for v in reach.values() if v is not None)
    scores = [(-math.log(reach[a] / best) / TAU) if reach.get(a) is not None else -1e9 for a in ACTIONS]
    m = max(scores)
    w = [math.exp(x - m) for x in scores]
    z = sum(w)
    return [x / z for x in w]


def is_good(row, action):
    r = row["action_reach"].get(action)
    return r is not None and r <= min(v for v in row["action_reach"].values() if v is not None) * GOOD_TOL


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def split(states, seed=0):
    test_sizes = {w: KERNELS[w].test_size for w in WORKLOAD_NAMES}
    test = [s for s in states if s["size"] == test_sizes[s["workload"]]]
    rest = [s for s in states if s["size"] != test_sizes[s["workload"]]]
    rng = random.Random(seed)
    rng.shuffle(rest)
    n_val = max(1, int(len(rest) * 0.15))
    return rest[n_val:], rest[:n_val], test


def tensors(rows, device):
    feats = featurize([r["state"] for r in rows], device=device)
    y = torch.tensor([ACTIONS.index(r["best_action"]) for r in rows], device=device)
    good = torch.tensor([[is_good(r, a) for a in ACTIONS] for r in rows], device=device)
    soft = torch.tensor([soft_target(r) for r in rows], dtype=torch.float32, device=device)
    return {"feats": feats, "y": y, "soft": soft, "good": good, "stop": (y == STOP_INDEX).float(),
            "imp": torch.tensor([r["best_improvement"] for r in rows], dtype=torch.float32, device=device)}


def take(b, idx):
    return {"feats": {k: v[idx] for k, v in b["feats"].items()}, "y": b["y"][idx], "stop": b["stop"][idx],
            "imp": b["imp"][idx], "soft": b["soft"][idx], "good": b["good"][idx]}


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

def soft_ce(logits, soft):
    return -(soft * F.log_softmax(logits, dim=1)).sum(1).mean()


def good_rate(logits, good):
    return good.gather(1, logits.argmax(1, keepdim=True)).float().mean().item()


def loss_fn(out, b, cw):
    action = soft_ce(out["logits"], b["soft"])
    stop = F.binary_cross_entropy_with_logits(out["stop_logit"], b["stop"])
    imp = F.smooth_l1_loss(out["improvement"], b["imp"], beta=0.05)
    return action + 0.5 * stop + 1.0 * imp


@torch.no_grad()
def evaluate(net, b):
    net.eval()
    out = net(b["feats"])
    return soft_ce(out["logits"], b["soft"]).item(), good_rate(out["logits"], b["good"]), out


def train_one(size, tr, va, device, epochs=300, seed=0, batch=64, patience=40, lr=1e-3):
    torch.manual_seed(seed)
    net = HPCDS1Net(**SIZES[size], workloads=WORKLOAD_NAMES).to(device)
    counts = torch.bincount(tr["y"], minlength=len(ACTIONS)).float()
    cw = torch.where(counts > 0, (counts.sum() / (len(ACTIONS) * counts.clamp(min=1))) ** 0.5,
                     torch.ones_like(counts)).clamp(0.5, 4.0)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    best, bad, hist, t0 = {"val_loss": float("inf")}, 0, [], time.time()
    n = len(tr["y"])
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        tl, corr = 0.0, 0
        for i in range(0, n, batch):
            b = take(tr, perm[i:i + batch])
            out = net(b["feats"])
            loss = loss_fn(out, b, cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(b["y"])
            corr += good_rate(out["logits"], b["good"]) * len(b["y"])
        sched.step()
        vl, va_acc, _ = evaluate(net, va)
        hist.append({"epoch": ep + 1, "train_loss": round(tl / n, 5), "train_acc": round(corr / n, 5),
                     "val_loss": round(vl, 5), "val_acc": round(va_acc, 5)})
        if vl < best["val_loss"] - 1e-4:
            best = {"val_loss": vl, "val_acc": va_acc, "epoch": ep + 1,
                    "state": {k: v.detach().clone() for k, v in net.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    net.load_state_dict(best["state"])
    return net, {"size": size, "hparams": SIZES[size], "parameters": net.count_parameters(),
                 "best_epoch": best["epoch"], "epochs_run": len(hist), "val_loss": round(best["val_loss"], 5),
                 "val_acc": round(best["val_acc"], 5), "train_time_sec": round(time.time() - t0, 1),
                 "history": hist}


@torch.no_grad()
def fit_temperature(logits, good):
    """Temperature that best calibrates 'the chosen move is a best move'."""
    grid = torch.exp(torch.linspace(math.log(0.2), math.log(8.0), 160, device=logits.device))
    ok = good.gather(1, logits.argmax(1, keepdim=True)).squeeze(1).cpu()
    eces = [reliability(F.softmax(logits / t, 1).max(1).values.cpu(), ok)[0] for t in grid]
    return grid[int(np.argmin(eces))].item()


def reliability(conf, correct, bins=10):
    edges = torch.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = correct[m].float().mean().item(), conf[m].mean().item()
            ece += m.float().mean().item() * abs(acc - c)
            table.append({"lo": round(lo.item(), 2), "hi": round(hi.item(), 2), "count": int(m.sum()),
                          "accuracy": round(acc, 4), "confidence": round(c, 4)})
    return round(ece, 4), table


def save_checkpoint(path, net, meta, guard=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hparams": net.hparams, "state_dict": net.state_dict(), "feature_version": FEATURE_VERSION,
                "meta": meta, "guard": guard.to_dict() if guard else None}, path)


def fit_guard(tr_rows, va_rows, hardware):
    return FamiliarityGuard.fit([r["state"] for r in tr_rows], [r["state"] for r in va_rows], hardware)
