#!/usr/bin/env python3
"""
Fine-tune the Laya-style question encoder (hpc_ds1/encoder.py) on questions
answered from the real measurements, and evaluate it honestly.

  splits     test = every question about the held-out test sizes (never trained on)
             phrasings: the last wording of every question family is never trained on
             -> report seen/unseen wording x seen/unseen problem
  baseline   always answering the most common answer of that question family
  outputs    models/encoder/ (fine-tuned weights, git-ignored), results/encoder.json

    python scripts/train_encoder.py            # ~15-25 min on an RTX 4050
"""

import json
import os
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from hpc_ds1.dataset import build_states, build_tables, load_hardware, load_measurements  # noqa: E402
from hpc_ds1.encoder import BASE_MODEL, build_examples, encode, load_tokenizer_and_model  # noqa: E402
from hpc_ds1.kernels import KERNELS  # noqa: E402
from hpc_ds1.training import fit_temperature, reliability  # noqa: E402

warnings.filterwarnings("ignore")
OUT = PROJECT_ROOT / "models" / "encoder"
EPOCHS, BATCH, LR = int(os.environ.get("ENCODER_EPOCHS", 5)), 16, 3e-5


def log(m=""):
    print(m, flush=True)


@torch.no_grad()
def predict(model, tok, exs, device, bs=64):
    model.eval()
    logits = []
    for i in range(0, len(exs), bs):
        chunk = exs[i:i + bs]
        b = encode(tok, [(e.question, e.options, e.state_text) for e in chunk], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(**b).float()
        pad = torch.full((lg.shape[0], 10), -1e4, device=lg.device)
        pad[:, :lg.shape[1]] = lg
        logits.append(pad.cpu())
    return torch.cat(logits)


def score(exs, logits, temperature):
    y = torch.tensor([e.answer for e in exs])
    p = F.softmax(logits / temperature, -1)
    conf, pred = p.max(1)
    ok = pred == y
    ece, _ = reliability(conf, ok)
    return {"n": len(exs), "accuracy": round(ok.float().mean().item(), 4), "ece": ece}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = load_hardware()
    tables, failed = build_tables(load_measurements())
    states = build_states(tables, hardware)
    exs = build_examples(states, tables, failed, hardware, per_state=4, seed=0)
    test_sizes = {(w, k.test_size) for w, k in KERNELS.items()}
    on_test = [e for e in exs if (e.workload, e.size) in test_sizes]
    rest = [e for e in exs if (e.workload, e.size) not in test_sizes]
    train_pool = [e for e in rest if not e.held_out_phrasing]
    unseen_wording = [e for e in rest if e.held_out_phrasing]
    random.Random(1).shuffle(train_pool)
    n_val = int(len(train_pool) * 0.1)
    val, train = train_pool[:n_val], train_pool[n_val:]
    log(f"{len(exs)} questions from {len(states)} measured states: train {len(train)}, val {len(val)}, "
        f"unseen wording {len(unseen_wording)}, test sizes {len(on_test)}")
    log("families: " + ", ".join(f"{k} {v}" for k, v in Counter(e.family for e in exs).items()))

    tok, model = load_tokenizer_and_model(BASE_MODEL, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * (len(train) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    t0, step = time.time(), 0
    for ep in range(EPOCHS):
        random.Random(100 + ep).shuffle(train)
        model.train()
        for i in range(0, len(train) - BATCH + 1, BATCH):
            chunk = train[i:i + BATCH]
            b = encode(tok, [(e.question, e.options, e.state_text) for e in chunk], device=device)
            y = torch.tensor([e.answer for e in chunk], device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(**b).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                log(f"  epoch {ep + 1} step {step}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")

    vl = predict(model, tok, val, device)
    temp = fit_temperature(vl, torch.nn.functional.one_hot(torch.tensor([e.answer for e in val]), 10).bool())
    model.temperature.fill_(temp)

    # most-common-answer baseline per family (from training data)
    common = {f: Counter(e.answer for e in train if e.family == f).most_common(1)[0][0]
              for f in {e.family for e in train}}

    def report(name, subset):
        if not subset:
            return None
        lg = predict(model, tok, subset, device)
        r = score(subset, lg, temp)
        r["baseline_most_common"] = round(float(np.mean([common.get(e.family, 0) == e.answer for e in subset])), 4)
        fam = defaultdict(list)
        pred = lg.argmax(1)
        for e, p in zip(subset, pred.tolist()):
            fam[e.family].append(p == e.answer)
        r["per_family"] = {f: round(float(np.mean(v)), 3) for f, v in fam.items()}
        log(f"{name:38s} accuracy {100 * r['accuracy']:5.1f}%  (most-common baseline {100 * r['baseline_most_common']:5.1f}%)  "
            f"ECE {r['ece']:.3f}  n={r['n']}")
        return r

    res = {"base_model": BASE_MODEL, "epochs": EPOCHS, "temperature": round(temp, 4), "train_questions": len(train),
           "train_minutes": round((time.time() - t0) / 60, 1)}
    res["seen_wording_seen_problems(val)"] = report("validation (seen wording)", val)
    res["unseen_wording"] = report("unseen wording, seen problems", unseen_wording)
    res["test_sizes_seen_wording"] = report("unseen problems (test sizes)", [e for e in on_test if not e.held_out_phrasing])
    res["test_sizes_unseen_wording"] = report("unseen problems AND unseen wording", [e for e in on_test if e.held_out_phrasing])

    t1 = time.perf_counter()
    for e in on_test[:50]:
        predict(model, tok, [e], device, bs=1)
    res["latency_ms_per_question"] = round((time.perf_counter() - t1) * 1000 / 50, 2)
    log(f"latency {res['latency_ms_per_question']} ms per question on {device}")

    OUT.mkdir(parents=True, exist_ok=True)
    model.base.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    torch.save({"score": model.score.state_dict(), "temperature": temp,
                "meta": {k: v for k, v in res.items() if not isinstance(v, dict)}}, OUT / "scorer.pt")
    (PROJECT_ROOT / "results" / "encoder.json").write_text(json.dumps(res, indent=2))
    log(f"saved {OUT} and results/encoder.json")


if __name__ == "__main__":
    main()
