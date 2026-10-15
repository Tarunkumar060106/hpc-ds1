#!/usr/bin/env python3
"""
Does answering "will this change help?" from System 1's estimate beat the encoder?

Same questions and splits as scripts/train_encoder.py (out-of-fold System 1
hints, held-out wordings and synonyms, held-out test sizes). For every
"action_helps" question the grounded router (hpc_ds1.encoder.grounded_answer)
is compared with the fine-tuned encoder on exactly the same questions. The
router's confidence table (accuracy by size of the estimated change) is
measured on the training split only and reported on the others.

    python scripts/eval_router.py   -> results/experiments/router_vs_encoder.json
"""

import json
import math
import os
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from hpc_ds1.dataset import build_states, build_tables, load_hardware, load_measurements  # noqa: E402
from hpc_ds1.encoder import NOUL, grounded_answer, oof_hint_provider, build_examples, load_tokenizer_and_model  # noqa: E402
from hpc_ds1.kernels import KERNELS  # noqa: E402
from hpc_ds1.paths import encoder_dir  # noqa: E402
from train_encoder import predict  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = load_hardware()
    tables, failed = build_tables(load_measurements())
    states = build_states(tables, hardware)
    provide = oof_hint_provider(PROJECT_ROOT / "results" / "system1_oof.json")
    exs = build_examples(states, tables, failed, hardware, per_state=4, seed=0, hints=provide)
    # recover each example's hints (same order as build_examples walks the states)
    by_text = {}
    for row in states:
        h = provide(row["state"])
        from hpc_ds1.encoder import describe
        by_text[describe(row["state"], h)] = h
    test_sizes = {(w, k.test_size) for w, k in KERNELS.items()}
    split = {}
    for e in exs:
        if (e.workload, e.size) in test_sizes:
            split[id(e)] = "test sizes + unseen wording" if e.held_out_phrasing else "test sizes"
        else:
            split[id(e)] = "unseen wording" if e.held_out_phrasing else "train"
    helps = [e for e in exs if e.family == "action_helps"]

    # router confidence table from the training split only
    buckets = [(0.05, []), (0.2, []), (float("inf"), [])]
    rows = []
    for e in helps:
        g = grounded_answer(e.question, by_text.get(e.state_text))
        rows.append(g)
        if g is not None and split[id(e)] == "train":
            next(b for edge, b in buckets if g[2] <= edge).append(NOUL.index(g[0]) == e.answer)
    table = [(edge, round(float(np.mean(v)), 3) if v else None, len(v)) for edge, v in buckets]
    print("router accuracy by |log ratio| on the training split:", table)

    tok, model = load_tokenizer_and_model(str(encoder_dir()), device=device)
    extra = torch.load(encoder_dir() / "scorer.pt", map_location=device, weights_only=False)
    model.score.load_state_dict(extra["score"])
    model.temperature.fill_(extra["temperature"])
    enc_pred = predict(model, tok, helps, device).argmax(1).tolist()

    res = {"confidence_table_train": table, "splits": {}}
    for name in ("unseen wording", "test sizes", "test sizes + unseen wording"):
        idx = [i for i, e in enumerate(helps) if split[id(e)] == name]
        covered = [i for i in idx if rows[i] is not None]
        enc_all = float(np.mean([enc_pred[i] == helps[i].answer for i in idx]))
        hybrid = float(np.mean([(NOUL.index(rows[i][0]) if rows[i] else enc_pred[i]) == helps[i].answer for i in idx]))
        r = {"questions": len(idx), "router_coverage": round(len(covered) / max(1, len(idx)), 3),
             "router_accuracy_where_used": round(float(np.mean([NOUL.index(rows[i][0]) == helps[i].answer for i in covered])), 3) if covered else None,
             "encoder_accuracy_same_questions": round(float(np.mean([enc_pred[i] == helps[i].answer for i in covered])), 3) if covered else None,
             "encoder_alone": round(enc_all, 3), "router_then_encoder": round(hybrid, 3)}
        res["splits"][name] = r
        print(f"{name:30s} n={r['questions']:4d}  router covers {100 * r['router_coverage']:.0f}%: "
              f"router {r['router_accuracy_where_used']} vs encoder {r['encoder_accuracy_same_questions']} on those | "
              f"overall encoder {r['encoder_alone']} -> router+encoder {r['router_then_encoder']}")
    out = PROJECT_ROOT / "results" / "experiments" / "router_vs_encoder.json"
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
