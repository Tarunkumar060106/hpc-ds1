"""Free-text questions about a job, answered Jev/Laya-style by a fine-tuned encoder.

Laya's design, specialised to HPC: one bidirectional encoder (ModernBERT) reads

    question  [SEP]  [OPT] option 1  [OPT] option 2 ...  [SEP]  the job, as text

in a single pass; the hidden state at each [OPT] marker is scored, and a
softmax over the options (temperature-calibrated) gives the typed answer and
its confidence. Nothing is generated.

Every training question is answered from the real measurements, never made
up: "will bf16 be faster here?" is labelled by the measured runtimes of the
two configurations. Each question family has several phrasings; some are held
out of training, so the evaluation shows how it handles wording it has never
seen. The familiarity guard still decides whether an answer may be acted on.

Question types follow Jev: "noul" (yes / no / unclear) and "choice" (pick one
of the given options).
"""

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import KERNELS
from .space import ACTION_HELP, ACTIONS, STOP, apply_action, config_key, make_config, valid_actions

BASE_MODEL = "answerdotai/ModernBERT-base"
OPT = "[OPT]"
NOUL = ["yes", "no", "unclear"]
CLEAR = 1.05  # a change counts as faster/slower only beyond 5% (the measurement noise floor)


# ---------------------------------------------------------------------------
# the job, as text
# ---------------------------------------------------------------------------

def config_text(c):
    if c["gpu"]:
        return f"on the GPU with {c['threads']} CPU threads, {c['precision']}"
    return f"{c['ranks']} rank{'s' if c['ranks'] > 1 else ''} x {c['threads']} thread{'s' if c['threads'] > 1 else ''} on the CPU, {c['precision']}"


def describe(state):
    """One state (hardware, workload, config, telemetry) as the text the encoder reads."""
    hw, wl, c, t = state["hardware"], state["workload"], state["config"], state.get("telemetry") or {}
    k = KERNELS.get(wl["name"])
    job = f"{k.display_name} ({wl['name']}), size {wl['size']:,} {k.size_unit}" if k else f"{wl['name']}, size {wl['size']:,}"
    gpu = f"GPU {hw['gpu_model']} with {hw.get('gpu_memory_gb', 0)} GB" if hw.get("gpu_model") else "no GPU"
    text = (f"Job: {job}. Machine: {hw.get('cpu_cores')} cores / {hw.get('cpu_threads')} threads, "
            f"{hw.get('ram_gb')} GB RAM, {gpu}. Current setup: {config_text(c)}.")
    if t.get("runtime_sec") is not None:
        text += (f" Measured: runtime {t['runtime_sec'] * 1000:.2f} ms, CPU busy {t.get('cpu_util_pct', 0):.0f}%, "
                 f"GPU busy {t.get('gpu_busy_pct', 0):.0f}%, communication {t.get('comm_overhead_pct', 0):.0f}%.")
    return text


# ---------------------------------------------------------------------------
# questions, answered from the measurements
# ---------------------------------------------------------------------------
# Each family: list of phrasings (the last ones are held out of training) and a
# labeller(row, table, has_gpu, rng) -> (fill dict, options, answer index) or None.

def _rt(table, c):
    t = table.get(config_key(c))
    return t["runtime_sec"] if t else None


def _verdict(before, after):
    if after is None:
        return 1  # the change fails or gives a wrong answer: "no"
    if after < before / CLEAR:
        return 0  # clearly faster: "yes"
    if after > before * CLEAR:
        return 1  # clearly slower: "no"
    return 2      # within the noise: "unclear"


ACTION_WORDS = {
    "ADD_RANKS": "doubling the number of ranks", "REMOVE_RANKS": "halving the number of ranks",
    "ADD_THREADS": "doubling the threads per rank", "REMOVE_THREADS": "halving the threads per rank",
    "USE_GPU": "moving the job to the GPU", "USE_CPU": "moving the job back to the CPU",
    "USE_BF16": "switching to bf16 precision", "USE_FP32": "switching back to fp32 precision",
}


def _q_action_helps(row, table, has_gpu, rng):
    c = row["state"]["config"]
    acts = [a for a in valid_actions(c, has_gpu)]
    if not acts:
        return None
    a = rng.choice(acts)
    return {"change": ACTION_WORDS[a]}, NOUL, _verdict(row["runtime"], _rt(table, apply_action(c, a)))


def _q_compare(row, table, has_gpu, rng):
    keys = [k for k in table]
    if len(keys) < 2:
        return None
    a, b = rng.sample(keys, 2)
    ca, cb = make_config(*a), make_config(*b)
    ra, rb = table[a]["runtime_sec"], table[b]["runtime_sec"]
    if max(ra, rb) < min(ra, rb) * CLEAR:
        return None  # too close to call honestly
    opts = [config_text(ca), config_text(cb)]
    return {"a": opts[0], "b": opts[1]}, opts, 0 if ra < rb else 1


def _q_next_move(row, table, has_gpu, rng):
    c = row["state"]["config"]
    acts = valid_actions(c, has_gpu) + [STOP]
    words = [ACTION_HELP[a] for a in acts]
    best = row["best_action"]
    if best not in acts:
        return None
    return {}, words, acts.index(best)


def _q_near_best(row, table, has_gpu, rng):
    ratio = row["runtime"] / row["instance_best_runtime"]
    return {}, NOUL, 0 if ratio <= 1.10 else (1 if ratio > 1.25 else 2)


def _q_device(row, table, has_gpu, rng):
    if not has_gpu:
        return None
    best = min(table, key=lambda k: table[k]["runtime_sec"])
    return {}, ["the CPU", "the GPU"], 1 if best[2] else 0


def _q_precision(row, table, has_gpu, rng):
    best = min(table, key=lambda k: table[k]["runtime_sec"])
    return {}, ["fp32", "bf16"], 1 if best[3] == "bf16" else 0


def _q_fails(row, table, has_gpu, rng, failed=None):
    return None  # filled in by build_examples, which knows the failed configs


FAMILIES = {
    "action_helps": (["Will {change} make this job faster?",
                      "Would {change} speed this up?",
                      "If we try {change}, does the runtime go down?",
                      "Is {change} a good idea for performance here?",
                      "Does {change} help this job run quicker?",
                      "Could {change} reduce the time this takes?"], _q_action_helps, "noul"),
    "compare": (["Which setup is faster for this job: {a} or {b}?",
                 "Between {a} and {b}, which one finishes sooner?",
                 "Should this job run as {a} or as {b} to be quicker?",
                 "Which would run this job faster, {a} or {b}?",
                 "Pick the quicker configuration: {a} versus {b}."], _q_compare, "choice"),
    "next_move": (["What single change should I make next?",
                   "What is the best next tuning step?",
                   "Which one adjustment would help this job most right now?",
                   "If you could change one setting, which one?",
                   "What should we tweak next to speed this up?"], _q_next_move, "choice"),
    "near_best": (["Is this setup already close to the best possible?",
                   "Is this configuration within 10% of the fastest one?",
                   "Can we stop tuning, is this near optimal?",
                   "Is there little left to gain from tuning this job?",
                   "Is the current setup about as fast as it gets?"], _q_near_best, "noul"),
    "device": (["Should this job run on the CPU or the GPU?",
                "Which device is best for this job?",
                "Where will this job run fastest, CPU or GPU?",
                "Is the GPU or the CPU the right place for this workload?"], _q_device, "choice"),
    "precision": (["Which precision is fastest for this job, fp32 or bf16?",
                   "Should this job use fp32 or bf16?",
                   "Is fp32 or bf16 the better choice here for speed?",
                   "What number format runs this job quickest?"], _q_precision, "choice"),
    "fails": (["Will {cfg} crash or give a wrong answer for this job?",
               "Is it safe to run this job as {cfg}?",
               "Does running this as {cfg} fail?",
               "Would {cfg} break this job?"], _q_fails, "noul"),
}
HELD_OUT_PHRASINGS = 1  # the last phrasing of every family is never used in training

# Synonyms people actually use. Training questions randomly swap in the "train"
# words; the "held_out" words appear only in the unseen-wording evaluation, so it
# also measures vocabulary the encoder was never trained on.
SYNONYMS = {
    "GPU": {"train": ["CUDA device", "accelerator", "NVIDIA GPU"], "held_out": ["graphics card"]},
    "CPU": {"train": ["processor", "host CPU"], "held_out": ["CPU cores"]},
    "bf16": {"train": ["bfloat16", "half precision"], "held_out": ["16-bit floats"]},
    "fp32": {"train": ["single precision", "float32"], "held_out": ["32-bit floats"]},
    "ranks": {"train": ["processes", "workers"], "held_out": ["MPI ranks"]},
    "threads": {"train": ["CPU threads", "OpenMP threads"], "held_out": ["threads per process"]},
    "faster": {"train": ["quicker", "speedier"], "held_out": ["less time-consuming"]},
}


def _swap_words(q, rng, held_out, p=0.6):
    import re
    for word, alts in SYNONYMS.items():
        pool = alts["held_out"] if held_out else alts["train"]
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, q) and rng.random() < p:
            q = re.sub(pattern, rng.choice(pool), q, count=1)
    return q


@dataclass
class Example:
    question: str
    options: List[str]
    answer: int
    family: str
    kind: str
    state_text: str
    workload: str
    size: int
    phrasing: int
    held_out_phrasing: bool


def build_examples(states, tables, failed, hardware, per_state=3, seed=0):
    """Examples for each state (dataset.build_states rows), answered from the measurements.
    failed: {(workload, size): {config_key: reason}} from dataset.build_tables."""
    rng = random.Random(seed)
    has_gpu = bool(hardware.get("gpu_model"))
    out = []
    for row in states:
        inst = (row["workload"], row["size"])
        table = tables[inst]
        text = describe(row["state"])
        fams = rng.sample(list(FAMILIES), min(per_state, len(FAMILIES)))
        for fam in fams:
            phrasings, labeller, kind = FAMILIES[fam]
            if fam == "fails":
                bad = list(failed.get(inst, {}))
                good = list(table)
                if not bad:
                    continue
                is_bad = rng.random() < 0.5
                c = make_config(*rng.choice(bad if is_bad else good))
                got = ({"cfg": config_text(c)}, NOUL, 0 if is_bad else 1)
            else:
                got = labeller(row, table, has_gpu, rng)
            if got is None:
                continue
            fill, options, answer = got
            p = rng.randrange(len(phrasings))
            held = p >= len(phrasings) - HELD_OUT_PHRASINGS
            q = _swap_words(phrasings[p].format(**fill), rng, held)
            if fam == "fails" and "safe" in q:  # "is it safe" inverts the yes/no meaning
                answer = {0: 1, 1: 0}.get(answer, answer)
            out.append(Example(q, list(options), answer, fam, kind, text, row["workload"], row["size"], p, held))
    return out


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class OptionScorer(nn.Module):
    """ModernBERT encoder + a linear scorer applied at every [OPT] marker."""

    def __init__(self, base, hidden):
        super().__init__()
        self.base = base
        self.score = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 1))
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask):
        h = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = marker_pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, h.shape[-1])
        logits = self.score(torch.gather(h, 1, idx)).squeeze(-1)
        return logits.masked_fill(~marker_mask, -1e4)


def load_tokenizer_and_model(path=BASE_MODEL, cache_dir=None, device="cpu"):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, cache_dir=cache_dir)
    if OPT not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [OPT]})
    # eager torch: no torch.compile / Triton, which aren't available on Windows
    base = AutoModel.from_pretrained(path, cache_dir=cache_dir, attn_implementation="sdpa")
    base.resize_token_embeddings(len(tok))
    return tok, OptionScorer(base, base.config.hidden_size).to(device)


def encode(tok, items, max_len=320, device="cpu"):
    """items: [(question, options, state_text)] -> batch tensors."""
    texts = [f"{q} {tok.sep_token} " + " ".join(f"{OPT} {o}" for o in opts) + f" {tok.sep_token} {s}"
             for q, opts, s in items]
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    opt_id = tok.convert_tokens_to_ids(OPT)
    n_opt = max(len(o) for _, o, _ in items)
    pos = torch.full((len(items), n_opt), -1, dtype=torch.long)
    for i in range(len(items)):
        where = (enc["input_ids"][i] == opt_id).nonzero().flatten()[:n_opt]
        pos[i, :len(where)] = where
    return {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device),
            "marker_pos": pos.to(device), "marker_mask": (pos >= 0).to(device)}


@dataclass
class Answer:
    key: str
    question: str
    kind: str
    choice: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float = 0.0
    familiar: bool = True
    reasons: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class Answerer:
    """Loads the fine-tuned encoder and answers typed free-text questions about a state."""

    def __init__(self, path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok, self.model = load_tokenizer_and_model(str(path), device=self.device)
        extra = torch.load(f"{path}/scorer.pt", map_location=self.device, weights_only=False)
        self.model.score.load_state_dict(extra["score"])
        self.model.temperature.fill_(extra["temperature"])
        self.model.eval()
        self.meta = extra.get("meta", {})

    @torch.no_grad()
    def ask(self, state, questions: Dict[str, dict], guard=None) -> List[Answer]:
        import time
        t0 = time.perf_counter()
        text = describe(state)
        items, specs = [], []
        for key, q in questions.items():
            kind = q.get("type", "noul")
            opts = list(q.get("options") or (NOUL if kind == "noul" else []))
            if not opts:
                raise ValueError(f"question {key!r} of type {kind!r} needs options")
            items.append((q.get("question", key), opts, text))
            specs.append((key, q.get("question", key), kind, opts))
        batch = encode(self.tok, items, device=self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = self.model(**batch).float()
        probs = F.softmax(logits / self.model.temperature, -1).cpu()
        check = guard.check([state])[0] if guard else {"familiar": True, "reasons": []}
        ms = (time.perf_counter() - t0) * 1000 / max(1, len(items))
        out = []
        for i, (key, qtext, kind, opts) in enumerate(specs):
            p = probs[i, :len(opts)]
            p = p / p.sum()
            j = int(p.argmax())
            out.append(Answer(key, qtext, kind, opts[j], round(float(p[j]), 4),
                              {o: round(float(v), 4) for o, v in zip(opts, p)}, round(ms, 2),
                              check["familiar"], check["reasons"]))
        return out
