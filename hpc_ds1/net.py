"""HPC-DS1 network: modular residual encoders -> attention fusion -> decision heads.

Pre-norm residual blocks and pre-norm transformer layers throughout. The
previous post-norm stack (torch's default TransformerEncoderLayer) collapsed
to a constant output once it got deep; pre-norm keeps an identity path from
every input to the heads, and hand-rolled fusion layers let us expose the
per-head attention maps for the layer explorer UI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import MODULE_ORDER, WORKLOAD_NAMES, module_features
from .space import ACTIONS

TRACE_BINS = 64


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.norm(x)))))


class ModuleEncoder(nn.Module):
    def __init__(self, in_dim, dim, num_blocks, dropout):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([ResidualBlock(dim, dropout) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x, rec=None, prefix=""):
        h = self.input_proj(x)
        if rec is not None:
            rec[f"{prefix}.input_proj"] = h
        for i, block in enumerate(self.blocks):
            h = block(h)
            if rec is not None:
                rec[f"{prefix}.block{i}"] = h
        h = self.out_norm(h)
        if rec is not None:
            rec[f"{prefix}.out_norm"] = h
        return h


class FusionLayer(nn.Module):
    def __init__(self, dim, heads, ffn_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, need_weights=False):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=need_weights, average_attn_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, a, w


class HPCDS1Net(nn.Module):
    def __init__(self, dim=512, encoder_blocks=2, fusion_layers=3, heads=8, ffn_dim=2048,
                 dropout=0.1, workloads=None):
        super().__init__()
        self.workloads = list(workloads or WORKLOAD_NAMES)
        self.modules_in = module_features(self.workloads)
        self.hparams = dict(dim=dim, encoder_blocks=encoder_blocks, fusion_layers=fusion_layers,
                            heads=heads, ffn_dim=ffn_dim, dropout=dropout, workloads=self.workloads)
        self.encoders = nn.ModuleDict({
            m: ModuleEncoder(len(self.modules_in[m]), dim, encoder_blocks, dropout) for m in MODULE_ORDER
        })
        self.type_embed = nn.Parameter(torch.randn(len(MODULE_ORDER), dim) * 0.02)
        self.layers = nn.ModuleList([FusionLayer(dim, heads, ffn_dim, dropout)
                                     for _ in range(fusion_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.pool_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.action_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, len(ACTIONS)))
        self.stop_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.improvement_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(),
                                              nn.Linear(dim // 2, 1))
        # Fit post-hoc on the validation set (temperature scaling), not trained.
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, feats, trace=False):
        rec = {} if trace else None
        tokens = []
        for m in MODULE_ORDER:
            if trace:
                rec[f"input.{m}"] = feats[m]
            tokens.append(self.encoders[m](feats[m], rec, f"encoder.{m}"))
        t = torch.stack(tokens, dim=1) + self.type_embed
        if trace:
            rec["fusion.tokens"] = t

        attention = []
        for i, layer in enumerate(self.layers):
            t, a, w = layer(t, need_weights=trace)
            if trace:
                rec[f"fusion.layer{i}.attn_out"] = a
                rec[f"fusion.layer{i}.out"] = t
                attention.append(w)
        t = self.final_norm(t)
        pooled = t.mean(dim=1)
        h = self.pool_proj(pooled)
        if trace:
            rec["fusion.final_norm"] = t
            rec["pool.mean"] = pooled
            rec["pool.proj"] = h

        action_hidden = self.action_head[1](self.action_head[0](h))
        logits = self.action_head[2](action_hidden)
        if "mask" in feats:
            logits = logits.masked_fill(~feats["mask"], -1e4)
        stop_logit = self.stop_head(h).squeeze(-1)
        improvement = self.improvement_head(h).squeeze(-1)

        out = {
            "logits": logits,
            "probs": F.softmax(logits / self.temperature, dim=-1),
            "stop_logit": stop_logit,
            "improvement": improvement,
        }
        if trace:
            rec["head.action.hidden"] = action_hidden
            rec["head.action.logits"] = logits
            rec["head.stop"] = stop_logit.unsqueeze(-1)
            rec["head.improvement"] = improvement.unsqueeze(-1)
            out["trace"] = rec
            out["attention"] = attention
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Layer catalog + trace summaries for the explorer UI
    # ------------------------------------------------------------------

    def layer_catalog(self):
        """Ordered description of every layer the explorer shows."""
        hp = self.hparams
        d = hp["dim"]
        n = lambda mod: sum(p.numel() for p in mod.parameters())  # noqa: E731
        cat = []
        for m in MODULE_ORDER:
            cat.append(dict(id=f"input.{m}", group="Inputs", module=m, kind="input",
                            label=f"{m} features", shape=[len(self.modules_in[m])], params=0,
                            features=self.modules_in[m]))
        for m in MODULE_ORDER:
            enc = self.encoders[m]
            cat.append(dict(id=f"encoder.{m}.input_proj", group="Encoders", module=m, kind="vector",
                            label=f"{m} · input projection", shape=[d], params=n(enc.input_proj),
                            op=f"Linear({len(self.modules_in[m])}→{d})"))
            for i, b in enumerate(enc.blocks):
                cat.append(dict(id=f"encoder.{m}.block{i}", group="Encoders", module=m, kind="vector",
                                label=f"{m} · residual block {i + 1}", shape=[d], params=n(b),
                                op=f"x + Linear(GELU(Linear(LayerNorm(x))))  {d}→{d}→{d}"))
            cat.append(dict(id=f"encoder.{m}.out_norm", group="Encoders", module=m, kind="vector",
                            label=f"{m} · output LayerNorm", shape=[d], params=n(enc.out_norm),
                            op="LayerNorm"))
        cat.append(dict(id="fusion.tokens", group="Fusion", kind="tokens",
                        label="token sequence + module-type embedding", shape=[4, d],
                        params=self.type_embed.numel(), op="stack(4 encoders) + type_embed"))
        for i, layer in enumerate(self.layers):
            cat.append(dict(id=f"fusion.layer{i}.attention", group="Fusion", kind="attention", layer=i,
                            label=f"fusion layer {i + 1} · self-attention weights",
                            shape=[hp["heads"], 4, 4], params=n(layer.norm1) + n(layer.attn),
                            op=f"MultiheadAttention({hp['heads']} heads, d={d})"))
            cat.append(dict(id=f"fusion.layer{i}.attn_out", group="Fusion", kind="tokens",
                            label=f"fusion layer {i + 1} · attention output", shape=[4, d], params=0,
                            op="softmax(QKᵀ/√d_head)·V, then out-proj"))
            cat.append(dict(id=f"fusion.layer{i}.out", group="Fusion", kind="tokens",
                            label=f"fusion layer {i + 1} · after feed-forward", shape=[4, d],
                            params=n(layer.norm2) + n(layer.ffn),
                            op=f"x + FFN(LayerNorm(x))  {d}→{hp['ffn_dim']}→{d}"))
        cat.append(dict(id="fusion.final_norm", group="Fusion", kind="tokens",
                        label="final LayerNorm", shape=[4, d], params=n(self.final_norm), op="LayerNorm"))
        cat.append(dict(id="pool.mean", group="Pooling", kind="vector", label="mean over 4 tokens",
                        shape=[d], params=0, op="mean(dim=tokens)"))
        cat.append(dict(id="pool.proj", group="Pooling", kind="vector", label="pooled projection",
                        shape=[d], params=n(self.pool_proj), op=f"GELU(Linear({d}→{d}))"))
        cat.append(dict(id="head.action.hidden", group="Heads", kind="vector", label="action head · hidden",
                        shape=[d], params=n(self.action_head[0]), op=f"GELU(Linear({d}→{d}))"))
        cat.append(dict(id="head.action.logits", group="Heads", kind="logits", label="action head · logits",
                        shape=[len(ACTIONS)], params=n(self.action_head[2]),
                        op=f"Linear({d}→{len(ACTIONS)}), invalid actions masked", names=ACTIONS))
        cat.append(dict(id="head.stop", group="Heads", kind="logits", label="stop head · logit",
                        shape=[1], params=n(self.stop_head), op=f"{d}→{d // 2}→1", names=["stop"]))
        cat.append(dict(id="head.improvement", group="Heads", kind="logits",
                        label="expected-improvement head", shape=[1], params=n(self.improvement_head),
                        op=f"{d}→{d // 2}→1", names=["improvement"]))
        return cat


def _vec_summary(v):
    v = v.float()
    d = v.shape[-1]
    bins = min(TRACE_BINS, d)
    binned = v[: (d // bins) * bins].reshape(bins, -1).mean(dim=1)
    return {
        "bins": [round(x, 4) for x in binned.tolist()],
        "mean": round(v.mean().item(), 4),
        "std": round(v.std().item(), 4) if d > 1 else 0.0,
        "min": round(v.min().item(), 4),
        "max": round(v.max().item(), 4),
        "l2": round(v.norm().item(), 3),
        "frac_negative": round((v < 0).float().mean().item(), 3),
    }


def summarize_trace(out, index=0):
    """Compact JSON-able per-layer trace for one sample of a traced batch."""
    rec, attention = out["trace"], out["attention"]
    layers = {}
    for key, t in rec.items():
        x = t[index].detach().cpu()
        if key.startswith("input.") or key.startswith("head."):
            layers[key] = {"values": [round(v, 5) for v in x.float().tolist()]}
        elif x.dim() == 2:
            layers[key] = {"tokens": [_vec_summary(row) for row in x]}
        else:
            layers[key] = _vec_summary(x)
    for i, w in enumerate(attention):
        layers[f"fusion.layer{i}.attention"] = {
            "heads": [[[round(v, 4) for v in row] for row in head] for head in w[index].detach().cpu().tolist()]
        }
    return layers
