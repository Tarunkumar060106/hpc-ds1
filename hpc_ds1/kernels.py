"""Eight real workloads (six HPC kernels, AI training, backend request handling),
written so the work can be split across ranks.

Each kernel follows the SPMD pattern MPI codes use: every rank builds its own
shard of the inputs (data is already distributed, so setup is not timed),
computes on it, and rank 0 gathers the pieces and does any final step.

    setup(size, shard, nshards, device, dtype) -> inputs     (untimed)
    run(inputs)                                -> partial    (timed, per rank)
    combine(parts, size)                       -> result     (timed, rank 0)

Inputs are generated from fixed seeds, so every configuration of a kernel
computes the same problem and results can be checked against an fp32
reference.
"""

import math

import torch

SEED = 1234


def _rows(n, shard, nshards):
    lo = n * shard // nshards
    hi = n * (shard + 1) // nshards
    return lo, hi


def _gen(seed, device="cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


class Kernel:
    name = ""
    display_name = ""
    character = ""
    sizes = []
    test_size = None  # held out entirely: never seen in training
    size_unit = ""
    tolerance = 2e-2  # max relative error vs fp32 reference for a result to count as correct

    def setup(self, size, shard, nshards, device, dtype):
        raise NotImplementedError

    def run(self, inp):
        raise NotImplementedError

    def combine(self, parts, size):
        return torch.cat(parts, dim=0)

    def work(self, size):
        """Rough operation count, used as a workload feature."""
        raise NotImplementedError


class GEMM(Kernel):
    name = "gemm"
    display_name = "Dense matrix multiply"
    character = "compute-bound, O(N³), large result to gather"
    sizes = [1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072]
    test_size = 2048
    size_unit = "N (N×N matrices)"

    def setup(self, n, shard, nshards, device, dtype):
        g = _gen(SEED)
        a = torch.randn(n, n, generator=g)
        b = torch.randn(n, n, generator=g)
        lo, hi = _rows(n, shard, nshards)
        return {"a": a[lo:hi].contiguous().to(device, dtype), "b": b.to(device, dtype)}

    def run(self, inp):
        return inp["a"] @ inp["b"]

    def work(self, n):
        return 2.0 * n ** 3


class FFT3D(Kernel):
    name = "fft"
    display_name = "3D FFT (slab decomposition)"
    character = "transpose/communication-heavy, O(N³ log N)"
    sizes = [64, 80, 96, 112, 128, 160, 192, 224, 256]
    test_size = 192
    size_unit = "N (N³ complex grid)"

    def setup(self, n, shard, nshards, device, dtype):
        g = _gen(SEED + 1)
        x = torch.complex(torch.randn(n, n, n, generator=g), torch.randn(n, n, n, generator=g))
        lo, hi = _rows(n, shard, nshards)
        cdtype = torch.complex32 if dtype in (torch.bfloat16, torch.float16) else torch.complex64
        return {"x": x[lo:hi].contiguous().to(device).to(cdtype)}

    def run(self, inp):
        return torch.fft.fft2(inp["x"], dim=(1, 2))

    def combine(self, parts, n):
        # The "transpose" step: rank 0 holds the full grid and finishes the last axis.
        x = torch.cat(parts, dim=0)
        return torch.fft.fft(x, dim=0)

    def work(self, n):
        return 5.0 * n ** 3 * math.log2(n ** 3)


class Stencil3D(Kernel):
    name = "stencil"
    display_name = "3D Jacobi stencil"
    character = "memory-bandwidth-bound, halo/ghost-zone decomposition"
    sizes = [96, 104, 112, 128, 144, 160, 176, 184, 192]
    test_size = 160
    size_unit = "N (N³ grid, 8 sweeps)"
    iters = 8

    def setup(self, n, shard, nshards, device, dtype):
        g = _gen(SEED + 2)
        u = torch.rand(n, n, n, generator=g)
        lo, hi = _rows(n, shard, nshards)
        # Ghost zone of width `iters` lets each rank sweep independently and
        # still produce exactly the global result for the rows it owns.
        glo, ghi = max(0, lo - self.iters), min(n, hi + self.iters)
        return {"u": u[glo:ghi].contiguous().to(device, dtype), "own": (lo - glo, hi - glo)}

    def run(self, inp):
        u = inp["u"]
        for _ in range(self.iters):
            v = u.clone()
            v[1:-1, 1:-1, 1:-1] = (u[:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1] +
                                   u[1:-1, 2:, 1:-1] + u[1:-1, 1:-1, :-2] + u[1:-1, 1:-1, 2:]) / 6.0
            u = v
        a, b = inp["own"]
        return u[a:b]

    def work(self, n):
        return 7.0 * n ** 3 * self.iters


class SpMV(Kernel):
    name = "spmv"
    display_name = "Sparse matrix-vector multiply"
    character = "memory-bound, irregular access, 16 nnz/row, 10 products"
    sizes = [200_000, 250_000, 300_000, 400_000, 600_000, 800_000, 1_000_000, 1_200_000, 1_600_000]
    test_size = 800_000
    size_unit = "rows"
    nnz_per_row = 16
    repeats = 10

    def setup(self, n, shard, nshards, device, dtype):
        g = _gen(SEED + 3)
        cols = torch.randint(0, n, (n, self.nnz_per_row), generator=g)
        vals = torch.randn(n, self.nnz_per_row, generator=g)
        x = torch.randn(n, generator=g)
        lo, hi = _rows(n, shard, nshards)
        rows = hi - lo
        crow = torch.arange(0, rows * self.nnz_per_row + 1, self.nnz_per_row)
        a = torch.sparse_csr_tensor(crow, cols[lo:hi].reshape(-1), vals[lo:hi].reshape(-1), size=(rows, n))
        return {"a": a.to(device).to(dtype), "x": x.to(device, dtype)}

    def run(self, inp):
        a, x = inp["a"], inp["x"]
        y = None
        for _ in range(self.repeats):
            y = a @ x
        return y

    def work(self, n):
        return 2.0 * n * self.nnz_per_row * self.repeats


class NBody(Kernel):
    name = "nbody"
    display_name = "N-body forces (all pairs)"
    character = "compute-bound, O(N²), small result to gather"
    sizes = [2048, 2304, 2560, 3072, 3584, 4096, 4608, 5120, 6144]
    test_size = 4096
    size_unit = "bodies"
    block = 256

    def setup(self, n, shard, nshards, device, dtype):
        g = _gen(SEED + 4)
        pos = torch.randn(n, 3, generator=g)
        mass = torch.rand(n, generator=g) + 0.5
        lo, hi = _rows(n, shard, nshards)
        return {"pos": pos.to(device, dtype), "mass": mass.to(device, dtype), "lo": lo, "hi": hi}

    def run(self, inp):
        pos, mass = inp["pos"], inp["mass"]
        out = []
        for s in range(inp["lo"], inp["hi"], self.block):
            p = pos[s:min(s + self.block, inp["hi"])]
            d = pos.unsqueeze(0) - p.unsqueeze(1)
            r2 = (d * d).sum(-1) + 1e-2
            w = mass / (r2 * r2.sqrt())
            out.append((d * w.unsqueeze(-1)).sum(1))
        return torch.cat(out, dim=0)

    def work(self, n):
        return 20.0 * n * n


class MonteCarlo(Kernel):
    name = "montecarlo"
    display_name = "Monte Carlo option pricing"
    character = "embarrassingly parallel, RNG-heavy, scalar result"
    sizes = [2_000_000, 2_500_000, 3_000_000, 4_000_000, 6_000_000, 8_000_000, 10_000_000, 12_000_000, 16_000_000]
    test_size = 8_000_000
    size_unit = "paths (Asian option, 16 steps)"
    steps = 16
    batch = 500_000
    tolerance = 1e-2

    def setup(self, m, shard, nshards, device, dtype):
        lo, hi = _rows(m, shard, nshards)
        g = torch.Generator(device=device)
        return {"n": hi - lo, "gen": g, "seed": SEED + 5 + shard, "device": device, "dtype": dtype}

    def run(self, inp):
        s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
        dt = t / self.steps
        drift, vol = (r - 0.5 * sigma * sigma) * dt, sigma * math.sqrt(dt)
        g = inp["gen"]
        g.manual_seed(inp["seed"])
        total = torch.zeros((), dtype=torch.float64, device=inp["device"])
        left = inp["n"]
        while left > 0:
            b = min(self.batch, left)
            z = torch.randn(b, self.steps, generator=g, device=inp["device"]).to(inp["dtype"])
            path = s0 * torch.exp(torch.cumsum(drift + vol * z, dim=1))
            payoff = torch.clamp(path.mean(dim=1) - k, min=0)
            total += payoff.double().sum()
            left -= b
        return total.reshape(1)

    def combine(self, parts, m):
        return (torch.cat(parts).sum() / m * math.exp(-0.05)).reshape(1)

    def work(self, m):
        return 30.0 * m * self.steps


class MLTrain(Kernel):
    """One data-parallel training step of a small neural network: forward,
    loss, backward. Each rank takes a slice of the batch; rank 0 sums the
    gradients (the all-reduce in data-parallel training). No optimizer update,
    so every repeat computes the same gradient and can be checked against fp32."""
    name = "mltrain"
    display_name = "Neural network training step"
    character = "AI training: matmul-heavy forward + backward, gradient all-reduce"
    sizes = [1024, 1280, 1536, 2048, 3072, 4096, 5120, 6144, 8192]
    test_size = 4096
    size_unit = "samples per step (MLP 256-1024-1024-10)"
    dims = (256, 1024, 1024, 10)
    tolerance = 3e-2  # bf16 mixed-precision gradients are expected to differ slightly

    def setup(self, n, shard, nshards, device, dtype):
        g = _gen(SEED + 6)
        ws = [(torch.randn(a, b, generator=g) / math.sqrt(a)) for a, b in zip(self.dims[:-1], self.dims[1:])]
        x = torch.randn(n, self.dims[0], generator=g)
        y = torch.randint(0, self.dims[-1], (n,), generator=g)
        lo, hi = _rows(n, shard, nshards)
        ws = [w.to(device, dtype).requires_grad_() for w in ws]
        return {"w": ws, "x": x[lo:hi].to(device, dtype), "y": y[lo:hi].to(device)}

    def run(self, inp):
        ws, h = inp["w"], inp["x"]
        for w in ws:
            w.grad = None
        for w in ws[:-1]:
            h = torch.relu(h @ w)
        loss = torch.nn.functional.cross_entropy((h @ ws[-1]).float(), inp["y"], reduction="sum")
        loss.backward()
        return torch.cat([w.grad.float().flatten() for w in ws])

    def combine(self, parts, n):
        return torch.stack(parts).sum(0) / n

    def work(self, n):
        params = sum(a * b for a, b in zip(self.dims[:-1], self.dims[1:]))
        return 6.0 * n * params  # forward ~2 flops/param/sample, backward ~4


_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
          "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango"]


def _handle_request(body):
    """What a backend endpoint does per request: parse JSON, run business logic,
    serialize the response, compress it, and sign it."""
    import hashlib
    import json
    import zlib
    req = json.loads(body)
    items = sorted(req["items"], key=lambda it: (-it["price"] * it["qty"], it["sku"]))
    total = sum(it["price"] * it["qty"] for it in items)
    words = {}
    for w in req["note"].split():
        words[w] = words.get(w, 0) + 1
    resp = json.dumps({"id": req["id"], "user": req["user"], "total": round(total, 2),
                       "top": [it["sku"] for it in items[:5]], "words": sorted(words.items())},
                      separators=(",", ":")).encode()
    digest = hashlib.sha256(zlib.compress(resp * 4, 6)).digest()
    return int.from_bytes(digest[:6], "little")  # 48 bits: exact in float64


class Server(Kernel):
    """A batch of web requests handled by a backend service. Ranks are worker
    processes (like gunicorn workers), threads are a thread pool per worker.
    Pure CPU and mostly Python, so the GIL limits threads; there is no GPU or
    bf16 version of this work, and those configs fail when measured."""
    name = "server"
    display_name = "Backend request handling"
    character = "server: JSON parse, business logic, compress, hash; Python/GIL-bound"
    sizes = [1000, 1250, 1500, 2000, 3000, 4000, 5000, 6000, 8000]
    test_size = 4000
    size_unit = "requests per batch"
    tolerance = 0.0

    def setup(self, n, shard, nshards, device, dtype):
        import json
        if str(device) != "cpu":
            raise RuntimeError("backend request handling runs on the CPU; there is no GPU version")
        if dtype != torch.float32:
            raise RuntimeError("precision does not apply to request handling")
        g = _gen(SEED + 7)
        lo, hi = _rows(n, shard, nshards)
        prices = torch.rand(n, 24, generator=g) * 100
        qty = torch.randint(1, 9, (n, 24), generator=g)
        words = torch.randint(0, len(_WORDS), (n, 120), generator=g)
        bodies = []
        for i in range(lo, hi):
            bodies.append(json.dumps({
                "id": i, "user": f"user{i % 997}",
                "items": [{"sku": f"sku{i % 50}-{j}", "price": round(float(prices[i, j]), 2), "qty": int(qty[i, j])}
                          for j in range(24)],
                "note": " ".join(_WORDS[k] for k in words[i].tolist())}).encode())
        return {"bodies": bodies}

    def run(self, inp):
        from concurrent.futures import ThreadPoolExecutor
        threads = torch.get_num_threads()
        if threads == 1:
            out = [_handle_request(b) for b in inp["bodies"]]
        else:
            with ThreadPoolExecutor(threads) as pool:
                out = list(pool.map(_handle_request, inp["bodies"], chunksize=32))
        return torch.tensor(out, dtype=torch.float64)

    def work(self, n):
        return 2e5 * n


# ---------------------------------------------------------------------------
# Real AI training workloads whose memory grows with the batch, so the larger
# sizes do not fit on small GPUs (a 6 GB laptop GPU) but do on large ones.
# Each returns a fixed slice of the gradients, so results can be checked
# against fp32 and gathered cheaply. Layer norms are per sample (LayerNorm,
# GroupNorm), so splitting the batch across ranks gives the same gradient.
# ---------------------------------------------------------------------------

def _bytes(dtype):
    return 2 if dtype in (torch.bfloat16, torch.float16, "bf16") else 4


def _grad_probe(params, n=2048):
    """A fixed sample of every parameter's gradient: enough to check correctness."""
    return torch.cat([p.grad.float().flatten()[:n] for p in params])


class LLMTrain(Kernel):
    """One training step (forward, loss, backward) of a GPT-style decoder
    language model: 8 transformer blocks, d_model 768, 12 heads, 256-token
    sequences, 8k vocabulary (~63M parameters). Memory grows with the batch:
    the activations of every layer are kept for the backward pass."""
    name = "llm_train"
    display_name = "Language model training step (GPT-style)"
    character = "AI training: attention + MLP blocks, memory grows with batch, GPU-class"
    sizes = [4, 8, 16, 24, 32, 48, 64, 96, 128]
    test_size = 24
    size_unit = "sequences of 256 tokens per step (GPT, 63M params)"
    tolerance = 6e-2  # bf16 end to end (weights and activations), checked on a gradient sample
    seq, d, layers, heads, vocab = 256, 768, 8, 12, 8192
    gpu_class = True

    def _params(self, g, device, dtype):
        d, ff = self.d, 4 * self.d

        def w(*shape, scale=None):
            t = torch.randn(*shape, generator=g) * (scale or 1.0 / math.sqrt(shape[0]))
            return t.to(device, dtype).requires_grad_()

        p = {"emb": w(self.vocab, d, scale=0.02), "pos": w(self.seq, d, scale=0.02), "blocks": []}
        for _ in range(self.layers):
            p["blocks"].append({"qkv": w(d, 3 * d), "out": w(d, d), "up": w(d, ff), "down": w(ff, d)})
        return p

    def setup(self, n, shard, nshards, device, dtype):
        g = _gen(SEED + 11)
        p = self._params(g, device, dtype)
        tokens = torch.randint(0, self.vocab, (n, self.seq + 1), generator=g)
        lo, hi = _rows(n, shard, nshards)
        return {"p": p, "x": tokens[lo:hi, :-1].to(device), "y": tokens[lo:hi, 1:].to(device), "n": n}

    def _all(self, p):
        out = [p["emb"], p["pos"]]
        for b in p["blocks"]:
            out += [b["qkv"], b["out"], b["up"], b["down"]]
        return out

    def run(self, inp):
        F = torch.nn.functional
        p = inp["p"]
        params = self._all(p)
        for t in params:
            t.grad = None
        b, s = inp["x"].shape
        d, h = self.d, self.heads
        x = p["emb"][inp["x"]] + p["pos"][:s]
        for blk in p["blocks"]:
            a = F.layer_norm(x, (d,))
            q, k, v = (a @ blk["qkv"]).split(d, dim=-1)
            q, k, v = (t.view(b, s, h, d // h).transpose(1, 2) for t in (q, k, v))
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + att.transpose(1, 2).reshape(b, s, d) @ blk["out"]
            x = x + F.gelu(F.layer_norm(x, (d,)) @ blk["up"]) @ blk["down"]
        logits = F.layer_norm(x, (d,)) @ p["emb"].t()
        loss = F.cross_entropy(logits.float().view(-1, self.vocab), inp["y"].reshape(-1), reduction="sum")
        loss.backward()
        return _grad_probe(params)

    def combine(self, parts, n):
        return torch.stack(parts).sum(0) / (n * self.seq)

    def work(self, n):
        params = self.vocab * self.d + self.layers * 12 * self.d ** 2
        tokens = n * self.seq
        return 6.0 * params * tokens + 12.0 * self.layers * self.seq * self.d * tokens

    def memory_gb(self, n, precision="fp32", nshards=1):
        """Rough peak memory of one step (weights, gradients, saved activations), used
        only before any device has measured this size."""
        bw = _bytes(precision)
        params = self.vocab * self.d + self.layers * 12 * self.d ** 2 + self.seq * self.d
        tokens = n * self.seq / nshards
        act = tokens * (self.layers * 20 * self.d + 3 * self.vocab) * bw
        return (2 * params * bw * nshards + act * nshards) / 2 ** 30


class CNNTrain(Kernel):
    """One training step of a ResNet-18-style image classifier on 112×112 images
    (~11M parameters, GroupNorm so batch shards are independent). Memory grows
    with the batch: every feature map is kept for the backward pass."""
    name = "cnn_train"
    display_name = "Image model training step (ResNet-18-style)"
    character = "AI training: convolutions, memory grows with batch, GPU-class"
    sizes = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048]
    test_size = 384
    size_unit = "images per step (112×112, ResNet-18-style)"
    tolerance = 6e-2
    widths, image, classes = (64, 128, 256, 512), 112, 100
    gpu_class = True

    def setup(self, n, shard, nshards, device, dtype):
        g = _gen(SEED + 12)

        def conv(cin, cout, k):
            t = torch.randn(cout, cin, k, k, generator=g) * math.sqrt(2.0 / (cin * k * k))
            return t.to(device, dtype).requires_grad_()

        p = {"stem": conv(3, self.widths[0], 7), "stages": []}
        cin = self.widths[0]
        for i, c in enumerate(self.widths):
            blocks = []
            for j in range(2):
                stride = 2 if (j == 0 and i > 0) else 1
                blk = {"c1": conv(cin, c, 3), "c2": conv(c, c, 3), "stride": stride}
                if stride != 1 or cin != c:
                    blk["proj"] = conv(cin, c, 1)
                blocks.append(blk)
                cin = c
            p["stages"].append(blocks)
        p["fc"] = (torch.randn(cin, self.classes, generator=g) / math.sqrt(cin)).to(device, dtype).requires_grad_()
        x = torch.randn(n, 3, self.image, self.image, generator=g)
        y = torch.randint(0, self.classes, (n,), generator=g)
        lo, hi = _rows(n, shard, nshards)
        return {"p": p, "x": x[lo:hi].to(device, dtype), "y": y[lo:hi].to(device)}

    def _all(self, p):
        out = [p["stem"]]
        for blocks in p["stages"]:
            for b in blocks:
                out += [b["c1"], b["c2"]] + ([b["proj"]] if "proj" in b else [])
        return out + [p["fc"]]

    def run(self, inp):
        F = torch.nn.functional
        p = inp["p"]
        params = self._all(p)
        for t in params:
            t.grad = None

        def norm(t):
            return F.group_norm(t, 8)

        h = F.max_pool2d(F.relu(norm(F.conv2d(inp["x"], p["stem"], stride=2, padding=3))), 3, 2, 1)
        for blocks in p["stages"]:
            for b in blocks:
                out = F.relu(norm(F.conv2d(h, b["c1"], stride=b["stride"], padding=1)))
                out = norm(F.conv2d(out, b["c2"], padding=1))
                skip = norm(F.conv2d(h, b["proj"], stride=b["stride"])) if "proj" in b else h
                h = F.relu(out + skip)
        logits = h.mean((2, 3)) @ p["fc"]
        loss = F.cross_entropy(logits.float(), inp["y"], reduction="sum")
        loss.backward()
        return _grad_probe(params)

    def combine(self, parts, n):
        return torch.stack(parts).sum(0) / n

    def work(self, n):
        return 3 * 0.46e9 * n  # ~0.46 GFLOP forward per 112×112 image (ResNet-18 at a quarter of 224² pixels)

    def memory_gb(self, n, precision="fp32", nshards=1):
        bw = _bytes(precision)
        per_image = 1.6e6  # saved activation values per 112×112 image across the network (convs + norms + relus)
        return (2 * 11.2e6 * bw * nshards + n * per_image * bw) / 2 ** 30


KERNELS = {k.name: k for k in (GEMM(), FFT3D(), Stencil3D(), SpMV(), NBody(), MonteCarlo(), MLTrain(), Server(),
                               LLMTrain(), CNNTrain())}


# ---------------------------------------------------------------------------
# Bring your own task. A facility adds its own workload by writing a Kernel
# subclass (setup / run / combine / work, plus sizes and test_size) and
# decorating it with @register, in a .py file placed in ~/.hpc_ds1/plugins/
# (or $HPC_DS1_HOME/plugins/), or in a module named in $HPC_DS1_PLUGINS
# ("pkg.module,other.module"). `hpc-ds1 setup` then measures it on each
# machine and retrains; until then the familiarity guard escalates it.
# Rank processes import this module too, so they see the same plugins.
# ---------------------------------------------------------------------------

def register(cls):
    """Class decorator that adds a Kernel subclass to the workloads HPC-DS1 knows."""
    k = cls()
    if not k.name or not k.sizes or k.test_size not in k.sizes:
        raise ValueError(f"{cls.__name__}: set name, sizes and a test_size taken from sizes")
    if k.name in KERNELS and type(KERNELS[k.name]) is not cls:
        raise ValueError(f"workload name {k.name!r} is already taken")
    KERNELS[k.name] = k
    return cls


def _load_plugins():
    import importlib
    import importlib.util
    import os
    from pathlib import Path
    loaded = []
    for mod in filter(None, (m.strip() for m in os.environ.get("HPC_DS1_PLUGINS", "").split(","))):
        importlib.import_module(mod)
        loaded.append(mod)
    folder = Path(os.environ.get("HPC_DS1_HOME", Path.home() / ".hpc_ds1")) / "plugins"
    for f in sorted(folder.glob("*.py")) if folder.is_dir() else []:
        spec = importlib.util.spec_from_file_location(f"hpc_ds1_plugin_{f.stem}", f)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        loaded.append(str(f))
    return loaded


PLUGINS = _load_plugins()
WORKLOAD_NAMES = list(KERNELS)
