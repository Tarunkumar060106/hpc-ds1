"""Runs a kernel for real on this machine and measures it.

MPI-style execution without an MPI install: rank 0 is the calling process,
ranks 1..P-1 are persistent spawned processes with their own memory, each
connected to rank 0 by a pipe. Every rank computes its shard with T threads;
rank 0 then gathers the partial results over the pipes and combines them.
Everything reported (runtime, CPU time, GPU busy time, communication time,
memory) is measured, not modelled.
"""

import multiprocessing as mp
import os
import platform
import statistics
import time

import psutil
import torch

from .kernels import KERNELS

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _to_wire(t):
    t = t.detach()
    if t.dtype == torch.complex32:
        t = t.to(torch.complex64)
    elif t.dtype in (torch.bfloat16, torch.float16):
        t = t.float()
    return t.cpu().numpy()


def _compute(cache, workload, size, shard, nshards, threads, precision, device):
    torch.set_num_threads(threads)
    key = (workload, size, shard, nshards, precision, device)
    kern = KERNELS[workload]
    if key not in cache:
        cache.clear()  # keep at most one problem resident per rank
        cache[key] = kern.setup(size, shard, nshards, device, _DTYPES[precision])
    t0 = time.perf_counter()
    out = kern.run(cache[key])
    return out, time.perf_counter() - t0


def _rank_main(conn):
    torch.set_num_interop_threads(1)
    cache = {}
    while True:
        msg = conn.recv()
        if msg is None:
            break
        seq, args = msg
        try:
            out, dt = _compute(cache, *args)
            conn.send((seq, "ok", _to_wire(out), dt))
        except Exception as e:  # report, keep the rank alive
            conn.send((seq, "err", f"{type(e).__name__}: {e}", 0.0))


def _recv(conn, proc, poll_sec=1.0):
    """A rank's reply; raises instead of waiting forever if the rank process has died
    (e.g. Windows could not load torch in it because memory ran out)."""
    while not conn.poll(poll_sec):
        if not proc.is_alive():
            raise RuntimeError(f"rank process {proc.pid} died (exit code {proc.exitcode}); "
                               "the machine may be out of memory")
    return conn.recv()


class RankGroup:
    """P-1 persistent worker ranks (rank 0 is the caller)."""

    def __init__(self, nranks):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(nranks - 1):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_rank_main, args=(child,), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)

    def pids(self):
        return [p.pid for p in self.procs]

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def detect_hardware():
    vm = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    hw = {
        "name": platform.node() or "this-machine",
        "cpu_model": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False) or os.cpu_count(),
        "cpu_threads": psutil.cpu_count(logical=True) or os.cpu_count(),
        "cpu_freq_ghz": round((freq.max or freq.current) / 1000, 2) if freq else 0.0,
        "ram_gb": round(vm.total / 2 ** 30, 1),
        "gpu_model": None,
        "gpu_memory_gb": 0.0,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        hw["gpu_model"] = props.name
        hw["gpu_memory_gb"] = round(props.total_memory / 2 ** 30, 1)
    return hw


class Engine:
    """Measures configurations. Keeps rank groups alive between runs."""

    def __init__(self):
        self.groups = {}
        self.local_cache = {}
        self.hardware = detect_hardware()
        self.proc = psutil.Process()
        self._seq = 0

    def _group(self, nranks):
        if nranks not in self.groups:
            self.groups[nranks] = RankGroup(nranks)
        return self.groups[nranks]

    def close(self):
        for g in self.groups.values():
            g.close()
        self.groups.clear()

    def _cpu_seconds(self, pids):
        total = sum(self.proc.cpu_times()[:2])
        for pid in pids:
            try:
                total += sum(psutil.Process(pid).cpu_times()[:2])
            except psutil.Error:
                pass
        return total

    def _rss_gb(self, pids):
        total = self.proc.memory_info().rss
        for pid in pids:
            try:
                total += psutil.Process(pid).memory_info().rss
            except psutil.Error:
                pass
        return total / 2 ** 30

    def _once(self, workload, size, cfg, group):
        kern = KERNELS[workload]
        P, T, prec = cfg["ranks"], cfg["threads"], cfg["precision"]
        if cfg["gpu"]:
            dev = "cuda"
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out, compute = _compute(self.local_cache, workload, size, 0, 1, T, prec, dev)
            torch.cuda.synchronize()
            compute = time.perf_counter() - t0
            res = kern.combine([out], size)
            res = _to_wire(res)
            wall = time.perf_counter() - t0
            return wall, compute, res
        self._seq += 1
        seq = self._seq
        conns = group.conns if group else []
        t0 = time.perf_counter()
        for r, conn in enumerate(conns, start=1):
            conn.send((seq, (workload, size, r, P, T, prec, "cpu")))
        err = None
        try:
            out0, c0 = _compute(self.local_cache, workload, size, 0, P, T, prec, "cpu")
            parts, computes = [torch.from_numpy(_to_wire(out0))], [c0]
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        # Always collect every rank's reply, even after a failure, so no stale
        # message is ever read as the answer to a later configuration.
        for conn, proc in zip(conns, group.procs if group else []):
            rseq, status, payload, dt = _recv(conn, proc)
            if rseq != seq:
                raise RuntimeError(f"rank reply out of sequence ({rseq} != {seq})")
            if status != "ok":
                err = err or payload
            elif err is None:
                parts.append(torch.from_numpy(payload))
                computes.append(dt)
        if err:
            raise RuntimeError(err)
        res = _to_wire(kern.combine(parts, size))
        wall = time.perf_counter() - t0
        return wall, max(computes), res

    def measure(self, workload, size, cfg, min_total=0.3, max_reps=30, slow_cap=5.0):
        """Run a configuration for real. Returns telemetry + the result array.

        One warm-up run (builds inputs, compiles plans), then enough repeats
        to cover `min_total` seconds (3-30) so very short runs aren't timer
        noise. A config whose warm-up already exceeds `slow_cap` seconds is
        recorded from that single run instead of being repeated.
        """
        cfg = normalize_config(cfg)
        kern = KERNELS[workload]
        if hasattr(kern, "memory_gb"):
            # memory-heavy workloads: keep only the worker group this run needs, so the RAM
            # check below sees memory that other groups' cached problems would otherwise hold
            keep = None if cfg["gpu"] else cfg["ranks"]
            for n in [n for n in self.groups if n != keep]:
                self.groups.pop(n).close()
            if cfg["gpu"] is False:
                self.local_cache.clear()
        if not cfg["gpu"] and hasattr(kern, "memory_gb"):
            # protect the machine: a CPU run that cannot fit in free RAM would swap or crash it
            need = kern.memory_gb(size, cfg["precision"], cfg["ranks"])
            free = psutil.virtual_memory().available / 2 ** 30
            if need > 0.8 * free:
                return {"ok": False, "skipped": True, "oom": True,
                        "error": f"skipped: needs ~{need:.1f} GB RAM, {free:.1f} GB free"}
        group = None if cfg["gpu"] or cfg["ranks"] == 1 else self._group(cfg["ranks"])
        pids = group.pids() if group else []
        if cfg["gpu"]:
            torch.cuda.reset_peak_memory_stats()
        try:
            if cfg["gpu"]:
                # Laptop GPUs idle at a low clock; spin for >=0.3s so we time
                # the boosted steady state, not the ramp-up.
                t_warm = time.perf_counter()
                while time.perf_counter() - t_warm < 0.3:
                    self._once(workload, size, cfg, group)
            cpu0 = self._cpu_seconds(pids)
            w0, c0, res = self._once(workload, size, cfg, group)
            if w0 > slow_cap:
                walls, computes, capped = [w0], [c0], True
            else:
                reps = max(3, min(max_reps, int(min_total / max(w0, 1e-6)) + 1))
                walls, computes, capped = [], [], False
                for _ in range(reps):
                    w, c, res = self._once(workload, size, cfg, group)
                    walls.append(w)
                    computes.append(c)
            cpu1 = self._cpu_seconds(pids)
            busy = w0 + (0.0 if capped else sum(walls))
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:200]}"
            if "rank process" in msg and "died" in msg and cfg["ranks"] in self.groups:
                self.groups.pop(cfg["ranks"]).close()  # start a fresh group next time
            oom = "out of memory" in msg.lower() or "outofmemory" in msg.lower() or "not enough memory" in msg.lower()
            if oom:
                self.local_cache.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return {"ok": False, "error": msg, "oom": oom}
        finally:
            torch.set_num_threads(self.hardware["cpu_threads"])

        wall = statistics.median(walls)
        compute = statistics.median(computes)
        tel = {
            "runtime_sec": round(wall, 6),
            "cpu_util_pct": round(min(100.0, 100.0 * (cpu1 - cpu0) / (busy * self.hardware["cpu_threads"])), 2),
            "gpu_busy_pct": round(100.0 * compute / wall, 2) if cfg["gpu"] else 0.0,
            "comm_overhead_pct": round(100.0 * max(0.0, wall - compute) / wall, 2),
            "memory_gb": round(self._rss_gb(pids) + (torch.cuda.max_memory_allocated() / 2 ** 30 if cfg["gpu"] else 0), 3),
            "noise_pct": round(100.0 * (max(walls) - min(walls)) / wall, 2),
        }
        if cfg["gpu"]:
            # real peak GPU memory of the job; above the card's memory the driver spills to system RAM
            tel["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        return {"ok": True, "telemetry": tel, "walls": [round(w, 6) for w in walls], "capped": capped,
                "result": res}


def normalize_config(cfg):
    gpu = bool(cfg.get("gpu", False))
    return {
        "ranks": 1 if gpu else int(cfg.get("ranks", 1)),
        "threads": int(cfg.get("threads", 1)),
        "gpu": gpu,
        "precision": cfg.get("precision", "fp32"),
    }
