"""Example of bringing your own task to HPC-DS1: an ETL batch job.

Install it for every HPC-DS1 process on this machine:

    copy examples\\plugins\\etl_batch.py %USERPROFILE%\\.hpc_ds1\\plugins\\     (Windows)
    cp examples/plugins/etl_batch.py ~/.hpc_ds1/plugins/                        (Linux/macOS)

    hpc-ds1 tasks                       # 'etl' is now listed
    hpc-ds1 setup --workloads etl       # measure it here, then retrain on everything measured

Until the machine has measured it, the familiarity guard escalates every
decision about it ("never trained on workload 'etl'").

A task is split the way MPI codes split work: setup() builds this rank's
shard (untimed), run() computes on it (timed), combine() merges the shards on
rank 0. Results must match across any number of shards, because every run is
checked against the single-rank fp32 result.
"""

import torch

from hpc_ds1.kernels import Kernel, _gen, _rows, register


@register
class ETLBatch(Kernel):
    name = "etl"
    display_name = "ETL batch: clean and aggregate"
    character = "data pipeline: clip outliers, group-by sum and count"
    sizes = [500_000, 1_000_000, 2_000_000, 4_000_000]
    test_size = 2_000_000
    size_unit = "records"
    groups = 4096

    def setup(self, n, shard, nshards, device, dtype):
        g = _gen(4242)
        keys = torch.randint(0, self.groups, (n,), generator=g)
        vals = torch.randn(n, generator=g) * 3
        lo, hi = _rows(n, shard, nshards)
        return {"k": keys[lo:hi].to(device), "v": vals[lo:hi].to(device, dtype)}

    def run(self, inp):
        v = inp["v"].clamp(-5, 5)                         # clean
        out = torch.zeros(2, self.groups, dtype=torch.float32, device=v.device)
        out[0].index_add_(0, inp["k"], v.float())          # sum per group
        out[1].index_add_(0, inp["k"], torch.ones_like(v, dtype=torch.float32))  # count per group
        return out

    def combine(self, parts, n):
        return torch.stack(parts).sum(0)

    def work(self, n):
        return 4.0 * n
