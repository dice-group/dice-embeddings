"""NCCL sanity check: one all_reduce across the two containers (ignores the dicee CLI flags)."""
import os

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(0)
x = torch.full((4,), float(rank + 1), device="cuda")
dist.all_reduce(x)
expected = sum(range(1, world + 1))
print(f"rank {rank}/{world} host={os.uname().nodename} all_reduce -> {x.tolist()} expected {expected}", flush=True)
assert torch.allclose(x, torch.full_like(x, expected))
dist.destroy_process_group()
