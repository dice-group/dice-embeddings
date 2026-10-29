# Multi-node DDP simulation on one GPU

Runs dicee as **two nodes with one rank each** in two Docker containers that share a single GPU, so distributed code paths (real NCCL, `torch.compile`, rank handling) can be exercised without a multi-GPU machine.

## What it can and cannot test

- Can: `torchDDP` and `PL` with 2 ranks over NCCL, rank-dependent callback logic (for example ASWA), rendezvous and shared-output handling.
- Cannot: 2 GPUs inside one node (`PL` with `devices=2`, `torchrun --nproc_per_node=2`), the `TP` trainer, performance or memory behaviour. Both ranks share one GPU.

## Requirements

Docker with `nvidia-container-toolkit` (`docker run --rm --gpus all ubuntu nvidia-smi` must list a GPU), one NVIDIA GPU, and `KGs/UMLS` in the repo root (see `CLAUDE.md` for the dataset download). The image installs torch 2.10.0 (cu128) and dicee's requirements, and the repo is bind-mounted at `/dicee`, so local changes are picked up without rebuilding.

## Usage

Run from this directory:

```bash
# NCCL sanity check only (one all_reduce across the two containers)
SCRIPT=tools/multinode_sim/smoke_allreduce.py docker compose up --build --abort-on-container-exit

# torchDDP + adaptive SWA
docker compose up --abort-on-container-exit

# PL + adaptive SWA, forced into real 2-rank DDP
TRAINER=PL SCRIPT=tools/multinode_sim/run_pl_ddp.py docker compose up --abort-on-container-exit

docker compose down
```

Environment variables read by `docker-compose.yml`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `TRAINER` | `torchDDP` | value of `--trainer` |
| `SCRIPT` | `dicee/scripts/run.py` | entry script, relative to the repo root |
| `EVAL` | `train_val_test` | value of `--eval_model` (`adaptive_swa` needs validation data) |
| `EXTRA` | `--adaptive_swa` | extra dicee flags |

## Notes

- Both ranks must write to the same `--path_to_store_single_run` (`out/shared`): rank 0 preprocesses the data and the other rank loads it, as on a shared filesystem.
- `PL` needs `run_pl_ddp.py`: dicee builds `lightning.Trainer` with `num_nodes=1` and `devices="auto"`, which resolves to a single device per container and therefore a non-distributed run. The wrapper forces `num_nodes=2, strategy="ddp"` and prints `RIG strategy=DDPStrategy world_size=2` per rank. Check for that line; an exit code of 0 alone does not show that the run was distributed.
- NCCL is set to the socket transport (`NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE`, `NCCL_IB_DISABLE`) because two ranks cannot share one GPU through P2P or shared memory.
- Containers run as root, so `out/` ends up root-owned; remove it with `sudo`.
