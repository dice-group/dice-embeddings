# torchFSDP sharded checkpoint save/resume — code changes + test report

Branch: `feat/fsdp-sharded-checkpoint-resume` (already pushed to origin)
Related: [issue #422](https://github.com/dice-group/dice-embeddings/issues/422), [issue-422-analysis.md](issue-422-analysis.md)

## 1. What changed (code)

**Problem.** Under `torchFSDP`, there was no way to recover a training run that
crashed or was preempted mid-training: periodic callbacks only had access to
rank 0's local entity shard (not the full table), and `--continual_learning`
crashed outright because `select_model()` tried to load a full `model.pt` into
a model shell whose sharded entity embeddings don't exist yet (they're created
later by `setup_fsdp_training()`). This is exactly what happened in issue
#422's Wikidata run: OOM-killed ~1.4% into epoch 1, with no way to resume.

**Fix**, committed on `feat/fsdp-sharded-checkpoint-resume`:

| File | Change |
|---|---|
| `dicee/models/fsdp_models.py` | `_LocalSparseAdam` gained `state_dict()`/`load_state_dict()`. `FSDPShardedEntityModel` gained `save_local_shard_checkpoint()`/`load_local_shard_checkpoint()` — each rank saves/loads only its own shard (O(local shard), not O(num_entities)) plus a world_size/shard-boundary check that raises instead of silently loading the wrong rows. |
| `dicee/trainer/torch_trainer_fsdp.py` | New `checkpoint_every_n_epochs` (via `--fsdp_trainer_kwargs`). Writes a crash-safe checkpoint (temp-file-then-rename, `meta.json` committed last) every N epochs. On startup, auto-resumes from the same experiment folder or from `--continual_learning`'s folder, refusing a world_size mismatch loudly. |
| `dicee/static_funcs.py` / `dicee/executer.py` | `select_model()` no longer crashes for FSDP continual learning; `ContinuousExecute` no longer requires `report.json` (only written on completion) to recover `num_entities`/`num_relations` — falls back to counting `entity_to_idx.csv`/`relation_to_idx.csv`, so resuming a crashed run is reachable via the CLI. |
| `README.md` | Documented `checkpoint_every_n_epochs`, the resume flow, and the `--reuse_existing_run_dir` requirement (see §3.3). |
| `tests/test_fsdp_shard_checkpoint.py` | 4 non-distributed unit tests: save/load round trip, optimizer state round trip, world_size-mismatch rejection, pre-setup guard. |

**Bug found and fixed during this test session** (not yet pushed — see §5):
`_save_sharded_checkpoint()` reused the existing `_gather_full_state_dict()`
helper to save dense params. But `entity_embeddings` is attached to `raw_model`
*after* FSDP wraps it, so a plain `state_dict()` walk finds it sitting there
and includes `entity_embeddings.weight` / `_fsdp_adapter.weight` in what was
supposed to be the "dense-only" checkpoint. `_materialize_model()` (pre-existing
code) already knew to strip these keys; my new `_save_sharded_checkpoint()`
didn't, and would crash on load with `RuntimeError: Unexpected key(s)... entity_embeddings.weight`.
Fixed by extracting the exclusion logic into a shared `_strip_entity_keys()`
static method used by both code paths.

## 2. The Docker question — why we didn't use containers

Original ask: simulate 2-node distributed training with 2 Docker containers.
Investigation found two blockers, in order of how fundamental they are:

1. **`docker info` shows no `nvidia` runtime** — `nvidia-container-toolkit`
   isn't installed on this machine, so containers can't reach the GPU at all
   right now. Fixable (needs sudo), but turned out not to matter — see #2.
2. **NCCL categorically refuses two ranks on the same physical GPU** —
   confirmed empirically (§4.1). This machine has exactly one GPU
   (`RTX 2000 Ada`, 8GB). Docker or no Docker, containers or bare processes,
   **two ranks cannot share one GPU under NCCL.** Installing
   `nvidia-container-toolkit` would not have unblocked a real 2-rank test —
   it would only have gotten us to the *same* "Duplicate GPU detected" error,
   just inside containers instead of bare processes.

**Decision:** skip Docker's extra layer entirely. Use plain `torchrun`
processes on the host to exercise the distributed code paths (rendezvous,
row-wise sharding, checkpoint save/resume), and accept that genuine multi-rank
(world_size ≥ 2) validation needs a machine with ≥2 physical GPUs — that's a
hardware requirement, not a config gap. If that hardware becomes available,
the exact same commands in §3 generalize directly to `--nnodes=2` with real
separate GPUs (just drop the `CUDA_VISIBLE_DEVICES=0` pin on both, or set it
per-node to a distinct index).

## 3. Environment setup details (so this doesn't need re-discovering)

### 3.1 The `dice` conda env had a CPU-only torch

`dicee`'s documented install (`CLAUDE.md`, `requirements.txt`) pins
`--extra-index-url https://download.pytorch.org/whl/cpu`. The env this repo
is developed in (`dice`) had `torch==2.13.0+cpu` — `torch.cuda.is_available()`
was `False` even though the machine has a real GPU (`nvidia-smi` works fine).
`torchFSDP`/`torchDDP` hardcode `backend="nccl"` in
`setup_distributed_training()` (`dicee/static_funcs.py`) with no CPU/gloo
fallback, so neither trainer can even initialize its process group without a
CUDA-enabled torch.

**Fix applied (in place, same env):**
```bash
pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu130
```
Chosen to match the driver (`nvidia-smi` reports `CUDA Version: 13.0`) and to
match a torch build already known to work on this exact machine in another
conda env (`tentris_llm`). Result: `torch 2.11.0+cu130`, `cuda available: True`,
`nccl available: True`. `dicee` still imports fine afterward (torch is its only
GPU-related dependency).

**If this needs redoing** (e.g. env got reset): re-run the pip install above
in whichever env `dicee` is installed in. Don't use the plain
`requirements.txt` CPU install for GPU/FSDP work.

### 3.2 Only one physical GPU, 8GB VRAM

`nvidia-smi`: one `NVIDIA RTX 2000 Ada Generation Laptop GPU`, 8188 MiB total.
Kept test configs small (`--embedding_dim 32 --batch_size 128` on UMLS,
135 entities / 46 relations / 5216 train triples) — this is a correctness
smoke test of the checkpoint mechanism, not a performance benchmark.

### 3.3 `--reuse_existing_run_dir` is required for same-directory auto-resume

`Execute._setup_single_run_directory()` (`dicee/executer.py`) **deletes
`path_to_store_single_run` if it already exists**, unless
`--reuse_existing_run_dir` is passed. Without it, relaunching the same command
after a crash wipes the checkpoint before `TorchFSDPTrainer.fit()` ever gets a
chance to look for it. This is now called out explicitly in `README.md`; it's
easy to forget because the failure mode (silent fresh start, no error) doesn't
announce itself.

### 3.4 `--reuse_existing_run_dir` and `--eval_model None` are flag/value quirks

`--reuse_existing_run_dir` is `action="store_true"` in `dicee/scripts/run.py`
— pass it bare, **not** `--reuse_existing_run_dir true` (the latter makes
argparse choke on the stray `true` token as an unrecognized argument).
`--eval_model None` (the literal string) is correct and intentional — it's how
you skip evaluation entirely, per the config table in `CLAUDE.md`.

## 4. What was actually tested, and results

Test env: `dice` conda env (now with `torch==2.11.0+cu130`), single GPU,
dataset `KGs/UMLS`, model `Keci`, `--scoring_technique NegSample --neg_ratio 2`,
`--embedding_dim 32 --batch_size 128`.

### 4.1 2-rank NCCL simulation (world_size=2, both ranks on the one GPU) — **blocked, as expected**

```bash
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --rdzv_backend=c10d --rdzv_endpoint=localhost:29501 ...
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29501 ...
```
Both processes reached rendezvous successfully, then failed identically at
`dist.init_process_group`:
```
ncclInvalidUsage: This usually reflects invalid usage of NCCL library.
Last error: Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1000
```
This is the empirical basis for §2's decision — confirms the limitation is
NCCL itself, not launch configuration.

### 4.2 Single-GPU (world_size=1) full run with periodic checkpointing — **passed**

6 epochs, `checkpoint_every_n_epochs=2`. Completed normally; `fsdp_shard_checkpoint/`
was written and left in place (not deleted after success), `model.pt` +
`report.json` produced correctly, `meta.json` final epoch = 6 matching
`loss_history` length.

### 4.3 Simulated crash + resume — **passed, after one fix**

Procedure (single bash script, no manual timing guesswork — polls
`fsdp_shard_checkpoint/meta.json`'s `epoch` field on disk rather than log
output, since checkpoint `logger.info(...)` calls don't reach stdout under
this project's default logging config):

1. Launch `--num_epochs 200 --fsdp_trainer_kwargs '{"checkpoint_every_n_epochs": 1}'`
   in the background.
2. Poll until `meta.json` reports `epoch >= 5`, then `SIGKILL` the process
   (simulates an OOM/node-failure/preemption crash).
3. Confirmed: `report.json` and `model.pt` both **absent** (genuinely
   incomplete run), `meta.json` shows `epoch: 5`, `loss_history` has exactly
   5 entries.
4. First attempt at this step hit the entity-key-leak bug described in §1 —
   fixed, then retried.
5. Relaunch the **identical** command (same `--path_to_store_single_run`,
   `--reuse_existing_run_dir`). Confirmed:
   - Run completes successfully (exit code 0), produces `model.pt` + `report.json`.
   - **The resumed run's log contains `Epoch:6` through `Epoch:200` — never
     `Epoch:1` through `Epoch:5`.** This is the key proof: with a fixed
     `--random_seed`, a silent restart-from-scratch would reach the same final
     loss values as a genuine resume (identical seed ⇒ identical trajectory),
     so comparing final numbers alone can't distinguish "resumed" from
     "quietly restarted." The epoch-label sequence can, and does.
   - Final `meta.json`: `epoch: 200`, `loss_history` length 200, and its
     first 5 entries are byte-identical to the interrupted run's saved
     `loss_history` — the carried-over history is exactly what was recorded
     before the crash, not recomputed.

### 4.4 Post-test regression check — **passed**

`pytest tests/test_fsdp_shard_checkpoint.py tests/test_unit_base_model.py
tests/test_custom_trainer.py` (47 tests) and `ruff check` on the touched
files both pass after the §1 fix.

## 5. New finding: FSDP models can't currently be loaded for inference — unrelated pre-existing bug

While sanity-checking the final resumed `model.pt` via `KGE(path=...)`
(`dicee/knowledge_graph_embeddings.py`), loading failed:
```
RuntimeError: Error(s) in loading state_dict for FSDPKeci:
    Unexpected key(s) in state_dict: "entity_embeddings.weight".
```
Root cause: `load_model()` (`dicee/static_funcs.py`) calls `intialize_model(configs)`
using `configuration.json`, which still says `"trainer": "torchFSDP"` from the
original training run. `intialize_model()` doesn't distinguish "building a
shell for FSDP *training*" (needs the sharded, entity-less shell) from
"loading a *completed* `model.pt` for inference" (needs the plain class,
since a materialized `model.pt` already has a normal, full `entity_embeddings.weight`).
So it always rebuilds the sharded shell, which has no `entity_embeddings`
submodule to load that key into — this happens for **every** completed
`torchFSDP` run, not just ones that used the new checkpoint feature. It
predates this branch entirely; it just hadn't been exercised end-to-end
before.

**Not fixed in this session** — it's outside the "sharded checkpoint
save/resume" scope this branch targets, and is a decision point on its own
(likely fix: an explicit `for_inference` flag threaded through
`intialize_model()`/`load_model()`/`load_model_ensemble()` that skips the
FSDP-sharding branch when loading a finished model). Flagging it here so it
isn't lost.
