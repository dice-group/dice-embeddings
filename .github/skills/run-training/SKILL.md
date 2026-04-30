---
name: run-training
description: "Configure and launch KGE model training in dicee. Use when: training a KGE model, choosing a trainer backend, selecting a scoring technique, setting up multi-GPU or DDP training, using weight averaging (SWA/EMA/SWAG), configuring periodic evaluation, resuming training with continual learning."
argument-hint: "Describe your setup: dataset, hardware (CPU/GPU count), scoring preference, any special requirements"
---

# Configure and Launch KGE Training

## When to Use
- Training a KGE model for the first time
- Choosing the right trainer (CPU, single GPU, multi-GPU)
- Selecting a scoring technique for your KG size
- Setting up continual/incremental learning
- Enabling weight averaging (SWA, EMA, SWAG, etc.)
- Configuring periodic evaluation and checkpointing

---

## 1. Choose Your Trainer

| Scenario | `--trainer` | Command |
|----------|-------------|---------|
| CPU or single GPU | `torchCPUTrainer` | `dicee --trainer torchCPUTrainer ...` |
| Multi-GPU (recommended) | `PL` | `dicee --trainer PL ...` |
| Native DDP (multi-GPU) | `torchDDP` | `torchrun --standalone --nnodes=1 --nproc_per_node=gpu dicee --trainer torchDDP ...` |
| Multi-node DDP | `torchDDP` or `PL` | `torchrun --nnodes 2 --nproc_per_node=gpu ...` |
| Tensor Parallelism | `TP` | `dicee --trainer TP ...` |

**Rules:**
- `PL` trainer uses all visible CUDA devices automatically — restrict with `CUDA_VISIBLE_DEVICES=0`
- `torchDDP` and multi-GPU runs **require** `--path_to_store_single_run` to prevent write conflicts
- Pass extra Lightning options via `--pl_trainer_kwargs '{"precision":"16-mixed","strategy":"ddp"}'`

---

## 2. Choose a Scoring Technique

| KG Size | Recommended | `--scoring_technique` | Notes |
|---------|-------------|----------------------|-------|
| Very large (>1M triples) | NegSample | `NegSample` | Set `--neg_ratio 10` |
| Large | NegSample/KvsSample | `KvsSample` | Balanced memory/accuracy |
| Medium | KvsAll | `KvsAll` | Default, best quality |
| Small | AllvsAll or KvsAll | `AllvsAll` | Full pairwise; very slow |
| Continual learning | FixedNegSample | `FixedNegSample` | Stable negative samples |

For `NegSample` and `FixedNegSample`, always set `--neg_ratio` (e.g., `--neg_ratio 10`).
Increase `--num_core` (e.g., `--num_core 10`) to parallelize data loading for large KGs.

---

## 3. Training Examples

### CPU training
```bash
dicee --dataset_dir "KGs/UMLS" \
      --model Keci \
      --trainer torchCPUTrainer \
      --scoring_technique KvsAll \
      --embedding_dim 64 \
      --num_epochs 100 \
      --lr 0.1 \
      --eval_model train_val_test
```

### Single GPU
```bash
CUDA_VISIBLE_DEVICES=0 dicee \
      --dataset_dir "KGs/UMLS" \
      --model Keci \
      --trainer PL \
      --scoring_technique KvsAll \
      --num_epochs 100 \
      --eval_model train_val_test
```

### Multi-GPU (PL)
```bash
dicee --dataset_dir "KGs/YAGO3-10" \
      --model Keci \
      --trainer PL \
      --scoring_technique KvsAll \
      --path_to_store_single_run "YAGO3_run" \
      --num_epochs 200
```

### Multi-GPU (torchDDP)
```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=gpu \
      dicee --dataset_dir "KGs/UMLS" \
            --model Keci \
            --trainer torchDDP \
            --scoring_technique KvsAll \
            --path_to_store_single_run "UMLS_DDP"
```

### Multi-node (2 nodes)
```bash
# Node 0:
torchrun --nnodes 2 --nproc_per_node=gpu --node_rank 0 \
  --rdzv_id 455 --rdzv_backend c10d --rdzv_endpoint=<host> \
  dicee --trainer PL --dataset_dir "KGs/YAGO3-10" --path_to_store_single_run "YAGO3_PL"

# Node 1:
torchrun --nnodes 2 --nproc_per_node=gpu --node_rank 1 \
  --rdzv_id 455 --rdzv_backend c10d --rdzv_endpoint=<host> \
  dicee --trainer PL --dataset_dir "KGs/YAGO3-10" --path_to_store_single_run "YAGO3_PL"
```

### Python API
```python
from dicee.executer import Execute
from dicee.config import Namespace

args = Namespace()
args.model = 'Keci'
args.dataset_dir = 'KGs/UMLS'
args.scoring_technique = 'KvsAll'
args.num_epochs = 100
args.embedding_dim = 64
args.lr = 0.1
args.eval_model = 'train_val_test'
args.path_to_store_single_run = 'MyRun'
result = Execute(args).start()
print(result['Test']['MRR'])
```

---

## 4. Continual Learning

Resume training from a previous run using `--continual_learning <path>`:

```bash
# Initial run (0 epochs — just preprocess and store artifacts)
dicee --dataset_dir "KGs/UMLS" --model Keci \
      --scoring_technique FixedNegSample \
      --path_to_store_single_run "UMLS_CL" \
      --num_epochs 0

# Resume for 50 more epochs
dicee --continual_learning "UMLS_CL" \
      --model Keci \
      --scoring_technique FixedNegSample \
      --num_epochs 50 --random_seed 1

# Resume again
dicee --continual_learning "UMLS_CL" \
      --model Keci \
      --scoring_technique FixedNegSample \
      --num_epochs 50 --random_seed 2
```

The `--continual_learning` path must contain:
- `configuration.json`
- `memory_map_train_set.npy`
- `entity_to_idx.csv` and `relation_to_idx.csv` (or legacy `.p` format)

---

## 5. Weight Averaging / Ensemble Methods

All methods work with `torchCPUTrainer` and `PL` (single GPU). Not supported for `TP` or `torchDDP`.

| Method | Flag | Notes |
|--------|------|-------|
| SWA | `--swa` | Averages weights from epoch 0 by default |
| ASWA | `--aswa` | Adaptive; ignores `--swa_start_epoch` |
| EMA | `--ema` | Exponential moving average |
| SWAG | `--swag` | Gaussian posterior over weights |
| TWA | `--twa` | Trainable blend of weight snapshots |

**Common options:**
- `--swa_start_epoch N` — start averaging from epoch N (default: 0, except ASWA)
- `--swa_c_epochs N` — averaging cycle interval (default: 1)

```bash
# SWA starting at epoch 50, cycling every 2 epochs
dicee --dataset_dir "KGs/UMLS" --model Keci --trainer PL \
      --scoring_technique KvsAll --num_epochs 100 \
      --swa --swa_start_epoch 50 --swa_c_epochs 2

# EMA from epoch 50
dicee --dataset_dir "KGs/UMLS" --model Keci --trainer PL \
      --scoring_technique KvsAll --num_epochs 100 \
      --ema --swa_start_epoch 50
```

---

## 6. Periodic Evaluation

Evaluate and/or save the model at fixed intervals or specific epochs during training.

| Flag | Description |
|------|-------------|
| `--eval_every_n_epochs N` | Evaluate every N epochs |
| `--eval_at_epochs 50 100 150` | Evaluate at specific epoch numbers |
| `--n_epochs_eval_model val_test` | Splits to evaluate: `val`, `test`, or `val_test` |
| `--save_every_n_epochs` | Save a model checkpoint at each evaluation point |

```bash
# Evaluate every 50 epochs, save checkpoints
dicee --dataset_dir "KGs/UMLS" --model Keci \
      --scoring_technique KvsAll --num_epochs 300 \
      --eval_every_n_epochs 50 --save_every_n_epochs \
      --n_epochs_eval_model val_test

# Evaluate at epochs 128 and 256 only
dicee --dataset_dir "KGs/UMLS" --model Keci \
      --scoring_technique KvsAll --num_epochs 300 \
      --eval_at_epochs 128 256 --n_epochs_eval_model val

# Combine both: every 100 plus at epochs 45 and 275
dicee --dataset_dir "KGs/UMLS" --model Keci \
      --scoring_technique KvsAll --num_epochs 300 \
      --eval_every_n_epochs 100 --eval_at_epochs 45 275 \
      --save_every_n_epochs --n_epochs_eval_model val
```

Periodic evaluation + weight averaging can be combined:
```bash
dicee --dataset_dir "KGs/UMLS" --model Keci \
      --scoring_technique KvsAll --num_epochs 300 \
      --eval_at_epochs 128 256 --save_every_n_epochs \
      --n_epochs_eval_model val --swa
```

---

## 7. Key Hyperparameter Guidelines

| Hyperparameter | Guidance |
|----------------|----------|
| `embedding_dim` | Start 64–128; large KGs benefit from 256–512 |
| `lr` | 0.1 with Adam is a strong default; reduce to 0.01 for fine-tuning |
| `batch_size` | 1024 default; increase for KvsAll on large KGs |
| `num_epochs` | 100–500; use periodic eval to find the sweet spot |
| `label_smoothing_rate` | 0.0–0.2; helps with over-confident models |
| `weight_decay` | 0.0 default; try 1e-5 to 1e-3 for regularisation |
| `normalization` | `"LayerNorm"` often improves stability; default is `"None"` |

For Clifford models (`Keci`, `CKeci`, `DeCaL`): ensure `embedding_dim % (p + q + 1) == 0`.
Default `p=0, q=1` gives standard Cl(0,1) ≈ complex numbers. Try `p=1, q=1` (Cl(1,1)) or `p=0, q=2`.

---

## 8. Input Data Formats

| Format | `--backend` | `--separator` | Notes |
|--------|-------------|---------------|-------|
| tab/space-separated triples | `pandas` (default) | `"\s+"` | train.txt, valid.txt, test.txt |
| Large N-Triples (.nt) | `polars` | `" "` | Faster for huge files |
| OWL / Turtle / RDF/XML | `rdflib` | — | Single file via `--path_single_kg` |
| SPARQL endpoint | — | — | Use `--sparql_endpoint URL` |

```bash
# OWL file
dicee --path_single_kg "KGs/Family/family.owl" --backend rdflib --model Keci

# Large N-Triples  
dicee --path_single_kg "KGs/large.nt" --backend polars --separator " " --model Keci

# SPARQL endpoint
dicee --sparql_endpoint "http://localhost:3030/dataset/" --model Keci
```

---

## 9. Result Inspection

After training, the experiment folder (under `Experiments/<timestamp>/` or `--path_to_store_single_run`) contains:
- `configuration.json` — full config snapshot
- `eval_report.json` — MRR, MR, HITS@1, HITS@3, HITS@10 per split
- `model.pt` — model weights for inference with `KGE(path=...)`

Load results:
```python
import json
with open("MyRun/eval_report.json") as f:
    report = json.load(f)
print(report)  # {'Train': {'MRR': ...}, 'Val': {'MRR': ...}, 'Test': {'MRR': ...}}
```
