# DICE Embeddings — Project Instructions

## Project Overview

`dicee` is a hardware-agnostic framework for training and using large-scale **Knowledge Graph Embedding (KGE)** models. Users train models on triples `(head, relation, tail)`, then query them for link prediction, multi-hop reasoning, and literal prediction.

**Key entry points:**
- CLI: `dicee --dataset_dir "KGs/UMLS" --model Keci`
- Python: `from dicee.executer import Execute; Execute(args).start()`
- Inference: `from dicee import KGE; model = KGE(path="...")`

---

## Architecture Pipeline

```
Input (dataset_dir | path_single_kg | sparql_endpoint)
  ↓ ReadFromDisk (dicee/read_preprocess_save_load_kg/read_from_disk.py)
  ↓ PreprocessKG  (dicee/read_preprocess_save_load_kg/preprocess.py)
     → entity_to_idx, relation_to_idx, er_vocab, re_vocab, ee_vocab
     → memory_map_train_set.npy
  ↓ construct_dataset() (dicee/dataset_classes/_factory.py)
     → torch.utils.data.Dataset per scoring technique
  ↓ DICE_Trainer (dicee/trainer/dice_trainer.py)
     → wraps TorchTrainer | TorchDDPTrainer | TensorParallel | PyTorch Lightning
  ↓ Evaluation (dicee/evaluation/evaluator.py)
     → MRR, MR, HITS@1, HITS@3, HITS@10
  ↓ Results stored under storage_path / path_to_store_single_run
```

---

## Models

All models extend `BaseKGE` in `dicee/models/base_model.py`. Required interface:

```python
class MyModel(BaseKGE):
    def __init__(self, args: dict): ...
    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor: ...   # (B,3) → (B,)
    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor: ...  # (B,2) → (B,|E|)
```

| File | Models |
|------|--------|
| `dicee/models/real.py` | DistMult, TransE, Pyke, Shallom, LFMult |
| `dicee/models/complex.py` | ComplEx, ConEx, AConEx |
| `dicee/models/quaternion.py` | QMult, ConvQ, AConvQ |
| `dicee/models/octonion.py` | OMult, ConvO, AConvO |
| `dicee/models/clifford.py` | Keci, CKeci, DeCaL, KeciTransformer |
| `dicee/models/function_space.py` | DualE |
| `dicee/models/transformers.py` | BytE, CoKE |
| `dicee/models/pykeen_models.py` | PykeenKGE (wraps any PyKEEN model) |

All models are exported from `dicee/models/__init__.py`.

**Clifford model convention:** `embedding_dim` must satisfy `embedding_dim / (p + q + 1) ∈ ℤ`.
The `args` dict passed to `__init__` comes from `config.Namespace.__dict__`.

---

## Trainers

| `--trainer` value | Class | Use Case |
|-------------------|-------|----------|
| `torchCPUTrainer` | `TorchTrainer` | CPU or single GPU |
| `PL` | PyTorch Lightning | Multi-GPU, recommended default |
| `torchDDP` | `TorchDDPTrainer` | Native DDP via `torchrun` |
| `TP` | `TensorParallel` | Tensor parallelism (1 model per GPU) |

Multi-GPU with PL uses all visible CUDA devices automatically.
Use `CUDA_VISIBLE_DEVICES=0` to restrict to one GPU.
`--path_to_store_single_run` is required for DDP/multi-GPU runs.

---

## Scoring Techniques

| `--scoring_technique` | Dataset Class | Memory | Speed | Best For |
|-----------------------|---------------|--------|-------|----------|
| `NegSample` | `TriplePredictionDataset` | Low | Fast | Large KGs |
| `FixedNegSample` | `FixedNegSampleDataset` | Low | Fast | Continual learning |
| `1vsAll` | `OnevsAllDataset` | Medium | Medium | Small KGs |
| `KvsAll` | `KvsAll` | High | Slow | Default recommended |
| `KvsSample` | `KvsSampleDataset` | Medium | Medium | Balanced |
| `AllvsAll` | `AllvsAll` | Very high | Very slow | Pairwise loss |

Dataset classes live in `dicee/dataset_classes/`. All are routed through `construct_dataset()` in `dicee/dataset_classes/_factory.py`.

---

## Key Configuration Parameters (`dicee/config.py`)

### Data
| Param | Default | Description |
|-------|---------|-------------|
| `dataset_dir` | None | Folder with train.txt / valid.txt / test.txt |
| `path_single_kg` | None | Single RDF/OWL file |
| `sparql_endpoint` | None | SPARQL endpoint URL |
| `backend` | `"pandas"` | `pandas` \| `polars` \| `rdflib` |
| `separator` | `"\s+"` | Triple file column separator |

### Model
| Param | Default | Description |
|-------|---------|-------------|
| `model` | `"Keci"` | Model name string |
| `embedding_dim` | 64 | Embedding vector size |
| `p` | 0 | Clifford p-parameter |
| `q` | 1 | Clifford q-parameter |
| `input_dropout_rate` | 0.0 | Dropout on input embeddings |
| `hidden_dropout_rate` | 0.0 | Dropout on hidden layer |
| `normalization` | `"None"` | `"LayerNorm"` \| `"BatchNorm1d"` \| `"None"` |

### Training
| Param | Default | Description |
|-------|---------|-------------|
| `num_epochs` | 150 | Training epochs |
| `batch_size` | 1024 | Mini-batch size |
| `lr` | 0.1 | Learning rate |
| `optim` | `"Adam"` | `"Adam"` \| `"SGD"` \| `"ADOPT"` |
| `weight_decay` | 0.0 | L2 regularization |
| `neg_ratio` | 0 | Negatives per positive (NegSample) |
| `label_smoothing_rate` | 0.0 | Label smoothing coefficient |
| `trainer` | `"torchCPUTrainer"` | Trainer backend |
| `scoring_technique` | `"KvsAll"` | Scoring/labelling strategy |
| `num_core` | 0 | DataLoader worker processes |

### Evaluation
| Param | Default | Description |
|-------|---------|-------------|
| `eval_model` | `"train_val_test"` | Splits to evaluate: `"None"` \| `"train"` \| `"train_val"` \| `"train_val_test"` \| `"test"` |
| `eval_every_n_epochs` | 0 | Periodic eval interval (0 = disabled) |
| `eval_at_epochs` | None | List of specific epochs to evaluate |
| `n_epochs_eval_model` | `"val_test"` | Splits for periodic eval |
| `save_every_n_epochs` | False | Save checkpoint at each periodic eval |

### Ensemble / Weight Averaging
| Flag | Method |
|------|--------|
| `--swa` | Stochastic Weight Averaging |
| `--aswa` | Adaptive SWA |
| `--ema` | Exponential Moving Average |
| `--swag` | SWA-Gaussian |
| `--twa` | Trainable Weight Averaging |
| `--swa_start_epoch N` | Start epoch (default: 0, except ASWA) |
| `--swa_c_epochs N` | Averaging cycle interval |

Weight averaging implemented in `dicee/weight_averaging.py` and `dicee/callbacks.py`.

---

## Callbacks Pattern (`dicee/callbacks.py`)

Callbacks must extend `AbstractCallback` from `dicee/abstracts.py`:

```python
class MyCallback(AbstractCallback):
    def on_fit_start(self, trainer, model): ...
    def on_train_epoch_end(self, trainer, model, loss): ...
    def on_fit_end(self, trainer, model, loss): ...
```

Register via: `args.callbacks = {"MyCallback": {...config...}}`

---

## Experiment Output Structure

```
Experiments/                          # storage_path (or path_to_store_single_run)
└── <timestamp>/                      # one folder per run
    ├── configuration.json            # full run config
    ├── entity_to_idx.csv             # entity → integer index
    ├── relation_to_idx.csv           # relation → integer index
    ├── memory_map_train_set.npy      # indexed triples
    ├── er_vocab.p                    # (entity,relation) → tail set
    ├── re_vocab.p                    # (relation,entity) → head set
    ├── model.pt                      # final model weights
    └── eval_report.json              # MRR, MR, HITS@k per split
```

---

## Build and Test

```bash
# Install (CPU)
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu

# Install (dev, all extras)
pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu

# Run all tests
python -m pytest -p no:warnings -x

# Run only last failed
python -m pytest -p no:warnings --lf
```

---

## Conventions

- Use `dicee/models/real.py` (`DistMult`) as the reference for simple bilinear models.
- Use `dicee/models/clifford.py` (`Keci`) as the reference for Clifford algebra models.
- `args` passed to model `__init__` is a `dict` (call `vars(namespace)` before passing).
- Loss functions are set in `BaseKGE.__init__` based on the scoring technique's labelling form.
- `form_of_labelling` is either `"EntityPrediction"` or `"RelationPrediction"`.
- Reciprocal training automatically creates `rel_inverse` relations when enabled.
- For BPE models (`--byte_pair_encoding`), entity/relation indices are subword token sequences.

See skills for specific workflows:
- `/add-model` — add a new KGE model
- `/run-training` — configure and launch training
- `/link-prediction-api` — use a trained model for inference
