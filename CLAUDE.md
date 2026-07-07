# dicee — Claude Code Instructions

`dicee` is a hardware-agnostic framework for training and using large-scale **Knowledge Graph Embedding (KGE)** models. Users train models on triples `(head, relation, tail)`, then query them for link prediction, multi-hop reasoning, and literal prediction.

**Entry points:**
- CLI: `dicee --dataset_dir "KGs/UMLS" --model Keci`
- Python training: `from dicee.executer import Execute; Execute(args).start()`
- Inference: `from dicee import KGE; model = KGE(path="...")`

Git workflow: PRs target `develop`, not `main` (see `CONTRIBUTING.md`).

---

## Architecture Pipeline

```
Input (dataset_dir | path_single_kg | sparql_endpoint)
  → ReadFromDisk       dicee/read_preprocess_save_load_kg/read_from_disk.py
  → PreprocessKG       dicee/read_preprocess_save_load_kg/preprocess.py
     → entity_to_idx, relation_to_idx, er_vocab, re_vocab, ee_vocab, memory_map_train_set.npy
  → construct_dataset()  dicee/dataset_classes/_factory.py
     → torch.utils.data.Dataset per scoring technique
  → DICE_Trainer       dicee/trainer/dice_trainer.py
     → wraps TorchTrainer | TorchDDPTrainer | TorchFSDPTrainer | TensorParallel | PyTorch Lightning
  → Evaluator          dicee/evaluation/evaluator.py (real implementation; dicee/evaluator.py is a re-export shim — edit the former)
     → MRR, MR, HITS@1, HITS@3, HITS@10
  → Results under storage_path / path_to_store_single_run
```

## Models

All models extend `BaseKGE` in `dicee/models/base_model.py` (`forward_triples` at line ~466, `forward_k_vs_all` at ~484). Required interface:

```python
class MyModel(BaseKGE):
    def __init__(self, args: dict): ...
    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor: ...   # (B,3) → (B,)
    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor: ...  # (B,2) → (B,|E|)
```

| File | Models |
|------|--------|
| `dicee/models/real.py` | DistMult, TransE, Shallom, Pyke, CoKE (+ CoKEConfig) |
| `dicee/models/complex.py` | ComplEx, ConEx, AConEx |
| `dicee/models/quaternion.py` | QMult, ConvQ, AConvQ |
| `dicee/models/octonion.py` | OMult, ConvO, AConvO |
| `dicee/models/clifford.py` | Keci, CKeci, DeCaL, KeciTransformer |
| `dicee/models/function_space.py` | FMult, GFMult, FMult2, LFMult1, LFMult |
| `dicee/models/dualE.py` | DualE |
| `dicee/models/pykeen_models.py` | PykeenKGE (wraps any PyKEEN model) |
| `dicee/models/transformers.py` | BytE (**not** exported from `models/__init__.py`; import directly) |

`dicee/models/__init__.py` star-imports real/complex/quaternion/octonion/pykeen_models/function_space and explicitly imports `Keci, CKeci, DeCaL, KeciTransformer` and `DualE`. `transformers.py`, `literal.py`, `ensemble.py`, and `fsdp_models.py` are **not** re-exported — import those submodules directly.

**Clifford model convention:** `embedding_dim / (p + q + 1)` must be an integer. The `args` dict passed to `__init__` comes from `vars(config.Namespace())`.

## Trainers

| `--trainer` | Class | File | Use Case |
|-------------|-------|------|----------|
| `torchCPUTrainer` | `TorchTrainer` | `torch_trainer.py` | CPU or single GPU |
| `PL` | `lightning.pytorch.Trainer` | — | Multi-GPU, recommended default |
| `torchDDP` | `TorchDDPTrainer` | `torch_trainer_ddp.py` | Native DDP via `torchrun` |
| `torchFSDP` | `TorchFSDPTrainer` | `torch_trainer_fsdp.py` | Fully-sharded data parallel |
| `TP` | `TensorParallel` | `model_parallelism.py` | Tensor parallelism (1 model/GPU) — "Multiple Run Ensemble Learning with Low-Dimensional KGE" |

Note: the dependency is the `lightning` package (`import lightning.pytorch as pl`), not the older standalone `pytorch-lightning`.

Multi-GPU with `PL` uses all visible CUDA devices automatically; restrict with `CUDA_VISIBLE_DEVICES=0`. `--path_to_store_single_run` is required for DDP/FSDP/multi-GPU runs.

## Scoring Techniques

Dataset classes live in `dicee/dataset_classes/` (`_negative_sampling.py`, `_label_based.py`, `_literal.py`, `_bpe.py`), all routed through `construct_dataset()` in `_factory.py`.

| `--scoring_technique` | Dataset Class | Notes |
|-----------------------|---------------|-------|
| `NegSample` | `TriplePredictionDataset` | Large KGs; set `--neg_ratio` |
| `FixedNegSample` | `FixedNegSampleDataset` | Continual learning; stable negatives |
| `1vsAll` | `OnevsAllDataset` | Small KGs |
| `KvsAll` | `KvsAll` | Default recommended |
| `KvsSample` | `KvsSampleDataset` | Balanced memory/speed |
| `AllvsAll` | `AllvsAll` | Full pairwise; very slow, avoid on large KGs |
| `1vsSample` | `OnevsSample` | Sampled 1-vs-all |
| `FSDP1vsSample` | `FSDP1vsSampleDataset` | FSDP-compatible variant |

## Key Configuration (`dicee/config.py`)

| Param | Default | Notes |
|-------|---------|-------|
| `dataset_dir` / `path_single_kg` / `sparql_endpoint` | `None` | Pick one input source |
| `backend` | `"pandas"` | `pandas` \| `polars` \| `rdflib` |
| `separator` | `"\s+"` | Triple file column separator |
| `model` | `"Keci"` | Model name string |
| `embedding_dim` | 64 | |
| `p`, `q` | 0, 1 | Clifford params |
| `normalization` | `"None"` | `"LayerNorm"` \| `"BatchNorm1d"` \| `"None"` |
| `num_epochs` | 150 | |
| `batch_size` | 1024 | |
| `optim` | `"Adam"` | `"Adam"` \| `"SGD"` \| `"ADOPT"` |
| `neg_ratio` | 0 | Must be ≥1 for `NegSample`/`FixedNegSample` |
| `trainer` | `"torchCPUTrainer"` | See trainer table above |
| `scoring_technique` | `"KvsAll"` | |
| `eval_model` | `"train_val_test"` | `"None"` \| `"train"` \| `"train_val"` \| `"train_val_test"` \| `"test"` |
| `eval_every_n_epochs` / `eval_at_epochs` | 0 / `None` | Periodic evaluation |
| `swa` / `adaptive_swa` (**not** `aswa`) / `ema` / `swag` / `twa` | `False` | Weight averaging flags — implemented in `dicee/weight_averaging.py` + `dicee/callbacks.py` |
| `swa_start_epoch` / `swa_c_epochs` | `None` / 1 | Averaging schedule |

## Callbacks (`dicee/callbacks.py`)

Extend `AbstractCallback` (`dicee/abstracts.py`, itself an ABC + `lightning.pytorch.callbacks.Callback`):

```python
class MyCallback(AbstractCallback):
    def on_fit_start(self, trainer, model): ...
    def on_train_epoch_end(self, trainer, model, loss): ...
    def on_fit_end(self, trainer, model, loss): ...
```
Register via `args.callbacks = {"MyCallback": {...config...}}`.

## Experiment Output Structure

```
Experiments/<timestamp>/            # storage_path or path_to_store_single_run
├── configuration.json
├── entity_to_idx.csv / relation_to_idx.csv
├── memory_map_train_set.npy
├── er_vocab.p / re_vocab.p
├── model.pt
└── eval_report.json                # MRR, MR, HITS@k per split
```

## Build, Lint, Test

```bash
pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu   # CPU dev install
ruff check dicee/ --line-length=200                                             # required before commit
mypy dicee/ --config-file=pyproject.toml                                        # recommended
python -m pytest -p no:warnings -x                                              # full suite (also default addopts)
```
Requires Python >= 3.11. Test datasets: `wget https://files.dice-research.org/datasets/dice-embeddings/KGs.zip && unzip KGs.zip`.

## Conventions

- Reference implementations: `real.py::DistMult` for simple bilinear models, `clifford.py::Keci` for Clifford algebra models.
- `args` passed to a model's `__init__` is a plain `dict` (`vars(namespace)`), not a `Namespace` — use `self.args.get("key", default)`.
- Don't redefine `entity_embeddings`/`relation_embeddings` in a new model — `BaseKGE.__init__` creates them.
- Loss function is chosen in `BaseKGE.__init__` based on `form_of_labelling` (`"EntityPrediction"` or `"RelationPrediction"`).
- Reciprocal training auto-creates `rel_inverse` relations when enabled.
- BPE models (`--byte_pair_encoding`) use subword token sequences as entity/relation indices.
- Type hints required on new public functions (`Optional[T]`, explicit return types); Google-style docstrings.
- Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `perf:`); PRs target `develop`.
- `KGs.zip` and any loose root-level scratch scripts (e.g. `demo.py`, `example*.md`) are local artifacts, not canonical entry points — don't treat them as documented API.

## Specialized Subagents

For focused work, delegate via the Agent tool to these project subagents (`.claude/agents/`) rather than guessing APIs from memory:

| Task | Subagent |
|------|----------|
| Implement/extend a KGE model, new scoring function, algebra-based embedding | `kge-model-developer` |
| Configure/launch training, trainer/scoring-technique choice, multi-GPU, SWA/EMA, continual learning | `kge-trainer` |
| Diagnose poor MRR/HITS@k, NaN loss, over/underfitting (read-only) | `kge-debugger` |
| Inference (`KGE` class), `predict_topk`, multi-hop queries, embeddings, literal prediction, Gradio | `kge-analyst` |

Skills for detailed step-by-step workflows: `/add-model`, `/run-training`, `/link-prediction-api`. Slash commands: `/debug-evaluation`, `/extend-scoring-technique`.
