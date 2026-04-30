# Examples → Tests Migration Plan

## Executive Summary
Replace the `examples/` folder with direct links to test files in README.md. Tests are the single source of truth — they're maintained, verified by CI, and guaranteed to work.

---

## Audit: Examples vs Tests Coverage

### ✅ COVERED: Examples with Test Equivalents

| Example File | Purpose | Test File(s) | Status |
|-------------|---------|--------------|--------|
| `training.py` | Basic model training | `test_execute_start.py`, `test_regression_*.py` | ✅ Well covered |
| `eval_link_prediction.py` | Link prediction evaluation | `test_predict_kge.py`, `test_download_and_eval.py` | ✅ Well covered |
| `eval_multi_hop_query_answering.py` | Multi-hop queries | `test_answer_multi_hop_query.py` | ✅ Well covered |
| `download_pretrained_model.py` | Download & use pretrained | `test_download_and_eval.py` | ✅ Covered |
| `KGE_literal_prediction.py` | Literal prediction | `test_predict_kge_literals.py` | ✅ Covered |
| `adaptive_swa.sh` | SWA weight averaging | `test_adaptive_swa.py`, `test_swa.py` | ✅ Covered |
| `run_pykeen.py` | PyKEEN integration | `test_pykeen.py` | ✅ Covered |
| `train_kge_on_kg_larger_than_memory.py` | Large KG training | `test_large_kg.py` | ✅ Covered |
| `experiment_example.sh` | Shell script training | README examples | ✅ Already in README |
| `Trainers.ipynb` | Trainer comparison | `test_trainers.py`, `test_custom_trainer.py` | ✅ Covered |
| `KnowledgeGraphEmbeddings.ipynb` | Interactive tutorial | Multiple test files | ✅ Covered |

### 📝 DOCUMENTATION-ONLY: Keep in Docs

| File | Type | Recommendation |
|------|------|----------------|
| `Datasets.md` | Documentation | Move to `docs/` or README section |
| `Pykeen.md` | Documentation | Move to `docs/` or README section |
| `multi_hop_query_answering/` | Documentation | Merge into README with test links |

### 🔍 UTILITIES

| File | Purpose | Action |
|------|---------|--------|
| `util.py` | Helper functions | Move to `tests/conftest.py` if needed |
| `comparison_example.py` | Model comparison | Check if duplicates test functionality |

---

## Proposed README Structure

### Section 1: Quick Start (Keep Current)
```markdown
## Quick Start

| Feature | Command/Code |
|---------|-------------|
| **Install (CPU)** | `pip install dicee --extra-index-url https://download.pytorch.org/whl/cpu` |
| **Train model** | `dicee --dataset_dir "KGs/UMLS" --model Keci` |
| **Load pretrained** | `from dicee import KGE; model = KGE(path='...')` |
```

### Section 2: Training Models (Enhanced)
```markdown
## Training Models

### Basic Training (Python API)

```python
from dicee.executer import Execute
from dicee.config import Namespace

args = Namespace()
args.model = 'Keci'
args.scoring_technique = "KvsAll"
args.dataset_dir = "KGs/UMLS"
args.num_epochs = 100
args.embedding_dim = 32
args.batch_size = 1024

reports = Execute(args).start()
print(reports["Train"]["MRR"])  # => 0.9912
print(reports["Test"]["MRR"])   # => 0.8155
```

**📖 [See more training examples in tests →](tests/test_execute_start.py)**

### Multi-GPU / Distributed Training

```bash
# PyTorch Lightning (recommended)
dicee --dataset_dir "KGs/UMLS" --trainer "PL" --model "Keci"

# Native DDP
torchrun --standalone --nnodes=1 --nproc_per_node=gpu dicee --dataset_dir "KGs/UMLS" --model Keci --trainer "torchDDP"
```

**📖 [See distributed training examples →](tests/test_trainers.py)**

### Large Knowledge Graphs

```python
from dicee.executer import Execute
from dicee.config import Namespace

args = Namespace()
args.path_single_kg = "large_kg.nt"
args.model = 'Keci'
args.backend = "polars"  # Efficient for large KGs
args.scoring_technique = "NegSample"
args.neg_ratio = 10

Execute(args).start()
```

**📖 [See large KG training examples →](tests/test_large_kg.py)**
```

### Section 3: Inference & Evaluation (New)
```markdown
## Inference & Link Prediction

### Download & Use Pretrained Models

```python
from dicee import KGE

# Download from URL
model = KGE(url="https://files.dice-research.org/projects/DiceEmbeddings/KINSHIP-Keci-dim128-epoch256-KvsAll")

# Make predictions
model.predict(h="person49", r="term12", t="person39")
```

**📖 [See download & evaluation examples →](tests/test_download_and_eval.py)**

### Link Prediction

```python
from dicee import KGE

model = KGE(path="Experiments/2024-01-15...")

# Predict missing tail
predictions = model.predict_topk(h=["entity1"], r=["relation1"], topk=10)

# Predict missing head  
predictions = model.predict_topk(r=["relation1"], t=["entity2"], topk=10)
```

**📖 [See complete link prediction examples →](tests/test_predict_kge.py)**

### Multi-Hop Query Answering

```python
from dicee import KGE

model = KGE(path="...")

# 1-hop: Who are the siblings of F9M167?
predictions = model.answer_multi_hop_query(
    query_type="1p",
    query=('http://www.benchmark.org/family#F9M167',
           ('http://www.benchmark.org/family#hasSibling',)),
    tnorm="min", k=3
)

# 2-hop: To whom is a sibling of F9M167 married?
predictions = model.answer_multi_hop_query(
    query_type="2p",
    query=("http://www.benchmark.org/family#F9M167",
           ("http://www.benchmark.org/family#hasSibling",
            "http://www.benchmark.org/family#married")),
    tnorm="min", k=3
)
```

**📖 [See multi-hop query examples →](tests/test_answer_multi_hop_query.py)**  
**📖 [EPFO query types: 1p, 2p, 3p, 2i, 3i, ip, pi, 2u, up →](docs/multi_hop_queries.md)**

### Literal Prediction

```python
from dicee import KGE

model = KGE(path="...")

# Predict numeric/literal values
predictions = model.predict_literals(h=["entity"], r=["hasAge"])
```

**📖 [See literal prediction examples →](tests/test_predict_kge_literals.py)**
```

### Section 4: Advanced Training (Enhanced)
```markdown
## Advanced Training

### Weight Averaging Ensembles

```bash
# Stochastic Weight Averaging
dicee --dataset_dir "KGs/UMLS" --model "Keci" --swa --swa_start_epoch 50

# Adaptive SWA
dicee --dataset_dir "KGs/UMLS" --model "Keci" --aswa

# Exponential Moving Average
dicee --dataset_dir "KGs/UMLS" --model "Keci" --ema --swa_start_epoch 50
```

**📖 [See weight averaging examples →](tests/test_swa.py) | [Adaptive SWA →](tests/test_adaptive_swa.py)**

### Continual Learning

```bash
# Initial training
dicee --path_single_kg "KGs/Family/family.owl" --model Keci --path_to_store_single_run KeciFamilyRun

# Resume training
dicee --continual_learning "KeciFamilyRun" --num_epochs 50
```

**📖 [See continual learning examples →](tests/test_continual_training.py)**

### Periodic Evaluation

```bash
# Evaluate every 50 epochs and save checkpoints
dicee --dataset_dir "KGs/UMLS" --model Keci --num_epochs 300 \
      --eval_every_n_epochs 50 --save_every_n_epochs --n_epochs_eval_model val_test
```

**📖 [See periodic evaluation examples →](tests/test_periodic_eval_callback.py)**
```

### Section 5: Scoring Techniques & Backends
```markdown
## Scoring Techniques

| Technique | Memory | Speed | Use Case |
|-----------|--------|-------|----------|
| `KvsAll` | Medium | Fast | Default (recommended) |
| `NegSample` | Low | Very Fast | Large KGs |
| `1vsAll` | High | Medium | Small KGs |

```bash
dicee --dataset_dir "KGs/YAGO3-10" --model Keci --scoring_technique "NegSample" --neg_ratio 10
```

**📖 [See scoring technique comparisons →](tests/test_k_fold_cv_*.py)**

## Data Backends

| Backend | Format | Use Case |
|---------|--------|----------|
| `pandas` | TSV/CSV | Default |
| `polars` | N-Triples | Large KGs (faster) |
| `rdflib` | RDF/OWL/Turtle | Semantic web |

```bash
# Large n-triples file
dicee --path_single_kg "large.nt" --backend "polars" --separator " "

# OWL ontology
dicee --path_single_kg "ontology.owl" --backend "rdflib"
```

**📖 [See backend examples →](tests/test_different_backends.py)**
```

### Section 6: PyKEEN Integration
```markdown
## PyKEEN Integration

Use any PyKEEN model with DICEE infrastructure:

```python
from dicee.executer import Execute
from dicee.config import Namespace

args = Namespace()
args.model = 'Pykeen_QuatE'  # or any PyKEEN model
args.dataset_dir = "KGs/UMLS"
args.num_epochs = 100

Execute(args).start()
```

**📖 [See PyKEEN examples →](tests/test_pykeen.py)**  
**📖 [Available PyKEEN models →](https://github.com/pykeen/pykeen#models)**
```

---

## Migration Steps

### Phase 1: Documentation Update ✅
1. ✅ Update README.md with new structure (code snippets + test links)
2. Move `Datasets.md` → `docs/datasets.md`
3. Move `Pykeen.md` → `docs/pykeen_integration.md`
4. Create `docs/multi_hop_queries.md` from `multi_hop_query_answering/`

### Phase 2: Test Verification ✅
1. Verify all referenced tests pass: `pytest -p no:warnings -x`
2. Add missing test cases (if any gaps found)
3. Ensure test code is well-documented with docstrings

### Phase 3: Cleanup 🗑️
1. Delete `examples/` folder
2. Update any CI/documentation references from `examples/` → `tests/`
3. Update `setup.py` / `MANIFEST.in` if needed

### Phase 4: Validation ✅
1. Run full test suite
2. Check all README links work
3. Verify CI passes

---

## Benefits Summary

✅ **Single source of truth** — tests are always up to date  
✅ **CI-verified** — examples are guaranteed to work  
✅ **Less maintenance** — update once, not twice  
✅ **Better docs** — README stays concise, tests show full context  
✅ **Discoverability** — users learn the test suite structure  

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Tests are harder to read than examples | Add detailed docstrings to test functions |
| Users want copy-paste code | Include complete snippets in README |
| Notebooks are interactive | Link to Colab/Binder versions in docs |
| Breaking changes in tests | Pin examples to specific test functions, not line numbers |

---

## Next Steps

**Review this plan, then I will:**
1. Draft the new README sections
2. Verify test coverage gaps
3. Create missing documentation in `docs/`
4. Execute the migration (update README, move files, delete examples/)
5. Run `ruff check` and commit

**Your decision:** Proceed with migration? Any sections to adjust?
