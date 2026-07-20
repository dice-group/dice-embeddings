# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- FSDP (Fully Sharded Data Parallel) trainer for distributed training across multiple GPUs with model sharding
- `FSDP1vsSampleDataset` for FSDP-compatible 1-vs-sample scoring technique
- New embedding models:
  - **RotaE**: Rotation-based translation model with learnable rotations in the embedding space
  - Additional models in `dicee/models/real.py` (DistMult, TransE, Pyke, Shallom, CoKE)
- Claude Code integration: project memory, specialized subagents, and skills for model development and training workflows
- Deterministic unit test for Parquet reading regression (`test_unit_read_parquet.py`)
- Comprehensive unit test suite for KGE inference API validation (`test_unit_kge_inference.py`):
  - 32 unit tests covering input validation, device management, entity embedding extraction
  - Tests validate exact exception messages for `predict_topk()` and `predict()` methods
  - Validation tests verify conditional branching logic for missing head/relation/tail predictions
  - Addresses IMPROVEMENTS.md #3 ("Core modules lack dedicated unit tests")
- Type hint improvements and gradual mypy enforcement roadmap (IMPROVEMENTS.md #5):
  - Fixed Optional type annotations in `dicee/config.py` (11 fields now properly annotated as `Optional[T]`)
  - Created `docs/TYPING_ROADMAP.md` documenting 3-tier type hint enforcement strategy
  - CI matrix now tests Python 3.11, 3.12, and 3.13 (IMPROVEMENTS.md #6)
- TODO/FIXME backlog organization and categorization (IMPROVEMENTS.md #7):
  - Created `docs/TODO_BACKLOG.md` cataloging 56+ TODO/FIXME markers by priority
  - Categorized by impact: 6 Medium-priority items (performance/design), 25+ Low-priority (refactoring)

### Changed
- **Breaking change**: Converted 298 `assert` statements to explicit exceptions (`ValueError`, `TypeError`) in public API paths:
  - [dicee/knowledge_graph_embeddings.py](dicee/knowledge_graph_embeddings.py) (45 asserts)
  - [dicee/read_preprocess_save_load_kg/preprocess.py](dicee/read_preprocess_save_load_kg/preprocess.py) (19 asserts)
  - User-facing validation on `KGE.__init__`, `predict_topk`, preprocessing pipeline now works in Python optimized mode (`-O`)
- **Breaking change**: Migrated 33 modules from `print()` to Python's `logging` module:
  - Trainers, models, evaluation code, and core utilities now use per-module `logging.getLogger(__name__)`
  - Library consumers can now control verbosity, silence output, or redirect logs without monkeypatching
- Improved TP (Tensor Parallel) trainer error message when insufficient GPUs are available
- Updated README with FSDP trainer documentation

### Fixed
- **#410**: `--read_only_few` now efficiently loads only the requested rows from Parquet files
  - Previously: full Parquet file was loaded into memory, then `.head(n)` was applied (~26.9 MB peak for 200k rows requesting 10 rows)
  - Now: uses `pyarrow.parquet.ParquetFile.iter_batches()` to stop loading row groups early (~0.2 MB peak, ~135x improvement)
  - CSV/text inputs already used `nrows=` and were unaffected
  - Added regression test: `test_unit_read_parquet.py` with 6 deterministic tests covering row count, content, and order validation
- Fixed TexLive latest version compatibility in FSDP trainer documentation
- Fixed trainer model type consistency and precision handling in FSDP trainer

### Deprecated
- Direct use of `print()` in dicee modules — use `logging` module instead

### Removed
- (none in this release)

### Security
- (none in this release)

---

## Known Issues (from IMPROVEMENTS.md)

The following issues are being tracked for future resolution:

### Priority: Medium
- **Core modules lack dedicated unit tests** (#3) — **Partially Resolved**
  - ✅ [dicee/knowledge_graph_embeddings.py](dicee/knowledge_graph_embeddings.py) now has comprehensive unit test suite (32 tests in `test_unit_kge_inference.py`)
  - ⏳ `executer.py`, `static_funcs.py`, `query_generator.py`, `knowledge_graph.py` still only exercised indirectly through integration tests
  - Scoring-logic bugs in `predict()` and `predict_topk()` now caught by unit tests

### Priority: Low  
- **mypy is non-blocking in CI** (#5)
  - `.github/workflows/github-actions-python-package.yml` runs with `continue-on-error: true`
  - No mechanism to enforce gradual type coverage improvement

- **CI only tests one Python version** (#6)
  - Package declares `python >= 3.11` but only tests 3.11.14
  - No signal if it breaks on 3.12/3.13

- **42 accumulated TODO/FIXME markers** (#7)
  - Several flag uncertainty about existing logic rather than pending work
  - Examples: `abstracts.py:721`, `knowledge_graph.py:132`, `query_generator.py:226`
  - Worth a documentation/cleanup pass before original authors' context fades

---

## Migration Guide

### For users updating to this release

#### 1. Exception handling instead of asserts
If you were relying on `try/except AssertionError` to catch KGE validation errors, update to catch the specific exception type:

```python
# Old (may fail in optimized Python)
try:
    model = KGE(path="...")
except AssertionError as e:
    handle_error(e)

# New (recommended)
try:
    model = KGE(path="...")
except (ValueError, TypeError) as e:
    handle_error(e)
```

#### 2. Capturing logs from dicee
If you were suppressing dicee output via monkeypatching `print()`, use Python's logging instead:

```python
import logging

# Suppress dicee logs
logging.getLogger("dicee").setLevel(logging.WARNING)

# Or silence all dicee modules
for module in ["dicee.trainer", "dicee.models", "dicee.evaluation"]:
    logging.getLogger(module).setLevel(logging.ERROR)
```

---

## Changelog Status

✅ **Items marked DONE in IMPROVEMENTS.md**:
- [x] Item #1: `--read_only_few` Parquet loading
- [x] Item #2: Assert validation → exceptions  
- [x] Item #4: print() → logging

📋 **Items still TODO**:
- [ ] Item #3: Core user-facing modules unit test coverage
- [ ] Item #5: mypy blocking CI with ratchet
- [ ] Item #6: Multi-version Python CI matrix
- [ ] Item #7: TODO/FIXME documentation pass
