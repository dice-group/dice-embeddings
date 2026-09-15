# dicee — Potential Improvements

Living tracking doc for codebase-quality findings and their status. Not user-facing documentation — a working list for maintainers.

## In progress / done

### 1. `--read_only_few` loads the full Parquet file before slicing
**Status:** Fixed (this branch)
**Tracking:** [dice-group/dice-embeddings#410](https://github.com/dice-group/dice-embeddings/issues/410)

`read_with_pandas()` in [dicee/read_preprocess_save_load_kg/util.py](dicee/read_preprocess_save_load_kg/util.py) called `pd.read_parquet(data_path)` unconditionally, then `.head(read_only_few)` — so the entire file was materialized in memory regardless of how few rows were requested. CSV/text inputs already used `nrows=` and didn't have this problem.

Fix: added `_read_parquet_head()`, which uses `pyarrow.parquet.ParquetFile.iter_batches()` to stop pulling row groups as soon as enough rows have accumulated, only falling back to a full `pd.read_parquet` when `read_only_few` is unset. Verified with a synthetic 200k-row Parquet file: requesting 10 rows dropped peak traced memory from ~26.9 MB (full read) to ~0.2 MB (~135x), with identical row content to the old behavior.

Regression test added: [tests/test_unit_read_parquet.py](tests/test_unit_read_parquet.py) generates a dummy Parquet file via a `tmp_path` fixture (no dependency on the `KGs/` dataset download) and checks: exact row count for `read_only_few`, row order/content matches `.head(n)` on the full file, oversized `read_only_few` returns all rows, `None` returns the full file. The actual regression guard for #410 mocks `pd.read_parquet` and asserts it is **not called** when `read_only_few` is set (only the row-group-bounded `_read_parquet_head` path runs) and **is called** when it's unset — a first attempt used `tracemalloc` peak-memory comparisons instead, which was flaky on small fixture files where fixed overhead dominated the signal in CI. All 6 tests pass, deterministically.

### 2. `assert` used for runtime validation in public API paths
**Status:** Fixed (scoped to API boundaries)

Converted the `assert` statements guarding user input at public API boundaries — `KGE.__init__`, `predict`, `predict_topk`, `to`, `answer_multi_hop_query`, `find_missing_triples`, `predict_literals` in [dicee/knowledge_graph_embeddings.py](dicee/knowledge_graph_embeddings.py), and the full validation path in [dicee/read_preprocess_save_load_kg/preprocess.py](dicee/read_preprocess_save_load_kg/preprocess.py) — to explicit `ValueError`/`TypeError`. Internal tensor-shape/invariant checks (not user input) were deliberately left as `assert`; ~246 remain across `dicee/`, mostly internal sanity checks in query-answering loops and model internals. Not literally "all asserts converted" — only the ones that guard values coming from outside the library.

### 4. `print()` instead of `logging` throughout the library
**Status:** Fixed

Migrated `print()` calls in trainers, models, evaluation code, and core utilities to per-module `logging.getLogger(__name__)`. Library consumers can now control verbosity, silence output, or redirect logs without monkeypatching. A handful of `print(...)` strings remain, but only inside docstrings/commented-out dead code (e.g. `models/clifford.py`, `trainer/torch_trainer.py`) — not live paths.

### 5. `mypy` is non-blocking in CI with no ratchet
**Status:** Fixed

`continue-on-error: true` is removed from the mypy step in [.github/workflows/github-actions-python-package.yml](.github/workflows/github-actions-python-package.yml). The step now runs mypy, compares the `Found N errors` count against a baseline recorded in [.github/mypy-baseline.txt](.github/mypy-baseline.txt) (1367, the count at the time this was added), and fails the build only if the count increases — new code can no longer add fresh type errors, while the existing 1367 stay non-blocking until someone pays them down deliberately (lowering the baseline file locks in each improvement). `pyproject.toml`'s `disallow_untyped_defs = false` and the "start lenient, tighten later" comment are unchanged; this is a regression gate, not a rewrite of the strictness config. Also removed two dead `ignore_missing_imports` entries (`gradio.*`, `owlready2.*`) that mypy itself flagged as "unused section(s)" — neither package is imported or declared as a dependency anywhere in the repo.

## Open findings

### 3. Core user-facing modules have no dedicated unit tests
**Status:** Partially resolved

[dicee/knowledge_graph_embeddings.py](dicee/knowledge_graph_embeddings.py) (the main inference API) now has a dedicated unit test suite ([tests/test_unit_kge_inference.py](tests/test_unit_kge_inference.py), 32 tests covering input validation, device management, entity embedding extraction). `executer.py`, `static_funcs.py`, `query_generator.py`, and `knowledge_graph.py` are still only exercised indirectly through regression/integration tests.

### 6. CI only tests a single Python version (3.11.14)
**Status:** Fixed

`.github/workflows/github-actions-python-package.yml` now runs the matrix against Python 3.11, 3.12, and 3.13, matching the `python_requires >= 3.11` declaration in `pyproject.toml`.

### 7. Accumulated "why do we need this?" TODOs
**Status:** Cataloged, not resolved

42 TODO/FIXME markers in `dicee/` (45 including `tests/`), several of which flag genuine uncertainty about existing logic rather than pending work — e.g. `abstracts.py:721` (`# @TODO: Do we really need this ?!`), `knowledge_graph.py:132` (`# TODO:CD: Why do we need to create this inverse mapping at this point?`), and `query_generator.py:226` (`# @TODO: Why do we need a deep copy here ?`). Cataloged by priority in [docs/TODO_BACKLOG.md](docs/TODO_BACKLOG.md); none of the underlying uncertainty has actually been resolved yet — that still needs the original authors' input.
