# dicee — Potential Improvements

Living tracking doc for codebase-quality findings and their status. Not user-facing documentation — a working list for maintainers.

## In progress / done

### 1. `--read_only_few` loads the full Parquet file before slicing
**Status:** Fixed (this branch)
**Tracking:** [dice-group/dice-embeddings#410](https://github.com/dice-group/dice-embeddings/issues/410)

`read_with_pandas()` in [dicee/read_preprocess_save_load_kg/util.py](dicee/read_preprocess_save_load_kg/util.py) called `pd.read_parquet(data_path)` unconditionally, then `.head(read_only_few)` — so the entire file was materialized in memory regardless of how few rows were requested. CSV/text inputs already used `nrows=` and didn't have this problem.

Fix: added `_read_parquet_head()`, which uses `pyarrow.parquet.ParquetFile.iter_batches()` to stop pulling row groups as soon as enough rows have accumulated, only falling back to a full `pd.read_parquet` when `read_only_few` is unset. Verified with a synthetic 200k-row Parquet file: requesting 10 rows dropped peak traced memory from ~26.9 MB (full read) to ~0.2 MB (~135x), with identical row content to the old behavior.

Regression test added: [tests/test_unit_read_parquet.py](tests/test_unit_read_parquet.py) generates a dummy Parquet file via a `tmp_path` fixture (no dependency on the `KGs/` dataset download) and checks: exact row count for `read_only_few`, row order/content matches `.head(n)` on the full file, oversized `read_only_few` returns all rows, `None` returns the full file. The actual regression guard for #410 mocks `pd.read_parquet` and asserts it is **not called** when `read_only_few` is set (only the row-group-bounded `_read_parquet_head` path runs) and **is called** when it's unset — a first attempt used `tracemalloc` peak-memory comparisons instead, which was flaky on small fixture files where fixed overhead dominated the signal in CI. All 6 tests pass, deterministically.

## Open findings

### 2. `assert` used for runtime validation in public API paths
298 `assert` statements across `dicee/`, concentrated in [dicee/knowledge_graph_embeddings.py](dicee/knowledge_graph_embeddings.py) (45) and [dicee/read_preprocess_save_load_kg/preprocess.py](dicee/read_preprocess_save_load_kg/preprocess.py) (19). Asserts are stripped when Python runs with `-O`, so validation on the public `KGE` inference class and preprocessing pipeline silently disappears in optimized mode. Convert the ones guarding user input at API boundaries (`KGE.__init__`, `predict_topk`, etc.) to explicit `ValueError`/`TypeError`.

### 3. Core user-facing modules have no dedicated unit tests
[dicee/knowledge_graph_embeddings.py](dicee/knowledge_graph_embeddings.py) (1423 lines — the main inference API), `executer.py`, `static_funcs.py`, `query_generator.py`, and `knowledge_graph.py` are only exercised indirectly through regression/integration tests. A scoring-logic bug (e.g. in `predict_topk`) could pass CI as long as a higher-level regression test's loose assertions don't happen to catch it.

### 4. `print()` instead of `logging` throughout the library
33 modules call `print()` directly (trainers, models, evaluation code); only 1 module uses Python's `logging`. Library consumers embedding `dicee` in larger pipelines can't control verbosity, silence it, or redirect output without monkeypatching. Recommend migrating to per-module `logging.getLogger(__name__)`.

### 5. `mypy` is non-blocking in CI with no ratchet
`.github/workflows/github-actions-python-package.yml` runs mypy with `continue-on-error: true`, and `pyproject.toml` sets `disallow_untyped_defs = false` with a "start lenient, tighten later" comment. There's no mechanism forcing that tightening to actually happen over time.

### 6. CI only tests a single Python version (3.11.14)
`pyproject.toml` declares `python_requires >= 3.11` with no upper bound, but the CI matrix pins one patch version. No signal if the package breaks on 3.12/3.13, which matters given how fast the PyTorch/Lightning ecosystem moves.

### 7. Accumulated "why do we need this?" TODOs
42 TODO/FIXME markers, several of which flag genuine uncertainty about existing logic rather than pending work — e.g. `abstracts.py:721` (`# @TODO: Do we really need this ?!`), `knowledge_graph.py:132` (`# TODO:CD: Why do we need to create this inverse mapping at this point?`), and `query_generator.py:226` (`# @TODO: Why do we need a deep copy here ?`). These represent institutional knowledge at risk of being lost; worth a documentation/cleanup pass before the original authors' context fades further.
