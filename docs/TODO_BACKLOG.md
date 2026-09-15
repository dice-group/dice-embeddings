# TODO/FIXME Backlog

This document catalogs and organizes the 42 TODO/FIXME markers scattered throughout `dicee/` (45 including `tests/`), per `grep -rn "TODO\|FIXME" dicee/`. Keeping them in one place prevents institutional knowledge loss and improves maintainability.

**Last Updated:** 2026-07-20

---

## Architectural TODOs

### Query Generator Uncertainty
- **File:** `dicee/query_generator.py:40`
  - **Marker:** `"2i": [['e', ['r']], ['e', ['r']]], # @TODO: double check the evaluation`
  - **Issue:** Evaluation logic for 2-intersection queries not verified
  - **Priority:** Medium
  - **Status:** Open

- **File:** `dicee/query_generator.py:157-168`
  - **Marker:** Multiple `@TODO: Document the code`, `@TODO: unclear`
  - **Issue:** Query generation logic lacks documentation
  - **Priority:** Low
  - **Status:** Open
  - **Action:** Add inline documentation explaining query construction

- **File:** `dicee/query_generator.py:195, 224, 229`
  - **Marker:** `@TODO: Explain why this is needed`, `Incorrect reasoning: It can enter an infinite loop`, `Why do we need a deep copy here?`
  - **Issue:** Three separate concerns about copy semantics and loop behavior
  - **Priority:** Medium
  - **Action:** Code review + documentation

### Knowledge Graph Mapping Logic
- **File:** `dicee/knowledge_graph.py:132`
  - **Marker:** `# TODO:CD: Why do we need to create this inverse mapping at this point?`
  - **Issue:** Inverse relation mapping creation timing unclear (performance vs. correctness concern)
  - **Priority:** Medium (could be performance optimization)
  - **Status:** Open
  - **Action:** Investigate if we can defer mapping creation or cache it

### Graph Embeddings API
- **File:** `dicee/knowledge_graph_embeddings.py:708`
  - **Marker:** `# @TODO: refactor by torchargmax(aggregated_query_for_all_entities)`
  - **Issue:** Potential refactoring for multi-hop query aggregation
  - **Priority:** Low
  - **Status:** Code improvement candidate

- **File:** `dicee/knowledge_graph_embeddings.py:1399`
  - **Marker:** `# TODO :Should we initialize self.literal_model in __init__ ?`
  - **Issue:** Lazy initialization of literal value prediction model
  - **Priority:** Low
  - **Status:** Design decision needed

---

## Model Implementation TODOs

### Clifford Algebra
- **File:** `dicee/models/clifford.py:54`
  - **Marker:** `# TODO:Do we need coefficients for the real part ?`
  - **Issue:** Whether to include separate coefficients for real component in Clifford embeddings
  - **Priority:** Medium (affects model expressivity)
  - **Status:** Open
  - **Action:** Run ablation study

### Ensemble Models
- **File:** `dicee/models/ensemble.py:21`
  - **Marker:** `# TODO: Why we cant send the compile model to cpu ?`
  - **Issue:** Compiled model device transfer issue with PyTorch compilation
  - **Priority:** Medium
  - **Status:** Likely PyTorch version-dependent

### Real Embeddings (CoKE)
- **File:** `dicee/models/real.py:537, 589`
  - **Marker:** 
    - `block_size: int = 3  # triples -> TODO: LF: for multi-hop this needs to be bigger`
    - `pos_ids = torch.arange(0, 3, device=device)  # (3,) -> TODO: LF: here 3 has to change according to vocab size (in case we want multi-hop)`
  - **Issue:** Positional embeddings hardcoded to 3 (entity-relation-entity), but multi-hop requires flexibility
  - **Priority:** Medium (blocks multi-hop CoKE training)
  - **Status:** Open
  - **Action:** Parameterize block_size based on query type

### ADOPT Optimizer
- **File:** `dicee/models/adopt.py:235, 241`
  - **Marker:** 
    - `# TODO: support fused`
    - `# TODO(crcrpar): [low prec params & their higher prec copy]`
  - **Issue:** Fused optimizer kernel support and mixed-precision tracking
  - **Priority:** Low (optimization, not correctness)
  - **Status:** Open

### Transformer Models
- **File:** `dicee/models/transformers.py:252`
  - **Marker:** `# not 100% sure what this is, so far seems to be harmless. TODO investigate`
  - **Issue:** Unclear code comment suggests unverified behavior
  - **Priority:** Low
  - **Status:** Code review needed

---

## Training & Utility TODOs

### Training Loop
- **File:** `dicee/trainer/dice_trainer.py:288, 399`
  - **Marker:** 
    - `# TODO: Here we need to load memory pag`
    - `# TODO: Later, maybe we should write a callback to save the models in disk`
  - **Issue:** Memory page loading and periodic model checkpointing
  - **Priority:** Low
  - **Status:** Open

- **File:** `dicee/trainer/model_parallelism.py:129, 209`
  - **Marker:** 
    - `# TODO: Later, maybe we should write a callback to save the models in disk`
    - `# () TODO: Pytorch Bug https://github.com/pytorch/pytorch/issues/58005`
    - `# () TODO: test_queries has keys that are tuple ,e.g. ('e', ('r',))`
  - **Issue:** Checkpointing callback, PyTorch bug tracking, test data structure handling
  - **Priority:** Low
  - **Status:** Open

### Data Serialization
- **File:** `dicee/static_funcs.py:419, 420, 741, 777, 786-787`
  - **Marker:** 
    - `# TODO: CD: We do not need to keep the mapping in memory` (line 419)
    - `# TODO:CD: Deprecate the pickle usage for data serialization.` (line 420)
    - `# TODO:CD:Deprecate it` (line 741)
    - `# @TODO: CD: Renamed this function` (line 777)
    - `# @TODO: Dictionary keys do not need to be in order...` (lines 786-787)
  - **Issue:** Multiple data serialization and mapping memory concerns
  - **Priority:** Medium (affects memory footprint)
  - **Status:** Open
  - **Action:** Replace pickle with more efficient format; optimize mappings

- **File:** `dicee/static_funcs.py:252, 718`
  - **Marker:** 
    - `# @TODO: Could these funcs can be merged?` (line 252)
    - `# @TODO: This function should take any DASK/Pandas DataFrame or Series.` (line 718)
  - **Issue:** Function consolidation and data type flexibility
  - **Priority:** Low
  - **Status:** Refactoring candidate

### Abstracts & Base Classes
- **File:** `dicee/abstracts.py:206, 724`
  - **Marker:** 
    - `# @TODO: What to do ?` (line 206)
    - `# @TODO: Do we really need this ?!` (line 724)
  - **Issue:** Abstract method implementation unclear; potentially dead code
  - **Priority:** Low
  - **Status:** Code review needed

### Executor & Scripts
- **File:** `dicee/executer.py:322`
  - **Marker:** `# @TODO: Move to static funcs`
  - **Issue:** Code organization - function should be refactored into static_funcs.py
  - **Priority:** Low
  - **Status:** Refactoring opportunity

- **File:** `dicee/scripts/run.py:17, 18`
  - **Marker:** 
    - `# TODO: Deprecate --path_single_kg`
    - `# TODO: --dataset_dir either be a single KG file or a folder.`
  - **Issue:** CLI interface cleanup - consolidate overlapping options
  - **Priority:** Low (affects API design)
  - **Status:** Open
  - **Action:** Plan CLI v2

### Analysis
- **File:** `dicee/analyse_experiments.py:19`
  - **Marker:** `# TODO: features/columns for pandas dataframe`
  - **Issue:** Experiment analysis dataframe structure incomplete
  - **Priority:** Low
  - **Status:** Open

---

## Test TODOs

- **File:** `tests/test_regression_conex.py:83`
  - **Marker:** `# TODO: Write Test and compare ConEx with AConEx`
  - **Issue:** Missing regression test for variant comparison
  - **Priority:** Low
  - **Status:** Open

- **File:** `tests/test_regression_all_vs_all.py:33`
  - **Marker:** `# @TODO Investigate`
  - **Issue:** Unspecified investigation needed
  - **Priority:** Low
  - **Status:** Open (clarification needed)

- **File:** `tests/test_pickle.py:6`
  - **Marker:** `# TODO:CD: Using pickle is quite inefficient timewise. Why do we need the pickle models ?!`
  - **Issue:** Pickle model serialization inefficiency - should use PyTorch .pt files
  - **Priority:** Medium (affects test suite performance)
  - **Status:** Open
  - **Action:** Migrate pickle models to PyTorch format

---

## Documentation TODOs

- **File:** `dicee/query_generator.py:59, 162`
  - **Marker:** `@TODO: add description`, `@TODO: unclear`
  - **Issue:** Docstrings missing for query functions
  - **Priority:** Low
  - **Status:** Documentation task

---

## Summary by Priority

### 🔴 High Priority (Blocks features or affects correctness)
- None currently (improvements are all about quality/performance)

### 🟠 Medium Priority (Performance or design concerns)
- Query evaluation verification (query_generator.py:40)
- Inverse mapping timing optimization (knowledge_graph.py:132)
- Clifford real part coefficients ablation (clifford.py:54)
- CoKE multi-hop parameterization (real.py:537, 589)
- Data serialization migration (static_funcs.py:420)
- Pickle model format migration (test_pickle.py:6)

### 🟡 Low Priority (Nice-to-have improvements)
- Code organization & refactoring
- Documentation completeness
- Function consolidation
- CLI design updates

---

## Action Items for Next Sprint

1. [ ] Investigate inverse mapping timing in knowledge_graph.py:132
2. [ ] Plan CoKE multi-hop refactoring for real.py
3. [ ] Create migration plan for pickle → .pt models in tests
4. [ ] Document query generation logic in query_generator.py
5. [ ] Plan CLI v2 consolidation (dataset_dir vs path_single_kg)

---

## Maintenance Notes

- **Update Frequency:** Quarterly review recommended
- **Tool:** Use `grep -rn "TODO\|FIXME" dicee/ tests/` to regenerate this list
- **Format:** Prefer descriptive TODOs with issue context over vague markers
- **Best Practice:** Link to GitHub issues when available (`#410`, `#XXX`)
