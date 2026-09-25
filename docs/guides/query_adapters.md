# Query answering with frozen KGEs and KGFMs

Ordinary entity-prediction KGEs, ULTRA, TRIX, and Flock use the same query
evaluator. It supports the nine standard positive shapes, the +H `4p` and `4i`
shapes, and `2in`, `3in`, `inp`, `pin`, and `pni`. Relation-prediction models such as TRIXRelation and FlockRelation are
not entity query scorers.

## Named and indexed queries

```python
from dicee import KGE

kge = KGE(path="Experiments/ultra-zero-shot")
answers = kge.answer_multi_hop_query(
    "2p", ("entity_a", ("relation_r", "relation_s")),
    k=10, beam_size=64,
)
```

`k` limits returned answers, while `beam_size` limits intermediate entities at
each projection. Without `beam_size`, it defaults to `max(1, k)`; both counts are
capped by the entity domain when selecting entities. `k=0` returns no named
answers. `only_scores=True` always returns the complete entity vector in numeric
ID order. Ranked results break ties by numeric entity ID, regardless of mapping
insertion order. `queries=[...]` answers several queries of the given type.

For models loaded directly with `load_pretrained` and `set_graph`, no experiment
folder is needed:

```python
from dicee.query_answering import QueryAnswerer

answerer = QueryAnswerer(kge.model, row_batch_size=8)
memberships = answerer.predict((0, (0, 1)), beam_size=64)
log_memberships = answerer.predict((0, (0, 1)), return_log_scores=True)
print(answerer.last_info)  # row counts, cache hits, and pruning flags
```

Indexed query tuples use `-2` for negation and a final `(-1,)` branch for union.
Named tuples use `"not"`/`"n"` and `"union"`/`"u"` markers in the same positions.
For example, `pni` is `((a, (r, s, "not")), (b, (t,)))`: complement the entire
two-hop path, then intersect with the second atom.

The engine selects the global top-k **composed prefix** at each projection.
All final entities are scored. A beam covering the complete entity domain gives
exhaustive projection; smaller beams approximate existential quantification.
Complementing a pruned path reverses its approximation direction, reported as
`negated_pruning`.

Composition uses float64 log-memberships for numerical stability, with product
AND, probabilistic-sum OR, complement NOT, and max existential aggregation.
`tnorm="min"` selects min/max AND/OR. Sugeno and Yager negation remain available;
Sugeno requires `lambda_ > -1`, Yager `lambda_ > 0`.

## Observed facts and learned transforms

Graph models automatically provide their attached inference context. For a
transductive predictor, supply a `QueryContext` containing observed triples in
that model's public vocabulary:

```python
from dicee.query_answering import QueryContext

context = QueryContext(
    triples=[(0, 0, 1), (1, 2, 3)],
    num_entities=kge.model.num_entities,
    num_relations=kge.model.num_relations,
    inverse_relations=((0, 1), (2, 3)),
)
# For a graph model this must match its attached graph.
answerer = QueryAnswerer(kge.model, context=context, observed_mix=1.0)
```

Inverse pairs are explicit public IDs, with no assumption about adjacent or
even/odd IDs. The context deduplicates facts and adds those reciprocal edges.
Keep held-out validation/test facts out of inference context and its features.

The default is a sigmoid transform with no observed override. With
`observed_mix=gamma`, context edges receive `gamma + (1-gamma)*p`. At `gamma=1`,
positive answers proven entirely by the context are also restored to membership
1 even if beam search misses their witness. This additional proof preservation
does not apply to queries containing negation.

```python
from dicee.query_answering import QueryScoreAdapter

adapter = QueryScoreAdapter.load("my-adapter.json", model=kge.model)
answerer = QueryAnswerer(kge.model, adapter=adapter)

# Load a catalog entry and verify its original backbone checkpoint.
adapter = QueryScoreAdapter.load(
    "/path/to/adapter-catalog.json",
    key="all_sources", model=kge.model,
    checkpoint="/path/to/ultra_3g.pth",
)
```

Imported catalogs require the original checkpoint file. The loader verifies
both its recorded SHA256 and equality of its tensor state to the loaded model.
New adapter files bind directly to the backbone's state fingerprint. Changed or
fine-tuned weights fail verification. Vocabulary sizes may change for graph
models without changing their transferable parameters.

The learned formula is `sigmoid(a*z+b)`, with
`a=2**tanh(x@u)` and `b=bias_bound*tanh(x@v)`, with `bias_bound=4` by default.
Zero weights give the sigmoid baseline unless row normalization is enabled.
`scale_bound=C` permits scales between `1/C` and `C`, with `C=2` by default.
Other bounds use `a=exp(log(C)*tanh(log(2)/log(C)*(x@u)))`, preserving the
initial derivative across bounds. `scale_bound=None` uses `a=exp(log(2)*(x@u))`
without clipping; nonfinite scales fail explicitly. The CLI accepts
`--scale-bound none`. These are score adaptations, not probability guarantees.
Available feature modes are:

| Mode | Parameters | Features |
|---|---:|---|
| `global` | 2 | Constant; learned global temperature and bias |
| `context` | 8 | Constant and normalized query/head/relation degrees |
| `context_scores_v1` | 16 | Context plus complete-row mean, entropy, top-score gap, observed-score contrast |

`normalization="standard"` centers each complete score row and divides by its
standard deviation (floored at `1e-6`). Features still describe the original row.
`hidden_dim=16` adds a small tanh network over the same features; it predicts
the positive scale and bias without entity/relation ID embeddings. Its output
layer starts at zero, with locally seeded hidden weights. Version 3 artifacts
also record the scale bound; versions 1 and 2 retain the original bound of two.

Transforms preserve mathematical ordering among unobserved tails within a row but may
change composed rankings. They are fuzzy memberships, not calibrated
probabilities. All these options also apply to transductive predictors when
the required context is supplied; imported KGFM weights are not transferable
to arbitrary predictors.

## Prepare source training queries

```python
from dicee.query_answering import prepare_adapter_data

# source_context contains source-training facts, not target evaluation facts.
data = prepare_adapter_data(source_context, name="source_graph", seed=2026090851)
data.save("source-queries.json")
```

Preparation masks 30% of fact pairs, removing both reciprocal directions before
backbone scoring. By default it grounds `2i` and `3i` using the existing QueryGenerator,
computes complete answers on the source graph and easy answers on the masked
context, and requires at least one hard answer and one negative candidate.
Defaults are 96 training and 32 reserved-validation queries per shape per graph.
Pass `shapes=("2i", "3i", "2in", "3in")` to include negation. Negated queries
also require a context false positive: masking a negated fact can introduce an
incorrect answer. These false positives remain training negatives; only complete
source answers are filtered out. Query identity ignores branch order, preventing
split overlap.
All 14 original benchmark shapes are supported, including paths and unions.
`train_counts` can specify separate pool sizes by shape; fitting selects balanced
prefixes from these pools without using validation examples.
Small graphs may not provide enough distinct queries: reduce the counts or load
prepared data. The bounded generator raises an error rather than silently
returning fewer examples.

To enlarge a training set while preserving its context, validation split, and
existing training order, pass `extend=data` with the same source, seed, mask,
name, and shapes. Generate the largest pool once and use `train_per_shape` in
the fitter for nested subsets.

For prepared examples, construct `AdapterTrainingData(name, context, train,
validation)` with `AdapterQuery(query, answers)` entries. `answers` is the complete
source answer set; easy/hard sets are derived from context. `save`/`load` use
versioned JSON. The trainer checks domain membership, split disjointness, and
consistency with observed context answers. Prepared data remains responsible
for correct complete source supervision.

## Fit and compare adapters

```python
from dicee.query_answering import fit_query_adapter

result = fit_query_adapter(
    kge.model, [data], cache_dir="adapter-run/score-banks",
    epochs=20, batch_size=8,
)
result.adapter.save("adapter-run/context-adapter.json")
print(result.validation)

global_baseline = fit_query_adapter(
    kge.model, [data], feature_mode="global",
    cache_dir="adapter-run/score-banks",
)
```

One adapter is fitted for a frozen backbone. Graph models can switch between
multiple named source vocabularies; transductive sources must already use the
fixed backbone vocabulary. The caller's graph, weights, gradients, and training
mode are preserved. Only adapter parameters receive optimization updates.

Unique complete atomic rows are collected once, detached, and kept on CPU.
For multi-hop queries, the trainer shares the inference executor and selects
the beam again after adapter updates (`beam_size=64` by default). Gradients flow
through retained scores, not the discrete top-k indices. Rows are fetched as
needed from a persistent SQLite cache, with 128 MiB of prepared CPU rows and
8 GiB of stored raw-row payload per source/backbone. Eviction recomputes missing
rows without changing scoring. Each query backpropagates separately within its
optimizer batch to bound activation memory. Backbone microbatches remain small.
Optional disk banks verify row checksums and bind to backbone weights, graph,
query conditions, sampling, runtime, and precision settings. Changing these
settings creates a different bank. Banks include validation *rows* for reporting
but never feed validation answers into optimization. Flock uses a local seed
per graph/head/relation/sample count, independent of row order and outer query
batching, without modifying caller RNG state. Changing its internal sampling
configuration changes the recorded bank identity.

The **new** objective averages, for each hard answer `a`,
`log(1 + sum_negative exp(log_score_negative - log_score_a))`.
Negatives exclude every complete source answer. Query losses are weighted
equally across source/shape cells. Defaults are Adam at 0.02, 20 epochs, batch
size 8, `0.001*mean(weights**2)` regularization, and gradient clipping at 1.
The default feature mode is `context_scores_v1` for ULTRA and `context` otherwise;
observed memberships default to 1 during fitting. The final epoch is saved by default. `validation_every=5` instead selects the
epoch with the best reserved source-validation macro MRR. `training_sources`
and `validation_sources` support holding out whole source graphs; target
benchmark answers are never inputs to fitting. `train_shapes` and
`train_per_shape` allow matched training budgets over a shared score bank.

Set `epochs=500`, `validation_every=5`, and `early_stopping_patience=100` for
a 500-epoch ceiling with conservative early stopping. Patience counts epochs
since the best source-validation MRR; any strict improvement resets it. Checks
happen at validation intervals, and ties do not reset patience. The best
checkpoint is restored whether patience or the ceiling ends training. Reports
record completed epochs, optimizer steps, the selected epoch, and the stop
reason. Early stopping requires reserved source validation.

Validation reports full-domain filtered MRR/Hits with average ties, separately
by source and shape, for sigmoid, sigmoid plus observed facts, and the fitted
adapter. Fit `global` separately to compare its two parameters against the
larger adapter. These are source-validation diagnostics, not evidence of
unseen-graph or all-shape superiority. `validation_shapes` controls which source
query types enter checkpoint selection.

### Query-type and scale studies

`dicee/scripts/benchmark_query_adapters.py --study query-types` compares 2, 4,
10, and 14 training shapes. Each variant uses 840 training queries across three
sources, bias bound 8, and scale bound 2. All variants use the same 10-type
source-validation set (16 queries/type/source); `ip`, `pi`, `2u`, and `up`
are excluded from checkpoint selection. The 14-type arm additionally trains
on these structures. Source-holdout fits use 560 training queries on two sources.

`--study scale --reuse-from QUERY_TYPE_RUN` compares scale bounds 2, 4, 8,
and unrestricted scaling on the two-type setup. It reuses the completed bound-2
baseline, identical source pools, and raw-row caches. Both presets keep a
500-epoch ceiling, validation every 5 epochs, patience 100, and restore the best
source checkpoint. Target evaluation uses the same 50 validation queries/type
on all 23 datasets. `--after-run RUN --prepare-only` prepares a frozen run;
it requires a separate queue launcher and does not start a background process.

### Command line

```bash
python -m dicee.query_answering prepare --source source-context.json \
  --name source_graph --output source-queries.json

python -m dicee.query_answering fit --experiment Experiments/ultra-zero-shot \
  --data source-queries.json --output adapter-run --compare-global \
  --epochs 500 --validation-every 5 --early-stopping-patience 100
```

`source-context.json` contains the fields of `QueryContext.to_dict()`:
`triples`, `num_entities`, `num_relations`, and optional `inverse_relations`.
Fitting writes adapter JSON files, a validation/training report, and reusable
score banks. Checkpoints and datasets are not downloaded by these commands.

## Migration from the old evaluator

- `use_logits` now defaults to `False`. Set `True` explicitly for legacy raw-score
  algebra; it cannot be combined with adapters or observed overrides.
- Every query shape respects `k`; use `only_scores=True` for the complete vector.
- `inp`, `pin`, and `pni` no longer fail on atomic row dimensions.
- `pni` negates the completed path instead of the last edge before projection.
- Membership composition stays on the scoring device. Named results rank in log
  space before converting to memberships, avoiding underflow-induced ties.
- `beam_size` separates search accuracy from the number of displayed answers.

The implementation does not add native UltraQuery, per-branch CQD search,
or adapter fitting on paths. For evaluation on published datasets,
see the [logical-query benchmark runner](query_benchmarks.md).

## Inference caching

Keep one `QueryAnswerer` alive and put the backbone in `eval()` mode to reuse
complete atomic score rows across queries. `cache_bytes` bounds the retained
rows on the scoring device (64 MiB by default); for example,
`cache_bytes=512 * 2**20` allows a 512 MiB cache. `last_info` reports per-query
`raw_rows` and `cache_hits`, and the total retained `cache_bytes`.

`KGE.answer_multi_hop_query()` also retains its engine across compatible calls.
Use `kge.clear_query_cache()` to release it. Moving the KGE to another device
releases cached rows. `answerer.prefetch(queries)` batches unique anchor rows
before scoring a group of queries; it receives no answer labels.

The engine checks graph identity and tensor mutation versions, adapter weights
and settings, scoring precision, and Flock's sampling configuration. Changes
invalidate the cache. Backbone hashes are rechecked when tensor state changes,
not for every query. Training-mode callers, autocast, and tensors created under
`torch.inference_mode()` do not retain rows across predictions. Returned scores
can be modified without modifying cached rows.

Treat `QueryContext` and its derived lookup tables as immutable. Replace the
context when facts change. Use normal PyTorch updates (`copy_`, optimizers,
`load_state_dict`) rather than `.data` writes, which bypass mutation tracking.
Call `answerer.clear_cache()` after changing custom non-tensor scoring behavior
or to release retained device memory. Bound adapters still reject modified
backbone weights. Changes through `.data` are unsupported for cached inference.

Beam projection combines each row batch together while retaining stable prefix
selection, the selected beam width, and float64 composition. GPU backbones may
already have nondeterministic reductions; caching reuses a previously computed
row rather than drawing another floating-point realization of that row.

## Adapter screening

For a training-size ablation, reuse a completed sweep's wider-bias baseline:

```bash
python -m dicee.scripts.benchmark_query_adapters \
  --output Experiments/adapter-data-scaling \
  --data-scaling-from Experiments/adapter-screen \
  --training-multipliers 4 16 --prepare-only
```

This snapshots a queued configuration. Run the same output without
`--prepare-only` after the preceding sweep completes. The runner checks the
predecessor's lock, completion, and baseline checksums before proceeding.
Training subsets are nested; source validation and target evaluation queries
stay fixed. Each size uses the same epoch ceiling and early-stopping settings;
larger sets have more optimizer updates per epoch. The current preset uses `context_scores_v1`, observed
facts, and bias bound 8, with the same three source holdouts.

```bash
python -m dicee.scripts.benchmark_query_adapters \
  --root . --output Experiments/query-benchmarks/adapter-screen \
  --size 50 --epochs 500 --validation-every 5 --early-stopping-patience 100 \
  --backbones ultra trix
```

The sweep uses local ULTRA/TRIX checkpoints and the three source training graphs.
It compares positive versus negation supervision, bias bounds of 4/8, the three
feature modes, row normalization, and a 16-unit MLP, plus unfitted baselines.
Each fit uses 192 queries per source, with three additional source-holdout fits.
All target evaluations use the same uniform validation subset and beam size 64.
Target metrics are descriptive and do not select adapter checkpoints.

The output contains a code snapshot, hashes, source queries, per-epoch reports,
adapters, per-query benchmark traces, and a growing `comparison.md`. Workers
release GPU memory between jobs and resume from checkpoints. Repeat the command
with the same output directory to resume its recorded configuration and code.
This first screen uses one training seed; confirm close results with more seeds
and larger samples before drawing conclusions.
