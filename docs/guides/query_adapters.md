# Query answering with frozen KGEs and KGFMs

Ordinary entity-prediction KGEs, ULTRA, TRIX, and Flock use the same query
evaluator. It supports the nine positive shapes and `2in`, `3in`, `inp`, `pin`,
and `pni`. Relation-prediction models such as TRIXRelation and FlockRelation are
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
`a=2**tanh(x@u)` and `b=4*tanh(x@v)`. Zero weights give the identity transform.
Available feature modes are:

| Mode | Parameters | Features |
|---|---:|---|
| `global` | 2 | Constant; learned global temperature and bias |
| `context` | 8 | Constant and normalized query/head/relation degrees |
| `context_scores_v1` | 16 | Context plus complete-row mean, entropy, top-score gap, observed-score contrast |

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
backbone scoring. It grounds `2i` and `3i` using the existing QueryGenerator,
computes complete answers on the source graph and easy answers on the masked
context, and requires at least one hard answer and one negative candidate.
Defaults are 96 training and 32 reserved-validation queries per shape per graph.
Query identity ignores intersection branch order, preventing split overlap.
Small graphs may not provide enough distinct queries: reduce the counts or load
prepared data. The bounded generator raises an error rather than silently
returning fewer examples.

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
observed memberships default to 1 during fitting. The final epoch is saved;
reserved validation does not select checkpoints.

Validation reports full-domain filtered MRR/Hits with average ties, separately
by source and shape, for sigmoid, sigmoid plus observed facts, and the fitted
adapter. Fit `global` separately to compare its two parameters against the
larger adapter. These are source-validation diagnostics, not evidence of
unseen-graph or all-shape superiority. Training covers 2i/3i only; fitted
adapters can be used for all supported inference shapes.

### Command line

```bash
python -m dicee.query_answering prepare --source source-context.json \
  --name source_graph --output source-queries.json

python -m dicee.query_answering fit --experiment Experiments/ultra-zero-shot \
  --data source-queries.json --output adapter-run --compare-global
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
adapter fitting on paths/negation, or benchmark-suite orchestration.
