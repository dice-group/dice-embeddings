# Logical-query benchmarks

The shared CQD engine can evaluate ULTRA, TRIX, Flock, and compatible saved
embedding models on the 23 datasets used by
[UltraQuery](https://arxiv.org/abs/2404.07198). This runner evaluates **CQD with
those backbones**. Native UltraQuery's multi-source GNN projection is not
implemented.

## Run a benchmark

Start with a small sample using an existing ULTRA link-prediction checkpoint:

```bash
python -m dicee.query_answering benchmark \
  --model ULTRA --checkpoint /path/to/ultra_3g.pth \
  --data-root /path/to/query-data --datasets WikiTopicsQuery:sci \
  --download --device cuda --beam-size 32 --max-queries-per-shape 5 \
  --output results/ultra-cqd-smoke.json
```

Remove `--max-queries-per-shape` for a complete split. The limit selects a
reproducible sorted prefix of each query shape, **not a representative sample**.
Reports explicitly record the limit and whether all queries and all 14 shapes
were evaluated. Use `--query-types 1p 2p 2i` to select particular shapes.

For representative subsets, add `--query-sampling uniform --sampling-seed 20260923`.
Selection uses a fixed SHA256 ordering within each dataset/query type, without
looking at answer labels. Limits of 50, 200 and 500 produce nested subsets with
16,100, 64,400 and 161,000 queries across the test suite. The same seed selects
the same queries for every backbone, adapter and execution order. Sampling uses
a separate seed from stochastic backbone inference. Graphs, candidates and
filtering remain complete. Use validation subsets for tuning and reserve test
subsets for confirmation. Small per-type samples can have substantial variance.

The Python APIs accept `query_sampling='uniform'` and `sampling_seed=20260923`.
An optional `on_query(query, shape, metrics)` observer receives newly evaluated
queries for paired statistical comparisons; completed queries skipped on resume
are not replayed to the observer. Persist those records alongside checkpoints.

Use `--datasets all` for all 23 datasets, or pass several names separated by
spaces. `--split test` is the default; `--split valid` selects validation.
Progress is saved atomically every 500 queries. Repeat the same command to
resume; completed queries are verified and skipped. Checkpoints live beside
the report in `<output>.checkpoints/`. Keep that directory with the report.
`--checkpoint-every` changes the interval. A single writer owns each run.
Changed data, labels, weights, code or inference settings require a new output
path. The JSON contains completed datasets and a separate `current_result`.

Replace `--model ULTRA` with `TRIX` or `Flock` and provide the matching checkpoint.
Optional `--model-config config.json` supplies model constructor settings, such
as nondefault dimensions or Flock walk counts. `--samples` and `--seed` control
Flock inference. No target-dataset fitting takes place.

The runner retains up to 512 MiB of device score rows per dataset by default.
`--cache-mb` changes this limit; zero disables row caching. Allow memory for
the backbone and its temporary tensors as well. A larger cache helps only
when it avoids repeated scoring.

`--row-batch-size` controls atomic batches and defaults to eight. ULTRA/TRIX's
neural microbatch follows it unless `--backend-batch-size` overrides it. Flock
keeps its model microbatch default of one unless explicitly overridden.
Reports record both effective sizes. Tune them on the intended graph and GPU;
larger batches can change floating-point rounding and use more memory.

The default `--query-order relation` groups compatible projections for cache
reuse. `--query-batch-size 32` prefetches shared anchor rows across 32 queries.
These settings use query structure only. Smoke-test selection occurs before
reordering. Use `--query-order published --query-batch-size 1` to retain the
original processing order. Final ranking keeps the reference tie protocol.

The CLI defaults to `--precision ieee --threads 2`. `--precision tf32` is an
explicit alternative, recorded in reports and checked on resume.

Flock uses independent per-row seeds and single-sample walk draws, then batches
neural scoring with bounded walk prefetch. This preserves historical walks
for `samples=1` and for multi-sample runs previously using microbatch one.
Historical multi-sample runs with larger sampling batches use a different
draw order. The report records the sampling policy; such runs cannot be mixed.

Alternatively, `--experiment /path/to/saved-experiment` loads a DICE experiment.
For transductive embedding models, the saved entity and relation mappings must
exactly match the published benchmark IDs; equal vocabulary sizes alone are
insufficient. Graph foundation models receive each dataset's inference graph
automatically, and the original graph is restored after evaluation.

A fitted adapter can be supplied with `--adapter adapter.json`. Without one,
the runner uses sigmoid memberships with no observed-fact overrides. Set
`--observed-mix 1` to evaluate the observed-fact variant. If an adapter is
provided, its observed-mix setting applies; conflicting overrides are rejected.
The report identifies the adapter, backbone fingerprint, seed, beam size,
precision settings, and graph/data fingerprints.

## Published data and graph splits

Dataset loading follows the
[official loader at the pinned reference commit](https://github.com/DeepGraphLearning/ULTRA/blob/427966ad8ed60420eef034063d44f3153addff90/ultra/datasets_query.py).
`--data-root` is the parent directory containing these extracted folders:

| Dataset names | Folder below data root | Scoring graph / candidates |
|---|---|---|
| `FB15k237LogicalQuery`, `FB15kLogicalQuery`, `NELL995LogicalQuery` | `FB15k-237-betae`, `FB15k-betae`, `NELL-betae` | Training graph for both splits; complete published entity vocabulary |
| `InductiveFB15k237Query:106` (also `113`, `122`, `134`, `150`, `175`, `217`, `300`, `550`) | The version, e.g. `106` | Training graph plus the selected inference split; rank only entities present in that combined graph |
| `WikiTopicsQuery:art` (also `award`, `edu`, `health`, `infra`, `loc`, `org`, `people`, `sci`, `sport`, `tax`) | `WikiTopics_QE/<topic>` | Training graph for validation; separate test inference graph for test |

Archives can be downloaded automatically with `--download` or extracted
manually. The official sources are
[BetaE](https://snap.stanford.edu/betae/KG_data.zip),
[inductive FB15k237](https://zenodo.org/records/7306046), and
[WikiTopics](https://reltrans.s3.us-east-2.amazonaws.com/WikiTopics_QE.zip).
The archives are large: approximately 1.36 GB for BetaE, 274 MB for WikiTopics,
and hundreds of MB per inductive split, before extraction. Pickle files must
come from these trusted sources. The runner does not regenerate queries or
answer labels.

Entity/relation IDs remain as published. Reciprocal relation pairs are explicit:
adjacent IDs for BetaE, offset IDs for inductive FB15k237, and the original
relation mapping for WikiTopics when available. The WikiTopics mapping matters
because unused relation IDs can make the observed vocabulary shorter than the
full vocabulary. The loader verifies that reciprocal handling leaves the
supplied graph unchanged. Target prediction edges and hard-answer labels never
enter the scorer's context graph.

## Metrics and comparability

For each hard answer, the evaluator filters all other easy and hard answers.
It averages MRR and Hits@1/3/10 over hard answers within each query, then over
queries within each shape. Dataset summaries give equal weight to each shape:
9 EPFO shapes and 5 negation shapes. Suite summaries give equal weight to each
dataset, with separate transductive, inductive-entity, and inductive-entity-and-
relation summaries. Missing groups are `null`, not zero.

The default `--tie-policy sort` follows the official evaluator's descending
PyTorch sorting before filtering. Exact ties can depend on device and PyTorch
version. `average`, `optimistic`, and `pessimistic` are explicit alternative
policies and are recorded in the report. Candidates outside a restricted domain
are excluded even when eligible answers have score negative infinity.

CQD scores are ranked in float64 log-membership space, avoiding artificial ties
from exponentiating tiny values. `--beam-size` controls intermediate search;
final answers are ranked over the full eligible domain. Candidate restrictions
apply to final ranking, not intermediate states. A full-domain beam removes
CQD pruning, but can be expensive: graph backbones score each retained head
separately. Record the beam when comparing runs.

The supported shapes are `1p`, `2p`, `3p`, `2i`, `3i`, `ip`, `pi`, `2u`, `up`,
`2in`, `3in`, `inp`, `pin`, and `pni`. `2u`/`up` load the published DNF forms;
De Morgan duplicates are excluded.

The separate faithfulness evaluation is available as
`InductiveFB15k237QueryExtendedEval:<version>`. It evaluates the published
training queries with their extended-graph answers, using the original query
order to associate answer lists before sorting. Its reports are labeled
`faithfulness`, and it is not part of `--datasets all`.

This runner implements the paper's ranking benchmark and the extended-graph
faithfulness ranking evaluation. The additional ROC-AUC, answer-cardinality
MAPE, and Spearman analyses are not included.

## Python API

```python
from dicee.query_answering import load_benchmark, benchmark_model

benchmark = load_benchmark("/path/to/query-data", "WikiTopicsQuery:sci")
report = benchmark_model(model, benchmark, beam_size=32, max_queries_per_shape=5)
```

`evaluate_benchmark(benchmark, predict)` also accepts any callable that returns
a complete higher-is-better score vector for an indexed query. This allows
future query executors to share the dataset loaders, filtering, and reporting.
Only the query is passed to that callable; labels remain inside the evaluator.
