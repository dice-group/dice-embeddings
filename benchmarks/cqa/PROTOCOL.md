# Evaluation protocol

This document states how the [benchmark suites](README.md) score methods and
where this evaluation knowingly differs from a publication. Where the evidence
for a published number is uncertain, the difference is recorded rather than
resolved by tuning.

## Metrics

Each query has easy answers, which follow from the observed graph, and hard
answers, which need at least one missing link. MRR and Hits@k rank every hard
answer against all candidate entities after filtering every other easy and hard
answer. Scores average hard answers per query, queries per query type, and
query types equally within a dataset group; suite means weight datasets
equally. Each result states whether it covers the full split and every query
type of its dataset.

Ties are scored twice from the same predictions: `sort` follows PyTorch's
default descending argsort, as the reference evaluator does. That sort is not
guaranteed stable, so exactly tied scores can order differently on another
device or library version. `expected` is the exact expectation of each rank
under uniformly random ordering within a tied block, which removes that
dependence. `expected` is not the midpoint rank. Rank traces store the sort
rank and the tie block of every hard answer, so both policies can be recomputed
without inference.

Each suite keeps its released candidates and inference graphs. UltraQuery
evaluates its 14 query types on the published inference graphs: the training
graph for transductive datasets and the released inference graphs for the
inductive ones. Test triples never enter inference in either suite.

## +H answer filters

The `benchs-1.0` archive has two sets of test answers for the same queries: the
root `test-{easy,hard}-answers.pkl` files and the `test-query-reduction/<type>/all/`
files, which the pinned
[analysis script](https://github.com/april-tools/is-cqa-complex/blob/d1ce74164936a7c09d9147e83190da047cb39429/read_queries_pair.py)
recomputes from the graph:

| Dataset | Type | Root hard answers | Reduction `all` hard answers |
|---|---|---:|---:|
| FB15k-237+H | 3in | 20,000 | 192,617 |
| FB15k-237+H | pin | 20,000 | 197,728 |
| FB15k-237+H | inp | 20,000 | 201,328 |
| NELL995+H | 3in | 20,000 | 126,893 |
| NELL995+H | pin | 20,000 | 133,156 |
| NELL995+H | inp | 20,000 | 128,812 |

For `2in` and `pni` the hard answers agree, but the root easy-answer filters
are wrong for every FB15k-237+H and NELL995+H query: 18,000 queries with 149,691
false filtered answers in total. All hard answers are true, and the other
negated types and ICEWS18+H agree with full-graph truth. The pinned authors'
`achieve_answer` gives the same result on all 52,413 checked queries.

The **corrected** filter, the +H default, takes the easy answers of `2in` and
`pni` on FB15k-237+H and NELL995+H from the authors' `2in/all` and `pni/all`
files and keeps every root hard answer. An evaluation of the test split first
checks those files against the complete graph. Every prediction is also scored
with the **released** filters as a control. Test facts are used only for this
check. On all FB15k-237+H `2in` queries, a CPU QTO diagnostic with train+valid
facts scores 16.11 MRR with released and 10.59 with corrected filters, against
the published 10.6; with training facts only, 13.16 and 9.62.

## +H inference graphs

Graph-based methods (GNN-QE, UltraQuery, QTO, CQD-Hybrid, ULTRA and TRIX)
answer test queries over train+valid facts by default, the observed graph from
which the released test difficulty groups were built. Validation always uses
training facts. The released GNN-QE, UltraQuery and QTO loaders use training
facts for these transductive graphs, and the +H CQD-Hybrid loader uses
train+valid for FB15k-237+H and NELL995+H but skips observed filters for
ICEWS18+H. Both graph conditions can be compared directly (see the README's
ablations); the `reference` field of each recipe records the setting of the
released loader.

## Operators

[Appendix F](https://arxiv.org/html/2410.12537v3#A6) of the +H paper specifies
the minimum t-norm for FB15k-237+H `2u`. With it, full-query CPU diagnostics
give CQD 27.77 and CQD-Hybrid 27.77 MRR; with the product t-norm, 40.05 and
42.17, against the published 40.1 and 42.2. The recipes therefore use product
for this query type. This is an inferred documentation mismatch, not proof of
the original command. The substitution does not explain `up`: product gives
7.88 (CQD) and 9.33 (CQD-Hybrid), below the minimum t-norm's 9.15 and 11.06 and
the published 10.6 and 12.0, so `up` keeps the documented minimum and remains
unreproduced.

## Query batches

The scores of ConE, CLMPT, GNN-QE and UltraQuery can depend on the queries that
share a batch. Their recipes keep the batch sizes and the query order of the
released code (`query_order: upstream-pickle`), and a sampled query is scored
within its complete batch.

The bounded CQD profile scores each stage in smaller batches. Its scores can
differ in rounding from the dense batches of the released code.

## Opt-in variants

Two variants are not published methods. UltraQuery with `observed_traversal`
also takes, at every projection, the maximum with an exact traversal of the
observed facts from the same fuzzy set. UltraQuery's released weights used as a
frozen ULTRA backbone (`relation_conditioning: query`) condition relation
reasoning on the queried relation, as UltraQuery's projection does; ULTRA's link
prediction conditions an inverse tail query on its direct relation instead.
With that setting the frozen backbone equals UltraQuery's single-source
projection for every relation (tests/test_query_baselines.py).

## Reproducibility

Inference is float32 with IEEE matrix products, no autocast and deterministic
algorithms. Message passing in ULTRA, TRIX, UltraQuery and GNN-QE reduces each
node's edges in a fixed order, so scores do not depend on thread scheduling.
The progress of a run is bound to its recipe, checkpoint, dataset, query order
and the query answering source code. A resumed run must match all of them;
completed entries are never recomputed.

Matrix products in the backbones are not batch-invariant: the same query can
receive different last bits in a different atomic batch. The KGFM adapters
reuse backbone rows through caches, so cache budgets, batch sizes and the
point at which an interrupted run resumes with empty caches can change the
last bits of later scores. Batch and cache settings are therefore part of a
recipe and recorded with each result. A batch-invariant backbone would remove
this dependence.

## Upstream evaluation edge case

UltraQuery's `batch_evaluate` masks excluded candidates with `-inf`, yet can
still count them ahead of a hard answer whose score is also `-inf`: with six
`-inf` scores, candidates `[0, 2, 4]` and hard answer 2, it returns rank 3
where the candidate-only rank is 2. The shared evaluator ranks candidates only.
Native UltraQuery returns finite logits, so this does not affect its published
results.
