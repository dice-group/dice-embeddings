# Evaluation protocol

This document states how the [ProofPFN benchmark](README.md) ranks a graph. The thesis runs used these rules, and the
golden tests compare the ranks with the rows that those runs recorded.

## Graph

| Step | Rule | Code |
|---|---|---|
| hash check | before a graph loads: the file SHA-256 of train.txt, valid.txt and test.txt where the manifest records them, and the content hash of train.txt and test.txt where it records one (SHA-256 of the sorted distinct triples; a line splits at tabs if it has one, else at whitespace) | `manifests.graph_problems` |
| ids | entities and relations are the sorted names over train, valid and test; the triples of each split keep their file order, duplicates included | `data.read_graph` |
| inference graph | train.txt; the model keeps its sorted unique triples plus technical inverses | `systems.attach` |
| test rows | `all`: every row. `first-1000`: all rows of a test split with at most 1,000 rows; else the first occurrence of every distinct test fact in file order is the pool and `sorted(random.Random(0).sample(pool, 1000))` is ranked. The row `index` is the position in test.txt | `data.ranked_positions` |
| filter | every fact of train, valid and test (unsampled test facts included), except the target | `data.known_sets` |
| query groups | `(head direction, relation, anchor)` with its test members in index order, stably sorted by relation | `data.query_groups` |

dicee's `KG` is not used for loading: its pandas reader turns tokens such as `NA` and `null` into missing values and
merges distinct entities (`tests/test_data.py`). The command `python -m dicee ... --eval_model test` also stops on a
graph with an empty valid.txt, which 26 of the 54 graphs of the standard suite have. On graphs that it reads, its
evaluator with `--eval_tie_policy pessimistic` gives the ranks of this benchmark.

## Scores and ranks

| Step | Rule | Code |
|---|---|---|
| models | built on the CPU, `query_batch_size` 1 (official Flock: its passes), `set_graph(train)` on the CPU, then moved to the device, evaluation mode, no gradients | `systems.attach` |
| scores | one call per query group: `forward_k_vs_all([[h, r]])` for a tail query, `forward_k_vs_all_heads([[r, t]])` for a head query (the first member's anchor), scores asserted finite | `evaluate.query_scores` |
| execution | deterministic algorithms with `CUBLAS_WORKSPACE_CONFIG=:4096:8` on CUDA, two CPU threads | `dicee.synthetic_kg.determinism` |
| view | `log_softmax(scores.double())` | `evaluate.rank_queries` |
| rank | pessimistic: `#{v >= v[target]}` over the candidates outside the filter, the target included | `FilteredRanker(tie_policy='pessimistic')` |
| metrics | MRR and Hits@1, 3, 10 over the rows of both directions, of the tail queries and of the head queries | `evaluate.metrics_of` |
| collection | the plain mean of the per-graph values; for the seeds of a name, the mean and the sample standard deviation over the seeds of the collection values | `evaluate.summarize` |

## Official Flock

The walk count of a graph is `2^round(log2(128 x HM(|V| / 24038, |E| / 181511.3)))`, clamped to 16..512, with `|V|` and
`|E|` the distinct entities and distinct triples of train.txt and HM the harmonic mean. The 54 graphs of the standard
suite take the walk count and the passes of the official script `scripts/entity_zeroshot.sh`. Other graphs take the
rule's walk count and 16 passes up to 128 walks, 4 passes for 256 and 2 for 512. The sampling seed is 42. The passes of
one query form one microbatch of the walk sampler (`--flock-passes-in-sequence` runs them one after another: the same
law, other walks). `check` prints the setting of each graph, and a result file records it as `flock_setting`.

## Worlds of the prior

| Step | Rule | Code |
|---|---|---|
| worlds | intact worlds of arm S's prior, seed 42, at held-out indices: the hook indices from rec1 8,810,000, attr2 7,810,000, list1 11,810,000 and categorical2 7,700,000; plain: the stage-2 indices from 120,000 that are plain worlds | `prior.heldout_slots` |
| scores | every model scores every query of a world (tasks sorted by name, tail then head queries, in chunks of 2) over all entities, through the training-mode path of pretraining | `prior.reciprocal_ranks` |
| rank | among all entities, with the query's other true answers in the world's truth and the anchor removed, ties at their mean rank | `data.rank_of` |
| metric | per world the mean reciprocal rank; per model name and family the mean over the seeds of the mean over the worlds | `prior.evaluate` |
| difference | D = the seed mean of A minus the seed mean of B; SE = sqrt(SE_world^2 + SE_seed^2), SE_world = the SD over worlds of the per-world seed-mean difference / sqrt(worlds), SE_seed = sqrt(var(A's seed MRRs) / n_A + var(B's) / n_B) | `prior.compare` |

The rank of this command is on purpose not the pessimistic rank of the evaluation on graphs.
