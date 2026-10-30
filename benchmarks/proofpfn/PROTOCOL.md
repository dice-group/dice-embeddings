# Evaluation protocol

This document states how the [ProofPFN benchmark](README.md) ranks a graph, and which research module each step
reproduces. The research chain was `fresh-v1-runtime/fv1_harness.py` → `allgraph-v1/ag_seedraw.py` → `ag_eval.py` →
`lockbox-v6-runtime/fair_eval.evaluate` → `lockbox-v5-runtime/lb5_eval` → `ensemble-v2-runtime/e3_eval` →
`inductive-benchmark-v1-runtime/ib_eval` and `ib_rule_eval` → `context-injection-v1-runtime/ci_evaluate.query_scores`
(paths below `Experiments/synthetic-kg` of the archived research code). Here each step is one function, without registry
patches, `sys.path` changes or frame hooks.

## Graph

| Step | Rule | Research | Here |
|---|---|---|---|
| hash check | before a graph loads: the file SHA-256 of train.txt, valid.txt and test.txt where the manifest records them, and the audit's content hash of train.txt and test.txt (SHA-256 of the sorted distinct triples, a line split at tabs if it has one, else at whitespace) where recorded; T1 also its counts | `ag_eval.graph_problems`, `suite_eval.input_problems`, `fv1_harness.graph_problems` | `manifests.graph_problems`, `data.load_graph` |
| ids | entities and relations are the sorted names over train, valid and test; the triples of each split keep their file order, duplicates included | `ib_eval.load_dataset` | `data.read_graph` |
| inference graph | train.txt; the model keeps its sorted unique triples plus technical inverses | `set_graph` | `systems.attach` |
| test rows | `first-1000`: all rows of a test split with at most 1,000 rows; else the first occurrence of every distinct test fact in file order is the pool and `sorted(random.Random(0).sample(pool, 1000))` is ranked. `all`: every row. The row `index` is the position in test.txt | `ag_eval.sample_positions`, `sampled_loader` | `data.ranked_positions` |
| filter | every fact of train, valid and test (unsampled test facts included), except the target | `e3_eval.known_sets` | `data.known_sets` |
| query groups | `(head direction, relation, anchor)` with its test members in index order, stably sorted by relation | `lb5_eval.query_groups` | `data.query_groups` |

dicee's `KG` is not used for loading: its pandas reader turns tokens such as `NA` and `null` into missing values and
merges distinct entities (`tests/test_data.py`).

## Scores and views

| Step | Rule | Research | Here |
|---|---|---|---|
| models | built on the CPU, `query_batch_size` 1 (official Flock: its passes), `set_graph(train)` on the CPU, then moved to the device, evaluation mode, no gradients | `lb5_eval.attach`, `e3_eval.attach` | `systems.attach` |
| scores | one call per query group and model: `forward_k_vs_all([[h, r]])` for a tail query, `forward_k_vs_all_heads([[r, t]])` for a head query (the first member's anchor), scores asserted finite | `ci_evaluate.query_scores` | `evaluate.query_scores` |
| execution | `deterministic_execution()` (deterministic algorithms, float32 matmul precision `highest`, no cuDNN TF32) with `CUBLAS_WORKSPACE_CONFIG=:4096:8`; two CPU threads | `ag_eval.main` | `evaluate.evaluate_graph` |
| model view | `log_softmax(scores.double())` | `fair_eval.evaluate` | `evaluate._group_rows` |
| family view `F` | the mean of the three seeds' model views (`torch.stack(...).mean(0)`); a reference's view is its model's | `fair_eval.evaluate` | `evaluate._group_views` |
| rule view | the anchor gets `-inf` unless at least 1% of the relation's train facts are self-loops (`share1`) | `ib_rule_eval.open_relations` | `data.reflexive_relations` |
| rank | pessimistic: `#{v >= v[target]}` over the unfiltered candidates, the target included; with the rule's `-inf` anchor, a target that is the anchor gets the last rank | `lb5_eval.rank` | `FilteredRanker(tie_policy='pessimistic')`, which gives the same rank for every vector without NaN (`tests/test_evaluate.py`); scores are finite by the check above |
| GT2 | on the allowed candidates, the probability mass of the candidates unseen in the query role of train.txt moves half-way in log space towards `g = (n1 + 0.5) / (N + 1)` of that role; the rank is pessimistic over the allowed candidates | `fair_eval.gt_inputs`, `fair_eval.gt2`, `lb5_gt.gt2_rank` | `data.gt2_inputs`, `evaluate.gt2_rank` |

`GraphRankPlan` and `evaluate_lp` of `dicee.evaluation` are not used: they rank float32 scores in batches and drop them,
while the family, rule and GT2 views need the vectors.

## Row keys

Per family `F` and seed `S` (42, 43, 44), and per reference `F`, in this order:

| Key | View |
|---|---|
| `F` | rule view (family mean; reference model) |
| `F:gt2` | rule view with GT2 |
| `F:raw` | no rule |
| `F:raw:gt2` | no rule, with GT2 |
| `F.seedS:raw` | seed S, no rule |
| `F.seedS` | seed S, rule view |
| `F.seedS:gt2` | seed S, rule view with GT2 |

The research rows got `F.seedS:raw` from a hook that read `fair_eval.evaluate`'s locals (`ag_seedraw.rank`); here it is a
rank like the others. The key order equals the research rows'. `metrics` holds the mean reciprocal rank per key
(`numpy.mean` of `1 / rank`, as `fair_eval`). The registered analysis reads the raw single-model keys: `F.seedS:raw` for
the families and `F:raw` for the references.

## Official Flock

`flockoff` is Flock (`checkpoints/flock/flock_entity.pth`) with sampling seed 42 at the graph's official test-time
setting (`fo_rule.setting`, `fo_flock.use`): the walk count is `2^round(log2(128 x HM(|V| / 24038, |E| / 181511.3)))`
clamped to 16..512, with |V| and |E| the distinct entities and distinct triples of train.txt (checked against the
manifest's `size`); the passes are 16 for at most 128 walks, 4 for 256 and 2 for 512. On the 54 graphs of the official
script (by suite id) its walk count and passes apply. The passes of one query run as one microbatch of the walk sampler
(`query_batch_size` = passes), as the research `BatchedFlock`; `--flock-passes-in-sequence` runs them one after another
instead (the same law, other walks). Each query draws its walks from a generator seeded with 42, so ranks do not depend on
the order of the queries or on sharding.

## Row file record

`allgraph` keeps the fields of the research record that the registered `fv1_tables.py` reads (`key`, `sha256` with the
three file names, `model_sha256`, `ranked_facts`) and the others (`collection`, `path`, `group`, `verdict`, `weight`,
`origin_class`, `suite_id`, `test_rows`, `sampled`, `sample_rule`, `positions`, `families`, `seeds`, `gpu`, `torch`,
`flock_official`; T1 graphs also `tail_only_view`). Differences from the research record, none of which a table reads:
`protocol` is this benchmark's string; `families` holds the system specs instead of the research run tuples; `loaders`,
`host`, `gpu_cap_gb`, `peak_gpu_gb` and `runner_sha256` are replaced by `provenance`; `flock_official` has no
`code_sha256`; the gate entry of grail-WN18RR_v1 keeps the audit's group (null), where the golden run's stand-in audit
gave it the contamination view.

## External baselines

The adapters (`adapters/`, README section External baselines) rank the same test facts in both directions with the same
filter and the same self-loop rule. The three adapters reproduce their research runners on their goldens: the KGPFN
probe ranks, the AnyBURL rank rebuild and the MERIT check ranks. The MERIT check ranks hold on an H100 NVL in MERIT's
environment (torch 2.5.1+cu124, torch-scatter 2.1.2, torch-geometric 2.4.0, easydict, jinja2 and PyYAML), where
`test_check_graph_ranks_equal_the_research_ranks` passed.

| Adapter | Scores | Ranks | Research |
|---|---|---|---|
| `merit` | the five MERIT checkpoints as references of `evaluate.evaluate_graph`: the views, ranks and GT2 ranks above | as above | `ag_merit.py` over `ag_eval.py`, with the model of `merit_ref.py` |
| `kgpfn` | KGPFN's scores of every candidate (`_score_all_candidate_tails`), the answer-blind context of seed 0, a head query as KGPFN's inverse query | pessimistic over the allowed candidates on the float64 scores. On small graphs the host checks them against `FilteredRanker` on the `log_softmax` of the dumped scores. A run with a non-finite score is refused (as the frozen `validate_kg`). `kgpfn` with the rule, `kgpfn:raw` without | `kgpfn_eval.py` (all-graph mode), `kgpfn_ag.py` (prepare, finalize, verify) |
| `anyburl` | AnyBURL's printed confidence per listed candidate, `-inf` for a candidate it does not list | `FilteredRanker(tie_policy='pessimistic')`. A target that is not listed, or that ties with the last entry of a full list, gets the number of allowed candidates. `anyburl` with the rule, `anyburl:raw` without | `anyburl_run.py` (`rebuild`) |

KGPFN numbers entities by first appearance in train, valid and test (its `TransductiveDataset`). A rank does not depend
on the numbering. Its worker parses a line at tabs if it has one, else at whitespace, and the host refuses a run whose
parse differs from the benchmark's. The rows of KGPFN and AnyBURL hold no GT2 keys (as the research rows), and their
`allgraph.model_sha256` is empty: the registered tables compare no model hash of them. AnyBURL's input files name
entity `i` and relation `j` of the benchmark's ids `e<i>` and `r<j>`. Its `valid.txt` also holds the test rows that are
not ranked, so that its filter is the benchmark's.

## N validity gate

The gate (`gate.py`, the command `gate`) is the manipulation check of arm N in amendment A1 of the registration (section
8.4): does N score below S on intact worlds of every nulled family, and close to S on plain worlds?

| Step | Rule | Research | Here |
|---|---|---|---|
| worlds | intact worlds of S's prior with seed 42, at held-out indices that no training order reaches: the hook indices from 8,810,000 (rec1), 7,810,000 (attr2), 11,810,000 (list1) and 7,700,000 (categorical2), and the stage-2 indices from 120,000 that are plain worlds (the slots of `prior-fix-v1/attr-v1/heldout/index.json`). At most 100 worlds per family | `icml-roadmap-20261025/step5/s5_manip.py`, as `pcu-training-20261026/manipulation.py` ran it | `gate.heldout_slots`, `priors.world` |
| models | each seed of the families of S and N, EMA weights, in training mode | `heldout_fit.load_model` | `checkpoint.model_from_checkpoint` |
| scores | every query of a world (tasks sorted by name, tail then head queries, in chunks of 2) over all entities, through the training-mode path of pretraining on the prepared world, in deterministic execution | `s5_manip.py` | `GraphKGE.forward_query_facts`, `gate.reciprocal_ranks` |
| rank | the rank of the answer among all entities, without the query's other true answers in the world's truth and without the anchor, ties at their mean rank. The value of a world is the mean reciprocal rank of its queries | `xr_diag.rank_of` | `data.rank_of` |
| comparison | per family: D = the mean over S's seeds minus the mean over N's seeds of the seed's mean over the worlds. SE = sqrt(SE_world^2 + SE_seed^2), with SE_world = the SD over worlds of the per-world seed-mean difference / sqrt(worlds), SE_seed = sqrt(var(S's seed MRRs) / 3 + var(N's) / 3) | `s5_manip.py` | `gate.compare` |
| reading | `n_validity` is `pass` when D > 2 SE on each of the four nulled families and abs(D) <= 2 SE on plain, else `fail` | the registration | `gate.reading` |

The gate's rank is the mean-tie rank of the cells (below), not the pessimistic rank of the evaluation.

## Cells

`cells.py` reproduces `fresh-v1-runtime/fv1_cells.py` with `icml-roadmap-20261025/step5/s5_measure.evaluate_rows` and
the predictors R1, R2 and PEER of `realsrc-v1/xrel/xr_diag.py` (module docstring). Their rank `data.rank_of` puts ties
at their mean rank and excludes the anchor, on purpose a different rule from the ranks above. The predictors POP, VOTE
and CF are not ported: no output holds them and they draw no random numbers.

The CSR matrix of a relation lists the edges of each row in train.txt order and keeps a repeated fact as its own entry,
as scipy 1.13.0 of the frozen run built it. R2 samples facts by their position in a row, and scipy 1.13.1 and later sort
the rows of `csr_matrix((data, (row, col)))`. So `cells.py` builds each matrix from an explicit stable sort and
`indptr`, and its `queries.csv` does not depend on the scipy version.
