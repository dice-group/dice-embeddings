# KG-ICL in DICE

[Paper](https://arxiv.org/abs/2410.12288) ·
[Pinned official implementation](https://github.com/nju-websoft/KG-ICL/tree/6a3166e347ae468acdfb30a70a2cf3608b66b8f1)

KG-ICL answers a query `(s, q, ?)` in context. It samples a few example facts
`(u, q, v)` of the query relation, extracts a prompt graph around each example
and encodes the relations of these prompt graphs. The mean prompt encoding
initializes the relations of a query-conditioned reasoner that expands one hop
per layer from `s` over the inference graph. DICE provides `KGICL` for entity
prediction with the graph, training and reload interfaces of [ULTRA](ultra.md),
[TRIX](trix.md) and [Flock](flock.md). The official `KG-ICL-4L`, `KG-ICL-5L` and
`KG-ICL-6L` checkpoints load without renaming or dropping parameters. Only
PyTorch is required; Triton adds the fused CUDA inference path.

## Checkpoints and zero-shot evaluation

```bash
mkdir -p checkpoints/kgicl/KG-ICL-6L
wget -P checkpoints/kgicl/KG-ICL-6L https://raw.githubusercontent.com/nju-websoft/KG-ICL/6a3166e347ae468acdfb30a70a2cf3608b66b8f1/checkpoint/KG-ICL-6L/model_best.tar
sha256sum checkpoints/kgicl/KG-ICL-6L/model_best.tar
# 80f885ffa2c0175cd6ed829af99ae45177fdb5f77302c7648591bbdf62bc1c81

python -m dicee --model KGICL --dataset_dir KGs/UMLS \
  --kgicl_checkpoint checkpoints/kgicl/KG-ICL-6L/model_best.tar --num_epochs 0 \
  --trainer torchCPUTrainer --scoring_technique NegSample \
  --batch_size 8 --eval_model test \
  --path_to_store_single_run Experiments/kgicl-zero-shot
```

The directory contains `train.txt`, `test.txt` and optionally `valid.txt` in
`(head, relation, tail)` order. Reasoning and prompt graphs use training facts
only. Validation and test facts are used for filtered evaluation. The released
`model_best.tar` holds `{"state_dict", "optimizer_state_dict", "epoch_id"}`;
`load_pretrained()` also accepts a `{"model": ...}` wrapper or a bare state
dictionary. Keys and shapes are checked before loading. Fine-tuning starts with
a new optimizer.

| Setting | Default | Meaning |
|---|---:|---|
| `kgicl_dim` | 32 | Hidden dimension |
| `kgicl_attn_dim` | 5 | Reasoner attention dimension |
| `kgicl_num_layers` | 6 | Reasoner layers: 4, 5 or 6 for the official checkpoints |
| `kgicl_prompt_layers` | 3 | Prompt encoder layers |
| `kgicl_prompt_hops` | 3 | Prompt distance bound `k`; sets the token table size |
| `kgicl_shots` | 5 | Prompt graphs per query relation |
| `kgicl_prompt_open_nodes` | 50 | Further entities per prompt graph, see below |
| `kgicl_prompt_seed` | 0 | Seed of example and entity sampling |
| `kgicl_masked_distances` | unset | Hop distances whose entities score zero |
| `kgicl_query_batch_size` | 8 | Queries per batched GPU reasoning pass |

The first five settings determine checkpoint shapes; leave them at their
defaults for `KG-ICL-6L`. Generic `embedding_dim` does not control KG-ICL.
`graph_relation_cache_mb` bounds the cached prompt encodings and
`graph_projection_cache_mb` the per-layer relation tables of the fused path.

## Prompt graphs

DICE builds prompt graphs from the attached graph as the official data
processing builds them (`utils_all.py`, `enclosing=False`):

- Examples are up to `kgicl_shots` distinct facts `(u, q, v)` of the direct
  relation `q`. With `n < shots` facts, shot `j` uses example `j mod n`.
- Distances are undirected hop counts in the attached graph, computed up to
  `k = 3` from `u` and from `v`. The prompt graph contains `u`, `v`, every
  entity `x` within `k` of both with `dist(x,u) + dist(x,v) <= k`, and up to 50
  further entities within `k` of both. It contains every fact among them.
- Entity tokens are `(dist(x,u), dist(x,v))`. `u` receives the head token and
  `v` the tail token. An inverse query relation `q⁻` reuses the examples of `q`
  with head and tail tokens swapped; the other tokens keep their `(u, v)` order,
  as in the released code.
- A relation without facts uses the official fallback: two nodes joined by one
  query-relation edge.

The official code samples examples and further entities with unseeded NumPy
calls once, offline. DICE draws them from CPU generators seeded with
`sha256("kgicl-prompt:{seed}:{relation}")`, for the prompt seed and the public
ID of the direct relation; further entities additionally use the example index.
Examples are drawn from the relation's facts sorted by `(head, tail)`. A graph,
relation and seed therefore always produce the same prompt graphs, independently
of query batches, call order and other relations. Prompt graphs are built on
first use. Their encodings are cached per relation during inference and cleared
when the graph, weights, device or settings change. During training, each shot
draws one of the relation's examples at random, as the official training does.

## Scores and the official evaluation protocol

Entities the reasoner does not reach within `kgicl_num_layers` hops score exactly
zero, as in the official code. The official evaluation differs from DICE's
benchmark protocol in three ways, which DICE exposes but does not apply by default:

- **Inference graph.** Official test queries reason over training plus
  validation facts; prompt graphs use training facts only. DICE uses the graph
  passed to `set_graph()` for both.
- **Distance mask.** Official evaluation computes, from validation facts, the
  fraction of answers at each hop distance and zeroes entities first reached at
  distances below 2% (always including the query head). Use
  `--kgicl_masked_distances 0 5 6` or `answer_distance_rates()` with
  `benchmarks/kgfm_zero_shot.py --distance-mask` to apply it.
- **Ties.** Official ranks break exact ties by entity ID (`rankdata(method='ordinal')`).

## Training and fine-tuning

Entity prediction supports `NegSample`, `FixedNegSample`, `KvsAll`, `1vsAll`,
`1vsSample` and `KvsSample`, with grouped strict negatives and adversarial BCE:

```bash
python -m dicee --model KGICL --dataset_dir KGs/UMLS \
  --kgicl_checkpoint checkpoints/kgicl/KG-ICL-6L/model_best.tar \
  --trainer torchCPUTrainer --scoring_technique NegSample \
  --strict_negative_sampling --adversarial_temperature 1 --neg_ratio 64 \
  --batch_size 8 --num_epochs 5 --optim Adam --lr 0.0005
```

Training removes supervised facts and their inverses from the reasoning graph;
pair queries hide all their known answers. Prompt graphs keep all training facts,
as in the official training. Training mode samples RReLU slopes, applies the
prompt encoder's dropout and draws prompt examples at random. DICE trains with
its own objectives; it does not reproduce the official multi-graph pretraining
loop or its relation-dropout augmentation. Native CPU/single GPU and
single-device Lightning are supported. BPE, distributed training,
cross-validation and static embedding export are rejected.

## Python use and experiment reloads

```python
import torch
from dicee.models import KGICL

facts = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 0], [0, 1, 2]])
model = KGICL(dict(num_entities=3, num_relations=2)).load_pretrained(
    "checkpoints/kgicl/KG-ICL-6L/model_best.tar")
model.set_graph(facts).eval()
with torch.no_grad():
    triples = model(torch.tensor([[0, 1, 2]]))
    tails = model(torch.tensor([[0, 1]]))                    # (head, relation)
    heads = model.forward_k_vs_all_heads(torch.tensor([[1, 2]]))  # (relation, tail)
```

Scores are logits. Call `.eval()` for inference. Triples use DICE's
`(head, relation, tail)` order. Head prediction conditions on the converted
inverse relation, as the official evaluation does. `set_graph()` adds inverse
edges, deduplicates facts and resets prompt graphs; pass
`inverse_relations={direct_id: inverse_id}` for vocabularies with explicit
inverses. `prompt_graphs(relation)` returns the prompt graphs of an internal
relation ID (inverses are `r + R`), `encode_prompts(relation, graphs)` encodes
any prompt graphs, and `use_prompts({relation: graphs})` replays fixed prompt
graphs.

```python
from dicee import KGE

kge = KGE(path="Experiments/kgicl-zero-shot")
tails = kge.predict_missing_tail_entity("entity_a", "relation_r")
```

Keep `kgicl_graph.pt` beside the weights, configuration and vocabularies when
moving an experiment. Prompt graphs are rebuilt from it with the saved seed.

## Determinism and batching

Inference is deterministic, and a row depends only on the graph, weights,
settings, head and relation:

- The PyTorch path scores each query alone during inference, so its rows never
  depend on the batch. On CUDA it is bitwise reproducible under
  `torch.use_deterministic_algorithms(True)` (set `CUBLAS_WORKSPACE_CONFIG`).
- The fused CUDA path (Triton) keeps a dense entity-major `[entities, queries, dim]`
  state with an active mask: messages leave only active entities, which
  reproduces the official hop-by-hop expansion. One program aggregates the
  incoming edges of one entity for a block of four queries, whose source states
  are contiguous; the attention's source term `Ws_attn h` is projected once per
  entity and layer instead of once per edge. Each row's aggregation and GRU
  update use a fixed operation order without atomics, so rows are bitwise
  identical for every query batch size and order, with or without deterministic
  algorithms. Entities with more than 256 incoming edges are summed in 64-edge
  partial sums that are added in order. Dense products use 3xTF32 tensor-core
  dots, which keep float32 accuracy, on GPUs with TF32 support, and IEEE FMA
  dots otherwise.
- Prompt encodings and relation tables are computed for one relation at a time.

`kgicl_query_batch_size` only trades memory for speed on the fused path: each
query holds about `420 * entities` bytes of state, and 16 queries per pass are
about 13% faster than 8 on FB15k-237. See [KGFM inference speed](kgfm_inference.md#kg-icl)
for timings of query answering rows.

## Upstream defects corrected in DICE

| ID | Location | Defect | Correction |
|---|---|---|---|
| B1 | `EntityEncoder.py` | The reasoner's attention activation is a fresh `nn.RReLU()` per call. A new module is in training mode, so evaluation samples random negative slopes: predictions change between runs and exact ties are broken at random. | Use the model's mode: random slopes in training, the mean slope `(1/8 + 1/3) / 2` in evaluation, as RReLU defines. |
| B2 | `data_loader.py` | Evaluation shot `i` of the batch-wide relation list selects case `i mod n_cases`. For a relation with 2 to 4 cases, the selected cases depend on the relation's rank among the batch's relations, so prompts depend on the (shuffled) query batch. | Shot `j` of a relation selects case `j mod n_cases`. |
| B3 | `utils_all.py` | Distance labels are aligned with the node list but are paired with nodes in Python set order, and the loader reads label rows as local node IDs. Most prompt entities receive another entity's distance tokens. The pretraining data was processed with the same code. | Pair each entity with its own label; write rows in local-ID order. |
| B4 | `utils_all.py` | A self-loop pair `(a, a)` is looked up in both directions, so every self-loop fact appears twice in the prompt graph (doubling its degree). | Each fact appears once. |
| B5 | `data_loader.py` | For a self-loop example `(u, q, u)` the tail token goes to local node 1, an unrelated entity. | Head and tail tokens go to the example entity (the tail token overwrites the head token, matching the token assignment order). |

No defect leaks test facts: prompt graphs come from training facts, and the
reasoning graph of test queries holds training and validation facts. DICE
always implements the corrected behavior; to evaluate the released behavior,
run the official code.

## Upstream behavior DICE keeps

These choices are consistent between the official training and evaluation, and
the checkpoints depend on them; DICE matches them:

- The prompt encoder multiplies entity messages by the symmetric degree
  normalization twice, and the reasoner applies each relation layer norm twice.
- Unused parameters (`layer_norms_query`, `query_transfer`, `W_score` and the
  NBFNet `conv` layers present when the checkpoints were trained) are retained
  for strict loading; the official evaluation drops them with `strict=False`.
- Prompt graphs follow the released processing (`enclosing=False`: up to 50
  further entities), not the paper's `dist(x,u) + dist(x,v) <= k` alone.
- The reasoner sums messages (the paper writes mean pooling) and updates
  entities with a GRU; unreached entities score zero.
- Labels of inverse-query prompts keep the direct example's `(u, v)` order; only
  the head and tail tokens are swapped.
- The official sigmoid attention (`attn_type='Sigmoid'`); the GAT option, which
  indexes its normalizer by a node ID where an edge ID is meant, is unused.

## Tests

```bash
python -m pytest tests/test_kgicl.py -q
```

The tests need no checkpoint and no upstream package. They cover prompt
sampling (seeded, independent of relation and call order, identical after
reattaching the graph or reloading an experiment), the scoring interfaces,
training, and batch-independent rows on CPU and CUDA.

## Complex query answering

`kgicl-adapter` evaluates KG-ICL as a frozen backbone in the query benchmarks
(`python -m benchmarks.cqa`), with the adapter, beam search and controls of the
ULTRA and TRIX recipes. Prompt graphs come from each benchmark's inference
graph with prompt seed 0 (a recipe may set `prompt_seed`); no distance mask is
applied. Score adapters bind to the backbone through the checkpoint's state
fingerprint, and cached training rows record the prompt settings. The
evaluator caches atomic rows across beams and controls; KG-ICL rows are
bitwise identical whatever the batching or call order, which the tests check
under deterministic algorithms on CPU and CUDA.

The recipes `benchmarks/plus_h/kgfm_kgicl.json` and
`benchmarks/ultraquery/kgfm_kgicl.json` use the adapter
`benchmarks/adapters/kgicl_product_intersections.json`, fitted with the shared
source recipe of ULTRA and TRIX (training seed 2026090851) and paired with an
identity control.
They cache the per-layer relation
tables of every query relation (`projection_cache_mb` 512, about 260 MiB on
FB15k237); the default 64 MiB recomputes them on graphs with hundreds of
relations, about 2.5 times slower. Outside the benchmark's Docker image, set
`CUBLAS_WORKSPACE_CONFIG=:4096:8`: the evaluator enables deterministic
algorithms, and the prompt encoder uses cuBLAS.

## License

The official KG-ICL code and checkpoints are distributed under GPL-3.0 by their
authors; download the checkpoints from the official repository. DICE's
implementation is an independent reimplementation that loads the released state
dictionaries.
