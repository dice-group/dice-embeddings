# TRIX in DICE

[Paper](https://arxiv.org/abs/2502.19512) ·
[Pinned official implementation](https://github.com/yuchengz99/TRIX/tree/7596e14eefefe89e61396205a0550172cadeddb0)

TRIX uses PyTorch with an optional fused Triton inference kernel. The two separately
trained official models are available as `TRIX` for entity prediction and
`TRIXRelation` for relation prediction. DICE does not require PyG, torch-scatter,
or a compiled graph extension to run either model.

## Download and evaluate the official checkpoints

```bash
mkdir -p checkpoints/trix
wget -P checkpoints/trix https://raw.githubusercontent.com/yuchengz99/TRIX/7596e14eefefe89e61396205a0550172cadeddb0/entity_prediction.pth
wget -P checkpoints/trix https://raw.githubusercontent.com/yuchengz99/TRIX/7596e14eefefe89e61396205a0550172cadeddb0/relation_prediction.pth

# Zero-shot head and tail prediction.
python -m dicee --model TRIX --dataset_dir KGs/UMLS \
  --trix_checkpoint checkpoints/trix/entity_prediction.pth --num_epochs 0 \
  --trainer torchCPUTrainer --scoring_technique NegSample \
  --batch_size 8 --eval_model test \
  --path_to_store_single_run Experiments/trix-entity

# Zero-shot relation prediction, scoring all relations together for each pair.
python -m dicee --model TRIXRelation --dataset_dir KGs/UMLS \
  --trix_checkpoint checkpoints/trix/relation_prediction.pth --num_epochs 0 \
  --trainer torchCPUTrainer --scoring_technique KvsAll \
  --batch_size 8 --eval_model test \
  --path_to_store_single_run Experiments/trix-relation
```

The dataset directory contains `train.txt` and `test.txt`, with an optional
`valid.txt`, in `(head, relation, tail)` order. Only training facts form the
message-passing graph. Validation and test facts contribute to filtered
evaluation. `TRIXRelation` keeps generated inverses internal to the graph; its
prediction labels are the relations supplied in the dataset.

`load_pretrained()` accepts the official `{"model": state, "optimizer": ...}`
checkpoint or a bare state dictionary. Every parameter name and shape must match;
passing the checkpoint for the other task raises an architecture mismatch error.
The optimizer state is not restored when fine-tuning.

Both default architectures use `trix_dim=32`. The generic `embedding_dim` option
does not change them. `trix_dim` can be changed for scratch training, while layer
counts and convolution settings follow the published models. Other architectures
are not automatically inferred from checkpoints.

## Train or fine-tune

Entity prediction supports the same DICE objectives as ULTRA: `NegSample`,
`FixedNegSample`, `KvsAll`, `1vsAll`, `1vsSample`, and `KvsSample`. To use grouped
head/tail corruptions, strict negatives, and the upstream adversarial BCE objective:

```bash
python -m dicee --model TRIX --dataset_dir KGs/UMLS \
  --trainer torchCPUTrainer --scoring_technique NegSample \
  --strict_negative_sampling --adversarial_temperature 1 \
  --neg_ratio 64 --optim AdamW --lr 0.0005 --batch_size 8 --num_epochs 10
```

For relation prediction, DICE's training/evaluation pipeline uses `KvsAll`:

```bash
python -m dicee --model TRIXRelation --dataset_dir KGs/UMLS \
  --trainer torchCPUTrainer --scoring_technique KvsAll \
  --optim AdamW --lr 0.0005 --batch_size 8 --num_epochs 10
```

Add `--trix_checkpoint` with the matching checkpoint to fine-tune. The relation
model also supports grouped relation corruptions through its direct Python API;
the CLI uses DICE's all-relation BCE objective rather than the paper's sampled
pretraining procedure. This integration does not reproduce the multi-graph
pretraining run or claim to reproduce the paper's benchmark results.

Training removes supervised facts and their inverses from every entity reasoning
stage. For pair queries, all observed positive answers to those queries are hidden.
The entity-labelled relation graph remains constructed from all training facts,
matching the official implementation.

CPU/single GPU native training and single-device Lightning are supported. BPE,
distributed training, cross-validation, and static embedding export are rejected.
Use `trix_query_batch_size` (default 8) to bound simultaneous queries. Message
passing fuses edge messages on supported CUDA inference paths; the portable
PyTorch fallback materializes them. See the [inference speed comparison](kgfm_inference.md)
for timings against the official implementation. Relation-graph storage grows with the
number of distinct incident-relation pairs per entity.

## Python inference and experiment reloads

```python
import torch
from dicee.models import TRIX, TRIXRelation

facts = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 0]])
args = dict(num_entities=3, num_relations=2)

entity_model = TRIX(args).load_pretrained("checkpoints/trix/entity_prediction.pth")
entity_model.set_graph(facts).eval()
relation_model = TRIXRelation(args).load_pretrained("checkpoints/trix/relation_prediction.pth")
relation_model.set_graph(facts).eval()
with torch.no_grad():
    triple_score = entity_model(torch.tensor([[0, 1, 2]]))
    tail_scores = entity_model(torch.tensor([[0, 1]]))       # (head, relation)
    head_scores = entity_model.forward_k_vs_all_heads(torch.tensor([[1, 2]]))
    relation_scores = relation_model(torch.tensor([[0, 2]])) # (head, tail)
```

Call `.eval()` for inference: modules start in training mode, where target edges
are hidden. Scores are logits. Entity-model pairs use `(head, relation)`, explicit
head queries use `(relation, tail)`, and relation-model pairs use `(head, tail)`.
All triples and `[batch, candidates, 3]` groups use DICE's `(head, relation, tail)`
order, whereas the upstream code uses `(head, tail, relation)`.

Entity-model groups must share a relation and one uncorrupted endpoint. Use
`forward_grouped(groups, head_prediction=True)` when head corruption cannot be
inferred (for example, a single candidate). Relation-model groups share both
endpoints. Flat entity triples always denote independent tail-oriented queries;
use the explicit head method when predicting missing heads.

`set_graph()` internally adds inverse edges and deduplicates facts. For a vocabulary
already containing explicit inverse IDs, provide `inverse_relations={direct_id:
inverse_id}`. IDs may be interleaved. Reattaching a graph with different entity and
relation counts transfers the same parameters without resizing embedding tables.

```python
from dicee import KGE

entity_kge = KGE(path="Experiments/trix-entity")
relation_kge = KGE(path="Experiments/trix-relation")
# Substitute names from your dataset.
tails = entity_kge.predict_missing_tail_entity("entity_a", "relation_r")
heads = entity_kge.predict_missing_head_entity("relation_r", "entity_b")
relations = relation_kge.predict_missing_relations("entity_a", "entity_b")
```

Keep `trix_graph.pt` beside the weights, configuration, and vocabularies when
moving an experiment. Graph context is excluded from the transferable model state
dictionary and restored separately, including for checkpoint averaging.

## Compatibility details and verification

The released code determines checkpoint behavior where it differs from the paper's
general equations or source comments:

- Relation edges retain the shared entity as their edge type. Incidences have
  binary support; `hh`/`tt` omit equal-relation loops, and `ht`/`th` retain them.
- Each of the four interaction families has its own DistMult/sum convolution,
  relation projection, layer normalization, and ReLU. Their outputs are summed
  before adding a residual. Boundary messages are included at each layer.
- The fused kernel aggregates from `edge_index[1]` into `edge_index[0]`.
- Entity prediction uses three relation layers. After the first, a two-layer
  entity reasoner feeds its projected features back into relation reasoning.
  A separate four-layer entity reasoner produces the final scores. Initial entity
  features for the first relation layer are ones. The intermediate entity scorer's
  unused MLP parameters are retained for strict checkpoint compatibility.
- Head corruption seeds the relation reasoner with the original direct relation,
  then uses the inverse relation in both entity reasoners. DICE reciprocal queries
  preserve this conditioning.
- Relation prediction runs three rounds of two entity layers followed by two
  relation layers. Head/tail labels are +1/-1 (they cancel for a self-query), and
  relation boundary features are ones.

The fixtures were generated by the **unmodified official models and compiled
CPU rspmm kernel**, at the pinned commit above. The generator imports no DICE
model. Both randomly initialized small models and the actual published checkpoints
are checked, including all parameter gradients (and unused parameters).

| Official checkpoint | Parameters | Maximum inference-logit error | Maximum gradient error |
|---|---:|---:|---:|
| `entity_prediction.pth` | 87,138 | 9.54e-7 | 3.82e-6 |
| `relation_prediction.pth` | 147,489 | 3.82e-6 | 7.63e-6 |

These are CPU float32 errors on the committed fixture graph, which includes
cycles, parallel relations, repeated incidences, a self-loop, and an isolated
entity. Tests use `atol=1e-5`, `rtol=1e-4` and also check CUDA scoring/backprop,
permutation equivariance, graph replacement, inverse mapping, training-edge
removal, checkpoint validation, and end-to-end DICE save/reload.

```bash
# Offline fixtures and DICE integration; no upstream packages required.
python -m pytest tests/test_trix.py tests/test_ultra.py -q

# Include the two actual official checkpoints; their SHA-256 hashes are checked.
TRIX_CHECKPOINT_DIR=checkpoints/trix python -m pytest tests/test_trix.py -q
```

To regenerate fixtures in an isolated reference environment, install PyTorch 2.5.1,
PyG 2.4.0, a matching torch-scatter 2.1.2 build, easydict, and ninja. Check out
the pinned TRIX commit, then run:

```bash
CUDA_VISIBLE_DEVICES='' MAX_JOBS=2 PYTHONPATH=/path/to/TRIX/src \
  python tests/fixtures/trix/generate_reference.py /path/to/TRIX tests/fixtures/trix
```

The generator verifies the checkout revision and that tracked `src` files are
unmodified. Fixture hashes identify the exact official checkpoint files used.
