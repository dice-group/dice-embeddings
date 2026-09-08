# ULTRA in DICE

ULTRA is a graph-conditioned link predictor implemented using PyTorch alone.
It supports the official `ultra_3g.pth`, `ultra_4g.pth`, and `ultra_50g.pth`
checkpoints from [DeepGraphLearning/ULTRA](https://github.com/DeepGraphLearning/ULTRA/tree/427966ad8ed60420eef034063d44f3153addff90/ckpts).
No PyG, torch-scatter, compiler, or upstream checkout is needed at runtime.

## Evaluate a pretrained checkpoint

Download a checkpoint from the upstream repository, then run:

```bash
python -m dicee --model ULTRA --dataset_dir KGs/WN18RR \
  --ultra_checkpoint /path/to/ultra_3g.pth --num_epochs 0 \
  --trainer torchCPUTrainer --scoring_technique NegSample \
  --batch_size 8 --eval_model test \
  --path_to_store_single_run Experiments/ultra-zero-shot
```

DICE indexes your graph, attaches **only training facts** to ULTRA, loads weights,
and runs its usual filtered evaluator. Validation/test facts are used for ranking
filters, not message passing. Existing DICE metric/tie conventions apply; loading
weights does not imply reproducing a published benchmark's entire data protocol.

The resulting directory can be loaded through the usual prediction API:

```python
from dicee import KGE

kge = KGE(path="Experiments/ultra-zero-shot")
# Use entity/relation names from your dataset:
scores = kge.predict_missing_tail_entity("entity_a", "relation_r")
head_scores = kge.predict_missing_head_entity("relation_r", "entity_b")
```

The experiment includes `ultra_graph.pt`, vocabulary CSVs, configuration, and
`model.pt`. Keep the graph artifact when moving the experiment. Transferable
weights contain no graph-specific tables or graph buffers, so the same weights
can be attached to another graph with different vocabulary sizes.

## Train or fine-tune with native DICE objectives

```python
from dicee.config import Namespace
from dicee.executer import Execute

args = Namespace()
args.model = "ULTRA"
args.dataset_dir = "KGs/UMLS"
args.trainer = "torchCPUTrainer"
args.scoring_technique = "KvsAll"  # NegSample, FixedNegSample, 1vsAll,
                                  # 1vsSample and KvsSample also work
args.batch_size = 8
args.num_epochs = 10
args.optim = "AdamW"
args.lr = 5e-4
# For fine-tuning, uncomment:
# args.ultra_checkpoint = "/path/to/ultra_3g.pth"
report = Execute(args).start()
```

Existing DICE BCE losses and sampling remain the defaults. ULTRA removes training
target edges and their inverses during entity message passing. For pair-based
objectives it removes all observed training targets for the queried `(h,r)` pairs;
label smoothing cannot accidentally expose a target. The relation graph is built
once from the complete training graph, as in upstream ULTRA.

## Reusable upstream-style training options

The following settings are available to other indexed `BaseKGE` models too,
including DistMult. They are independent of ULTRA's graph operations:

| Option | Default | Behavior |
| --- | --- | --- |
| `grouped_negative_sampling` | `False` | Keep each positive and its negatives together; first half of the batch corrupts tails, second half corrupts heads. |
| `strict_negative_sampling` | `False` | Exclude known training positives; sample with replacement. |
| `adversarial_temperature` | `None` | `None`: ordinary mean BCE. `0`: equal weighting of the positive and the uniform negative group. Positive value: weight negatives with detached `softmax(logits / temperature)`. |

Strict sampling or a configured temperature automatically enables grouped
sampling. These options require `NegSample`, positive `neg_ratio`, and an indexed
entity-prediction model. BPE and sharded/tensor-parallel training are unsupported.
A query without any valid strict negative produces an explanatory error.

Example using the upstream sampler/loss and optimizer settings:

```bash
python -m dicee --model ULTRA --dataset_dir KGs/UMLS \
  --trainer torchCPUTrainer --scoring_technique NegSample \
  --strict_negative_sampling --adversarial_temperature 1 \
  --neg_ratio 256 --optim AdamW --lr 0.0005 --batch_size 8 \
  --label_smoothing_rate 0 --num_epochs 10
```

Add `--ultra_checkpoint /path/to/ultra_3g.pth` to fine-tune. Replace `ULTRA` with
`DistMult` to use the same sampling and loss settings for an embedding model.
Epoch scheduling and evaluation remain DICE's; this is not a reproduction of the
upstream multi-graph pretraining schedule.

## Direct model API and checkpoint contract

```python
import torch
from dicee.models import ULTRA

model = ULTRA({"num_entities": 4, "num_relations": 2})
model.load_pretrained("/path/to/ultra_3g.pth")
model.set_graph(torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3]]))
model.eval()
with torch.no_grad():
    triple_scores = model(torch.tensor([[0, 0, 2]]))
    tail_scores = model(torch.tensor([[0, 0]]))
    head_scores = model.forward_k_vs_all_heads(torch.tensor([[0, 2]]))  # (r,t)
```

All public triples use DICE order `(head, relation, tail)`. `set_graph` accepts
optional vocabulary sizes and an explicit `inverse_relations={direct_id:
inverse_id}` mapping. Without that mapping every supplied relation is direct;
inverse traversal edges are created internally. Repeated facts are deduplicated.
The DICE executor supplies the mapping when reciprocal preprocessing is enabled.
Replacing a graph rebuilds its derived state.

`model((queries, candidate_ids))` scores sampled tails. Candidate IDs can be a
shared vector or a matrix with one row per query. `model.forward_grouped(triples)`
accepts `[batch, 1 + negatives, 3]` and preserves head-corruption conditioning.
Flat triple scoring is tail-oriented: use `forward_k_vs_all_heads` for head
ranking. DICE reciprocal queries use the original direct relation's seed,
matching upstream head prediction.

Official checkpoints require `ultra_dim=64` and `ultra_num_layers=6` (defaults).
These are ULTRA-specific architecture settings; the generic `embedding_dim`
controls embedding-table models. Custom dimensions/layer counts can be used for
scratch training. All checkpoint keys and shapes must match strictly. Both raw
state dictionaries and the official `{"model": ..., "optimizer": ...}` wrapper
are accepted; upstream optimizer state is not imported.

The implementation preserves the **fused rspmm kernel's** row/column convention,
which differs from PyG's unfused default propagation direction. Both reasoners
use DistMult messages, sum aggregation, layer normalization, and residuals.

## Capacity and supported execution

Start with training `batch_size=8` or smaller. `ultra_query_batch_size=8` bounds
simultaneous query evaluation, but training retains activations for the entire
optimizer batch. Pure-PyTorch message materialization has memory cost proportional
to edges × queries × hidden dimension. Increase graph/batch sizes deliberately.

CPU and single-GPU native/Lightning execution are supported. For Lightning use
`--trainer PL --pl_trainer_kwargs '{"accelerator":"gpu","devices":1}'`.
Distributed/sharded training, BPE, cross-validation, static embedding export,
UltraQuery, and multi-graph pretraining are outside this implementation.

A local CPU check on WN18RR's training graph (40,559 entities, 11 relations,
86,835 facts / 173,670 traversal edges) with four PyTorch threads measured:

| Operation | Time |
| --- | --- |
| Build graph and relation graph | 1.41 s |
| Score all tails for eight queries | 3.41 s |
| Forward/backward for two grouped queries | 2.35 s |

Peak process RSS across these operations was 3.14 GiB, including Python/PyTorch
and imported DICE dependencies. Measurements used PyTorch 2.13.0+cu130 on CPU;
they describe this host, not a performance guarantee.

## Verification

```bash
# Offline tests include a small upstream-generated model and reference outputs.
OMP_NUM_THREADS=2 python -m pytest tests/test_ultra.py -q

# Also verify all three official checkpoint files against upstream outputs:
ULTRA_CHECKPOINT_DIR=/path/to/ULTRA/ckpts OMP_NUM_THREADS=2 \
  python -m pytest tests/test_ultra.py -q
```

Reference outputs come from upstream commit
`427966ad8ed60420eef034063d44f3153addff90`, with its compiled CPU rspmm kernel.
Tests check relation graph edges, relation representations, head/tail logits,
and training logits with `atol=1e-5`, `rtol=1e-4`. The tiny reference runs offline;
released checkpoint tests also verify file SHA-256 hashes. Fixture regeneration
is documented in `tests/fixtures/ultra/generate_reference.py`.
