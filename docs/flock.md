# Flock in DICE

[Paper](https://arxiv.org/abs/2510.01510) ·
[Pinned official implementation](https://github.com/jw9730/flock/tree/f35103d25a78bdf4075de5c673a51de4979aa4d7)

Flock learns node and relation representations by encoding anonymous random walks
with bidirectional GRUs and pooling their outputs with learned attention. DICE
provides `Flock` for entity prediction and `FlockRelation` for relation prediction,
using the graph, training, and reload interfaces established for
[ULTRA](ultra.md) and [TRIX](trix.md). Both official checkpoints load without
renaming or dropping parameters. Only PyTorch is needed to run these models;
PyG, torch-scatter, and the compiled graph-walker package are needed only when
regenerating reference fixtures.

## Checkpoints and zero-shot evaluation

```bash
mkdir -p checkpoints/flock
wget -P checkpoints/flock https://raw.githubusercontent.com/jw9730/flock/f35103d25a78bdf4075de5c673a51de4979aa4d7/checkpoints/flock_entity.pth
wget -P checkpoints/flock https://raw.githubusercontent.com/jw9730/flock/f35103d25a78bdf4075de5c673a51de4979aa4d7/checkpoints/flock_relation.pth

python -m dicee --model Flock --dataset_dir KGs/UMLS \
  --flock_checkpoint checkpoints/flock/flock_entity.pth --num_epochs 0 \
  --trainer torchCPUTrainer --scoring_technique NegSample \
  --batch_size 2 --eval_model test \
  --path_to_store_single_run Experiments/flock-entity

python -m dicee --model FlockRelation --dataset_dir KGs/UMLS \
  --flock_checkpoint checkpoints/flock/flock_relation.pth --num_epochs 0 \
  --trainer torchCPUTrainer --scoring_technique KvsAll \
  --batch_size 2 --eval_model test \
  --path_to_store_single_run Experiments/flock-relation
```

The directory contains `train.txt`, `test.txt`, and optionally `valid.txt` in
`(head, relation, tail)` order. Message passing and walks use training facts only.
Validation and test facts are used for filtered evaluation. Relation prediction
ranks the supplied relation vocabulary; generated inverse relations stay internal
to the graph.

`--flock_checkpoint` accepts the official `{"model": ..., "optimizer": ...}`
checkpoint or a bare state dictionary. Keys and shapes are checked before loading;
using the checkpoint for the wrong task raises an architecture mismatch error.
Fine-tuning starts with a new optimizer.

| Setting | Default | Meaning |
|---|---:|---|
| `flock_dim` | 64 | Hidden dimension |
| `flock_walk_len` | 128 | Recorded positions per walk, including its starting node |
| `flock_refinements` | 6 | GRU and pooling refinement rounds |
| `flock_num_layers` | 1 | Layers in each bidirectional GRU |
| `flock_attention_heads` | 4 | Attention pooling heads |
| `flock_walk_num` | 128 | Base walk count per query and refinement |
| `flock_test_samples` | 1 | Stochastic predictions whose logits are averaged at inference |
| `flock_query_batch_size` | 1 | Maximum simultaneous queries, including ensemble samples |
| `flock_seed` | unset | Optional seed for repeatable sampling on the same call |

The first five settings determine checkpoint shapes; leave them at their defaults
for the published weights. Walk count, ensemble size, query batch size, and seed
can be changed with those weights. Generic `embedding_dim` does not control Flock.
For a quick CPU check, add `--flock_walk_num 2`; this reduces walk coverage.

Flock is stochastic even in evaluation mode. Fresh calls normally draw fresh
walks. `--flock_seed 42` makes repeated calls with the same graph, inputs, and
batching reproducible. Leave it unset during training to draw new walks on every
step. Changing query ordering, grouping, or chunk size can change sampled walks.
The sampler uses PyTorch's CPU RNG, which produces different draws from the
upstream C++ per-walk MT19937 generator for a given seed.

The authors' evaluation scripts use graph-specific walk counts and ensemble
sizes. DICE exposes these as explicit settings; it does not automatically choose
their per-dataset configuration. For example, `--flock_walk_num 128
--flock_test_samples 16` selects that published inference configuration's sampling
budget. Training always uses a single stochastic prediction per query.

## Training and fine-tuning

Entity prediction supports `NegSample`, `FixedNegSample`, `KvsAll`, `1vsAll`,
`1vsSample`, and `KvsSample`. Grouped strict negatives and adversarial BCE use the
same facilities as ULTRA and TRIX:

```bash
python -m dicee --model Flock --dataset_dir KGs/UMLS \
  --trainer torchCPUTrainer --scoring_technique NegSample \
  --strict_negative_sampling --adversarial_temperature 1 --neg_ratio 64 \
  --flock_walk_num 16 --batch_size 2 --num_epochs 10 \
  --optim AdamW --lr 0.0005 --weight_decay 0.01

python -m dicee --model FlockRelation --dataset_dir KGs/UMLS \
  --trainer torchCPUTrainer --scoring_technique KvsAll \
  --flock_walk_num 16 --batch_size 2 --num_epochs 10 \
  --optim AdamW --lr 0.0005
```

Add the matching `--flock_checkpoint` to fine-tune. Relation prediction uses
DICE's all-relation BCE training pipeline. Its Python model also accepts groups
of relation corruptions, but the CLI does not reproduce the authors' sampled
multi-graph relation-pretraining procedure.

Training removes supervised facts and their inverses **before sampling walks**.
Pair-query objectives remove all known positive answers for each query. Unlike
TRIX, Flock has no separate relation graph through which masked facts remain
visible. If masking empties the graph, DICE supplies isolated-node walks with
no-relation markers; this extends upstream handling to tiny training graphs.

Native CPU/single GPU and single-device Lightning are supported. BPE, distributed
training, cross-validation, and static embedding export are rejected. Walk
sampling runs on CPU; learned sequence processing and pooling run on the model's
device. Memory and runtime increase with walk count, length, query batch size,
and ensemble size. The default model follows the published GRU, additive
refinement, untied-parameter architecture without restarts or neighbor recording.
Inference caches relation-edge lists and prefetches one CPU walk microbatch while
CUDA scores the current one. See [KGFM inference performance](kgfm_inference.md)
for controls, reproducibility guarantees, and benchmarks.

## Python use and experiment reloads

```python
import torch
from dicee.models import Flock, FlockRelation

facts = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 0]])
args = dict(num_entities=3, num_relations=2, flock_walk_num=16, flock_seed=42)
entity_model = Flock(args).load_pretrained("checkpoints/flock/flock_entity.pth")
relation_model = FlockRelation(args).load_pretrained("checkpoints/flock/flock_relation.pth")
entity_model.set_graph(facts).eval()
relation_model.set_graph(facts).eval()

with torch.no_grad():
    triples = entity_model(torch.tensor([[0, 1, 2]]))
    tails = entity_model(torch.tensor([[0, 1]]))  # (head, relation)
    heads = entity_model.forward_k_vs_all_heads(torch.tensor([[1, 2]]))
    relations = relation_model(torch.tensor([[0, 2]]))  # (head, tail)
```

Scores are logits. Always call `.eval()` for inference; training mode hides target
edges. All triples and negative groups use DICE's `(head, relation, tail)` order,
while upstream uses `(head, tail, relation)`. Use the explicit head-prediction
method for missing heads. It conditions on the converted inverse relation,
matching Flock's official behavior.

`set_graph()` adds inverse edges, deduplicates facts, and rebuilds walk context.
If the external vocabulary already contains inverse IDs, provide
`inverse_relations={direct_id: inverse_id}`. Reattach a different graph with new
`num_entities` and `num_relations` to transfer the same pretrained parameters.
The sampler retains distinct direct/inverse relation IDs, as the released code
does; it does not collapse them during anonymization or query marking.

```python
from dicee import KGE

entity_kge = KGE(path="Experiments/flock-entity")
relation_kge = KGE(path="Experiments/flock-relation")
# Replace names with entries from your dataset.
tails = entity_kge.predict_missing_tail_entity("entity_a", "relation_r")
heads = entity_kge.predict_missing_head_entity("relation_r", "entity_b")
relations = relation_kge.predict_missing_relations("entity_a", "entity_b")
```

Keep `flock_graph.pt` beside the weights, configuration, and vocabularies when
moving an experiment. Model state dictionaries contain transferable parameters;
graph context is saved separately and rebuilt on reload, including for checkpoint
averaging. Walk RNG state is not saved in the graph artifact.

## Verification against the official implementation

Reference fixtures were generated at the pinned commit above using the unmodified
official entity and relation models plus the compiled graph-walker implementation.
The generator records the sampled walks and masked graphs without changing the
model's computation. DICE then replays those walks through its own model and
compares predictions and every parameter gradient.

| Official checkpoint | Parameters | Maximum score error | Maximum gradient error |
|---|---:|---:|---:|
| `flock_entity.pth` | 801,969 | 0 | 0 |
| `flock_relation.pth` | 810,289 | 0 | 0 |

These are CPU float32 comparisons **on identical recorded walks**, using the
published dimensions and two base walks per refinement on the fixture graph.
They cover entity head/tail scores, relation scores, and training. Exact equality
on these fixtures is not a claim that independently sampled predictions coincide
or that DICE reproduces the paper's benchmark metrics.

Separate tests compare anonymization exactly and first-step, non-backtracking,
edge-type, and direction distributions statistically against the compiled walker.
The released parser uses a boolean queue for directions, so self-loop marker `2`
is emitted as `1`; DICE preserves that checkpoint convention. Tests also cover
isolated nodes, parallel relations, empty masked graphs, permutation equivariance
with correspondingly renamed walks, stochastic averaging, CUDA, inverse-ID
mapping, checkpoint errors, training objectives, and experiment reloads.

```bash
# Offline fixtures and integration checks; no upstream dependencies required.
python -m pytest tests/test_flock.py tests/test_trix.py tests/test_ultra.py -q

# Include both published Flock checkpoints, verified by SHA-256.
FLOCK_CHECKPOINT_DIR=checkpoints/flock python -m pytest tests/test_flock.py -q
```

To regenerate fixtures, create an isolated environment with PyTorch, PyG 2.4.0,
a matching torch-scatter build, easydict, pybind11, and the pinned repository's
`graph-walker` package. Run the tasks in separate processes because the official
packages have the same import name:

```bash
PYTHONPATH=/path/to/flock/src_entity python tests/fixtures/flock/generate_reference.py \
  /path/to/flock tests/fixtures/flock entity
PYTHONPATH=/path/to/flock/src_relation python tests/fixtures/flock/generate_reference.py \
  /path/to/flock tests/fixtures/flock relation
```

The generator checks the revision and verifies that tracked model/walker source
files are unmodified. For custom replay, `sample_walks()` returns seven record
tensors shaped `[refinements, batch, walks, length]`; `score_walks()` consumes
these with **internal** relation IDs. Its docstring specifies the marker values
and query layout. Training callers supplying their own records must remove target
facts before constructing them.
