# ULTRA in DICE

[Official repository](https://github.com/DeepGraphLearning/ULTRA) · [Paper (ICLR 2024)](https://openreview.net/forum?id=jVEoydFOl9)

ULTRA is a graph-conditioned link predictor implemented in pure PyTorch. It supports
training, fine-tuning, and zero-shot inference with the official `ultra_3g.pth`,
`ultra_4g.pth`, and `ultra_50g.pth` [checkpoints](https://github.com/DeepGraphLearning/ULTRA/tree/427966ad8ed60420eef034063d44f3153addff90/ckpts).

## Evaluate a pretrained checkpoint

Download a checkpoint, then run:

```bash
python -m dicee --model ULTRA --dataset_dir KGs/WN18RR \
  --ultra_checkpoint /path/to/ultra_3g.pth --num_epochs 0 \
  --trainer torchCPUTrainer --scoring_technique NegSample \
  --batch_size 8 --eval_model test \
  --path_to_store_single_run Experiments/ultra-zero-shot
```

ULTRA uses training facts for message passing. Validation and test facts are used
only for filtered evaluation. The same pretrained weights can be used on graphs
with different entity and relation vocabularies.

## Train or fine-tune

Use ULTRA with DICE's existing training objectives:

```bash
python -m dicee --model ULTRA --dataset_dir KGs/UMLS \
  --trainer torchCPUTrainer --scoring_technique KvsAll \
  --optim AdamW --lr 0.0005 --batch_size 8 --num_epochs 10
```

Add `--ultra_checkpoint /path/to/ultra_3g.pth` to fine-tune pretrained weights.
ULTRA hides training targets and their inverse edges during entity message passing.

Official checkpoints use `ultra_dim=64` and `ultra_num_layers=6`, which are the
defaults. These settings can be changed for scratch training. The generic
`embedding_dim` option does not control ULTRA's architecture.

## Load an experiment and predict

```python
from dicee import KGE

kge = KGE(path="Experiments/ultra-zero-shot")
# Replace these names with entities and relations from your dataset.
tail_scores = kge.predict_missing_tail_entity("entity_a", "relation_r")
head_scores = kge.predict_missing_head_entity("relation_r", "entity_b")
```

Keep `ultra_graph.pt` alongside the model, configuration, and vocabulary files
when moving an experiment. ULTRA needs the graph to score queries.

For direct model use, import `ULTRA` from `dicee.models`, load weights with
`load_pretrained(path)`, and attach indexed training triples with `set_graph(...)`.
Triples follow DICE's `(head, relation, tail)` order.

## Tests

```bash
python -m pytest tests/test_ultra.py -q

# Include numerical comparisons for all three official checkpoints.
ULTRA_CHECKPOINT_DIR=/path/to/ULTRA/ckpts python -m pytest tests/test_ultra.py -q
```

Tests compare relation representations, head and tail scores, and training
behavior against upstream reference outputs. They also cover DICE training and
experiment save/reload.
