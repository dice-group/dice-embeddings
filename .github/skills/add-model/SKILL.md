---
name: add-model
description: "Add a new Knowledge Graph Embedding model to the dicee framework. Use when: implementing a new KGE model, adding a custom scoring function, creating a new algebra-based embedding, extending BaseKGE with a new architecture."
argument-hint: "Name and description of the new model (e.g., 'MyModel: a bilinear model with XYZ scoring')"
---

# Add a New KGE Model to dicee

## When to Use
- Implementing a new KGE model (bilinear, translational, convolutional, algebra-based)
- Adding a custom scoring function to the framework
- Extending an existing model with new components

## Reference Implementations
- **Simple bilinear model**: [`dicee/models/real.py`](../../../dicee/models/real.py) → `DistMult`, `TransE`
- **Clifford algebra model**: [`dicee/models/clifford.py`](../../../dicee/models/clifford.py) → `Keci`
- **Convolutional model**: [`dicee/models/quaternion.py`](../../../dicee/models/quaternion.py) → `ConvQ`
- **BaseKGE contract**: [`dicee/models/base_model.py`](../../../dicee/models/base_model.py)

---

## Step-by-Step Procedure

### Step 1 — Create the model file

Create `dicee/models/mymodel.py`. Every model must extend `BaseKGE` and implement two forward methods:

```python
from .base_model import BaseKGE
import torch

class MyModel(BaseKGE):
    def __init__(self, args: dict):
        super().__init__(args)
        self.name = 'MyModel'
        # Access config params via self.args dict:
        #   self.embedding_dim  — already set by BaseKGE
        #   self.num_entities   — already set by BaseKGE
        #   self.num_relations  — already set by BaseKGE
        #   self.args.get("p", 0)  — Clifford p param
        #   self.args.get("q", 1)  — Clifford q param
        # Define additional parameters:
        self.W = torch.nn.Parameter(torch.empty(self.embedding_dim, self.embedding_dim))
        torch.nn.init.xavier_normal_(self.W)

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Score individual triples.
        x: (batch_size, 3) — columns are [head_idx, relation_idx, tail_idx]
        Returns: (batch_size,) — one score per triple
        """
        head_ent_emb = self.entity_embeddings(x[:, 0])      # (B, D)
        rel_emb = self.relation_embeddings(x[:, 1])          # (B, D)
        tail_ent_emb = self.entity_embeddings(x[:, 2])       # (B, D)
        # Implement scoring: e.g., for DistMult:
        return (head_ent_emb * rel_emb * tail_ent_emb).sum(dim=1)

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Score (head, relation) pairs against ALL entities.
        x: (batch_size, 2) — columns are [head_idx, relation_idx]
        Returns: (batch_size, num_entities) — score matrix
        """
        head_ent_emb = self.entity_embeddings(x[:, 0])      # (B, D)
        rel_emb = self.relation_embeddings(x[:, 1])          # (B, D)
        # score against all entity embeddings:
        return (head_ent_emb * rel_emb) @ self.entity_embeddings.weight.T
```

**Important constraints:**
- `args` is a plain `dict` — use `self.args.get("key", default)` for custom params
- `BaseKGE.__init__` initialises `self.entity_embeddings`, `self.relation_embeddings`, the loss function, dropout layers, and normalisation — do not redefine them
- For Clifford models: `embedding_dim / (p + q + 1)` must be an integer; assert this in `__init__`
- Use `self.apply_layer_norm`, `self.apply_batch_norm`, `self.input_dp`, `self.hidden_dp` from `BaseKGE` for consistency

### Step 2 — Register the model

Edit `dicee/models/__init__.py` to export the new model:

```python
# Add to existing imports:
from .mymodel import MyModel  # noqa
```

The model is then selectable via `--model MyModel` on the CLI or `args.model = "MyModel"` in Python.

Also add `MyModel` to the `--model` list in `README.md` (search for the line starting with `` * ```--model `` under "Knowledge Graph Embedding Models").

### Step 3 — (Optional) Add custom config params

If the model needs new hyperparameters, add them to `dicee/config.py` inside the `Namespace.__init__` method:

```python
self.my_param: float = 0.5
"""Description of my_param"""
```

Access them in the model as `self.args.get("my_param", 0.5)`.

### Step 4 — Write tests

Add a test in `tests/` (or extend an existing test file). Minimum test:

```python
from dicee.executer import Execute
from dicee.config import Namespace

def test_my_model():
    args = Namespace()
    args.model = 'MyModel'
    args.dataset_dir = 'KGs/UMLS'
    args.scoring_technique = 'KvsAll'
    args.num_epochs = 1
    args.embedding_dim = 32
    args.eval_model = 'train_val_test'
    result = Execute(args).start()
    assert result['Train']['MRR'] > 0
```

---

## Common Pitfalls

| Mistake | Fix |
|---------|-----|
| Redefining `entity_embeddings` in `__init__` | Don't — `BaseKGE` creates it |
| Using `args.embedding_dim` (namespace-style) | Use `self.embedding_dim` (already parsed) |
| Clifford `r` not integer | Assert `(embedding_dim / (p+q+1)).is_integer()` early |
| Missing `forward_k_vs_all` | Required for `KvsAll` / `1vsAll` / `AllvsAll` scoring |
| Missing `forward_triples` | Required for `NegSample` / `FixedNegSample` scoring |
| Forgetting `# noqa` in `__init__.py` import | Linter removes "unused" imports |

---

## BaseKGE Attributes Available to All Models

```python
self.embedding_dim     # int — embedding vector size
self.num_entities      # int — vocabulary size (entities)
self.num_relations     # int — vocabulary size (relations)
self.entity_embeddings # nn.Embedding(num_entities, embedding_dim)
self.relation_embeddings # nn.Embedding(num_relations, embedding_dim)
self.input_dp          # nn.Dropout(input_dropout_rate)
self.hidden_dp         # nn.Dropout(hidden_dropout_rate)
self.feature_map_dp    # nn.Dropout(feature_map_dropout_rate)
self.loss              # loss function (BCEWithLogitsLoss or CrossEntropyLoss)
self.loss_history      # list of per-epoch losses
self.args              # dict of full config
```

---

## Quick Reference: Existing Models by Complexity

| Complexity | Model | File | Key idea |
|------------|-------|------|----------|
| Simplest | `DistMult` | `real.py` | Hadamard product h⊙r⊙t |
| Translational | `TransE` | `real.py` | Distance ‖h+r−t‖ |
| Complex | `ComplEx` | `complex.py` | Complex vectors |
| Quaternion | `QMult` | `quaternion.py` | Hamilton product |
| Clifford | `Keci` | `clifford.py` | Cl(p,q) algebra |
| Convolutional | `ConvQ` | `quaternion.py` | Conv over quaternion |
| Transformer | `CoKE` | `transformers.py` | Attention over triples |
