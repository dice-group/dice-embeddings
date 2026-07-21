---
description: Add a new scoring technique (training dataset class) to the dicee framework
argument-hint: "[describe the new negative sampling strategy or labelling scheme]"
---

# Add a New Scoring Technique to dicee

## Overview

A scoring technique in dicee defines how training labels are constructed from triples. Each technique is implemented as a `torch.utils.data.Dataset` subclass in `dicee/dataset_classes/` and wired into the central factory.

**Key files:**
- `dicee/dataset_classes/_factory.py` — `construct_dataset()` dispatcher
- `dicee/dataset_classes/_negative_sampling.py` — `NegSample`, `FixedNegSample` (reference for triple-level labelling)
- `dicee/dataset_classes/_label_based.py` — `KvsAll`, `1vsAll`, `AllvsAll`, `1vsSample` (reference for entity-level labelling)
- `dicee/models/base_model.py` — `BaseKGE.forward_triples` and `BaseKGE.forward_k_vs_all` contracts

Task for `$ARGUMENTS`: work through the steps below, using the existing classes as the template.

---

## Step 1 — Understand `form_of_labelling`

The factory uses `form_of_labelling` (`"EntityPrediction"` or `"RelationPrediction"`) and `scoring_technique` together to select the dataset class.

There are **two labelling shapes**:

| Shape | Example | `__getitem__` returns | Model method |
|-------|---------|----------------------|--------------|
| Triple-level | NegSample | `(x_batch, y_batch)` where x=(B,3), y=(B,) | `forward_triples(x)` |
| Entity-level | KvsAll | `(x_batch, y_batch)` where x=(B,2), y=(B,\|E\|) | `forward_k_vs_all(x)` |
| Sample-selected | KvsSample | `(x_batch, y_select, y_batch)` | `forward_k_vs_all((x,y_select))` |

Choose the shape that matches the loss semantics.

---

## Step 2 — Implement the Dataset Class

Create the class in the most appropriate file under `dicee/dataset_classes/`:

```python
import torch
import numpy as np
from torch.utils.data import Dataset

class MyNegSamplingDataset(Dataset):
    """
    Custom negative sampling strategy: <describe it here>.

    __getitem__ returns:
        x_batch: LongTensor (B, 3)  — [head_idx, rel_idx, tail_idx]
        y_batch: FloatTensor (B,)   — label per triple (1.0 for positive, 0.0 for negative)
    """

    def __init__(self, train_set: np.ndarray, num_entities: int, num_relations: int,
                 neg_sample_ratio: int, label_smoothing_rate: float = 0.0, seed: int = None):
        self.train_set = torch.tensor(train_set, dtype=torch.long)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.neg_sample_ratio = neg_sample_ratio
        self.label_smoothing_rate = label_smoothing_rate
        if seed is not None:
            torch.manual_seed(seed)

    def __len__(self) -> int:
        return len(self.train_set)

    def __getitem__(self, idx: int):
        triple = self.train_set[idx]   # (3,)
        h, r, t = triple[0], triple[1], triple[2]

        # --- Build negative samples ---
        neg_heads = torch.randint(0, self.num_entities, (self.neg_sample_ratio,))
        neg_triples = torch.stack([neg_heads, r.repeat(self.neg_sample_ratio),
                                   t.repeat(self.neg_sample_ratio)], dim=1)  # (K, 3)

        x = torch.cat([triple.unsqueeze(0), neg_triples], dim=0)  # (1+K, 3)
        y = torch.zeros(1 + self.neg_sample_ratio)
        y[0] = 1.0 - self.label_smoothing_rate  # positive label

        return x, y
```

For entity-level labelling (like `KvsAll`), return `(x, y)` where `x=(B,2)` and `y=(B, num_entities)`.

---

## Step 3 — Export from `__init__.py`

Edit `dicee/dataset_classes/__init__.py`:

```python
from ._negative_sampling import MyNegSamplingDataset  # noqa
```

---

## Step 4 — Wire into `construct_dataset()`

Edit `dicee/dataset_classes/_factory.py`. Add an `elif` branch inside `construct_dataset()`:

```python
elif scoring_technique == "MyNegSample":
    train_set = MyNegSamplingDataset(
        train_set=train_set,
        num_entities=len(entity_to_idx),
        num_relations=len(relation_to_idx),
        neg_sample_ratio=neg_ratio,
        label_smoothing_rate=label_smoothing_rate,
        seed=seed,
    )
```

Also add the import at the top of `_factory.py`:

```python
from ._negative_sampling import ..., MyNegSamplingDataset
```

---

## Step 5 — Ensure Model Has the Right Forward Method

The model must implement the forward method that matches the dataset's output shape.

**Triple-level** (returns `(B,3)` and `(B,)` labels) → model needs `forward_triples`:
```python
def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
    # x: (B, 3)
    ...
    return scores  # (B,)
```

**Entity-level** (returns `(B,2)` and `(B,|E|)` labels) → model needs `forward_k_vs_all`:
```python
def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
    # x: (B, 2)
    ...
    return scores  # (B, num_entities)
```

`BaseKGE` automatically selects the right forward method based on `form_of_labelling`:
- `"EntityPrediction"` → calls `forward_k_vs_all`
- Triple-level techniques → calls `forward_triples`

---

## Step 6 — (Optional) Add Config Parameter

If the technique needs a new hyperparameter, add it to `dicee/config.py`:

```python
self.my_technique_param: float = 0.5
"""Description of my_technique_param"""
```

Pass it into `construct_dataset()` by editing `dicee/executer.py` where `construct_dataset()` is called (search for the call site with `scoring_technique=`).

---

## Step 7 — Write a Test

```python
from dicee.executer import Execute
from dicee.config import Namespace

def test_my_neg_sample():
    args = Namespace()
    args.model = 'Keci'
    args.dataset_dir = 'KGs/UMLS'
    args.scoring_technique = 'MyNegSample'
    args.neg_ratio = 5
    args.num_epochs = 1
    args.embedding_dim = 32
    args.eval_model = 'train_val_test'
    result = Execute(args).start()
    assert result['Train']['MRR'] > 0
```

---

## Reference: Existing Dataset Classes

| Technique | Class | File | Returns |
|-----------|-------|------|---------|
| `NegSample` | `TriplePredictionDataset` | `_negative_sampling.py` | `(B,3), (B,)` |
| `FixedNegSample` | `FixedNegSampleDataset` | `_negative_sampling.py` | `(B,3), (B,)` |
| `1vsAll` | `OnevsAllDataset` | `_label_based.py` | `(B,2), (B,\|E\|)` |
| `KvsAll` | `KvsAll` | `_label_based.py` | `(B,2), (B,\|E\|)` |
| `KvsSample` | `KvsSampleDataset` | `_label_based.py` | `(B,2), select, (B,K)` |
| `AllvsAll` | `AllvsAll` | `_label_based.py` | `(B,2), (B,\|E\|)` |
| `1vsSample` | `OnevsSample` | `_label_based.py` | sampled variant |
| `FSDP1vsSample` | `FSDP1vsSampleDataset` | `_label_based.py` | FSDP-compatible variant |

---

## Common Pitfalls

| Mistake | Fix |
|---------|-----|
| Forgetting to add `elif` in `_factory.py` | Results in `ValueError: Unknown scoring technique` |
| Wrong `form_of_labelling` for entity-level | Must be `"EntityPrediction"` for `forward_k_vs_all` |
| `__getitem__` returns wrong tensor shape | Use existing classes as reference; shapes must be consistent |
| Not exporting from `__init__.py` | Import will fail at runtime in `_factory.py` |
| Missing `neg_ratio` requirement | Add CLI note: `--neg_ratio` required for NegSample variants |
