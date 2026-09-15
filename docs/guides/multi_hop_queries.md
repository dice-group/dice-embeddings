# Multi-Hop Query Answering

This guide explains how to use DICE Embeddings for complex multi-hop query answering using EPFO (Existential Positive First-Order) queries.

## Table of Contents

- [Overview](#overview)
- [Query Types](#query-types)
- [Usage Examples](#usage-examples)
- [Advanced Patterns](#advanced-patterns)
- [Performance Tips](#performance-tips)

---

## Overview

Multi-hop query answering allows you to ask complex questions that require reasoning over multiple relationships in a knowledge graph. DICE Embeddings supports **9 query types** covering projections, intersections, and unions.

### Supported Query Types

| Query Type | Name | Description | Complexity |
|------------|------|-------------|------------|
| `1p` | 1-hop Projection | Single relation traversal | ⭐ Simple |
| `2p` | 2-hop Projection | Chain of 2 relations | ⭐⭐ Medium |
| `3p` | 3-hop Projection | Chain of 3 relations | ⭐⭐ Medium |
| `2i` | 2-way Intersection | AND of 2 relations | ⭐⭐ Medium |
| `3i` | 3-way Intersection | AND of 3 relations | ⭐⭐⭐ Complex |
| `ip` | Intersection-Projection | AND then chain | ⭐⭐⭐ Complex |
| `pi` | Projection-Intersection | Chain then AND | ⭐⭐⭐ Complex |
| `2u` | 2-way Union | OR of 2 patterns | ⭐⭐⭐ Complex |
| `up` | Union-Projection | OR then chain | ⭐⭐⭐ Complex |

---

## Query Types

### 1-hop Projection (`1p`)

**Logical Form:** `?x : ∃x. r(anchor, x)`

**Natural Language:** "What entities are related to the anchor via relation r?"

**Example:**
```python
from dicee import KGE

model = KGE(path="...")

# Question: Who are the siblings of F9M167?
# Query: ?x : hasSibling(F9M167, x)
predictions = model.answer_multi_hop_query(
    query_type="1p",
    query=("F9M167", ("hasSibling",)),
    tnorm="min",
    k=5
)

# Returns: [(F9M157, 0.98), (F9F141, 0.96), ...]
```

**Diagram:**
```
[Anchor] --relation--> [?]
F9M167   --hasSibling-> ?
```

---

### 2-hop Projection (`2p`)

**Logical Form:** `?x : ∃y. r1(anchor, y) ∧ r2(y, x)`

**Natural Language:** "Follow relation r1, then relation r2"

**Example:**
```python
# Question: Who is married to a sibling of F9M167?
# Query: ?x : ∃y. hasSibling(F9M167, y) ∧ married(y, x)
predictions = model.answer_multi_hop_query(
    query_type="2p",
    query=("F9M167", ("hasSibling", "married")),
    tnorm="min",
    k=5
)

# Returns: [(F9F158, 0.95), (F9M142, 0.93), ...]
```

**Diagram:**
```
[Anchor] --r1--> [Intermediate] --r2--> [?]
F9M167   --hasSibling-> y --married-> ?
```

---

### 3-hop Projection (`3p`)

**Logical Form:** `?x : ∃y,z. r1(anchor, y) ∧ r2(y, z) ∧ r3(z, x)`

**Natural Language:** "Follow a chain of 3 relations"

**Example:**
```python
# Question: What types are married to a sibling of F9M167?
# Query: ?x : ∃y,z. hasSibling(F9M167, y) ∧ married(y, z) ∧ type(z, x)
predictions = model.answer_multi_hop_query(
    query_type="3p",
    query=("F9M167", ("hasSibling", "married", "rdf:type")),
    tnorm="min",
    k=5
)

# Returns: [(Person, 0.99), (Male, 0.98), (Father, 0.95), ...]
```

**Diagram:**
```
[Anchor] --r1--> [y] --r2--> [z] --r3--> [?]
F9M167   --hasSibling-> y --married-> z --type-> ?
```

---

### 2-way Intersection (`2i`)

**Logical Form:** `?x : r1(anchor1, x) ∧ r2(anchor2, x)`

**Natural Language:** "Entities satisfying both conditions"

**Example:**
```python
# Question: Who is both a child of PersonA AND a spouse of PersonB?
# Query: ?x : hasChild(PersonA, x) ∧ married(PersonB, x)
predictions = model.answer_multi_hop_query(
    query_type="2i",
    query=(("PersonA", ("hasChild",)), 
           ("PersonB", ("married",))),
    tnorm="min",
    k=5
)
```

**Diagram:**
```
[Anchor1] --r1--> [?] <--r2-- [Anchor2]
PersonA   --hasChild-> ? <-married- PersonB
```

---

### 3-way Intersection (`3i`)

**Logical Form:** `?x : r1(anchor1, x) ∧ r2(anchor2, x) ∧ r3(anchor3, x)`

**Natural Language:** "Entities satisfying all three conditions"

**Example:**
```python
# Question: Who is a friend of A, colleague of B, and neighbor of C?
# Query: ?x : friend(A, x) ∧ colleague(B, x) ∧ neighbor(C, x)
predictions = model.answer_multi_hop_query(
    query_type="3i",
    query=(("A", ("friend",)), 
           ("B", ("colleague",)), 
           ("C", ("neighbor",))),
    tnorm="min",
    k=5
)
```

**Diagram:**
```
[A] --friend--> [?] <--colleague-- [B]
                 ^
                 |
            neighbor
                 |
                [C]
```

---

### Intersection-Projection (`ip`)

**Logical Form:** `?x : ∃y. (r1(anchor1, y) ∧ r2(anchor2, y)) ∧ r3(y, x)`

**Natural Language:** "Find intersection, then project"

**Example:**
```python
# Question: What are the parents of people who are both:
#           - children of PersonA AND friends of PersonB?
# Query: ?x : ∃y. (hasChild(PersonA, y) ∧ friend(PersonB, y)) ∧ hasParent(y, x)
predictions = model.answer_multi_hop_query(
    query_type="ip",
    query=((("PersonA", ("hasChild",)), 
            ("PersonB", ("friend",))), 
           ("hasParent",)),
    tnorm="min",
    k=5
)
```

**Diagram:**
```
[Anchor1] --r1--> [y] <--r2-- [Anchor2]
                  |
                  r3
                  ↓
                 [?]
```

---

### Projection-Intersection (`pi`)

**Logical Form:** `?x : ∃y,z. r1(anchor, y) ∧ r2(y, x) ∧ r3(anchor, z) ∧ r4(z, x)`

**Natural Language:** "Two paths from anchor that converge"

**Example:**
```python
# Question: Who is reachable from F9M167 via two different paths?
# Path 1: F9M167 --hasSibling-> y --hasChild-> ?
# Path 2: F9M167 --hasSpouse-> z --hasChild-> ?
predictions = model.answer_multi_hop_query(
    query_type="pi",
    query=(("F9M167", ("hasSibling", "hasChild")),
           ("F9M167", ("hasSpouse", "hasChild"))),
    tnorm="min",
    k=5
)
```

**Diagram:**
```
         --r1--> [y] --r2--> [?]
        /                    ^
[Anchor]                     |
        \                    |
         --r3--> [z] --r4----+
```

---

### 2-way Union (`2u`)

**Logical Form:** `?x : r1(anchor1, x) ∨ r2(anchor2, x)`

**Natural Language:** "Entities satisfying either condition"

**Example:**
```python
# Question: Who is either a friend of PersonA OR a colleague of PersonB?
# Query: ?x : friend(PersonA, x) ∨ colleague(PersonB, x)
predictions = model.answer_multi_hop_query(
    query_type="2u",
    query=(("PersonA", ("friend",)), 
           ("PersonB", ("colleague",))),
    tnorm="min",
    k=10
)
```

**Diagram:**
```
[PersonA] --friend--> [?]
                      OR
[PersonB] --colleague-> [?]
```

---

### Union-Projection (`up`)

**Logical Form:** `?x : ∃y. (r1(anchor1, y) ∨ r2(anchor2, y)) ∧ r3(y, x)`

**Natural Language:** "Union first, then project"

**Example:**
```python
# Question: What are the spouses of people who are either:
#           - friends of PersonA OR colleagues of PersonB?
# Query: ?x : ∃y. (friend(PersonA, y) ∨ colleague(PersonB, y)) ∧ married(y, x)
predictions = model.answer_multi_hop_query(
    query_type="up",
    query=((("PersonA", ("friend",)), 
            ("PersonB", ("colleague",))), 
           ("married",)),
    tnorm="min",
    k=5
)
```

**Diagram:**
```
[PersonA] --friend--> [y]
                      OR  --married--> [?]
[PersonB] --colleague-> [y]
```

---

## Usage Examples

### Complete Training & Query Workflow

```python
from dicee.executer import Execute
from dicee.config import Namespace
from dicee import KGE

# Step 1: Train a model
args = Namespace()
args.model = 'Keci'
args.optim = 'Adam'
args.scoring_technique = "AllvsAll"
args.path_single_kg = "KGs/Family/family-benchmark_rich_background.owl"
args.backend = "rdflib"
args.num_epochs = 200
args.batch_size = 1024
args.lr = 0.1
args.embedding_dim = 512

result = Execute(args).start()

# Step 2: Load trained model
model = KGE(path=result['path_experiment_folder'])

# Step 3: Answer queries
# 1-hop: Who are the siblings of F9M167?
q1 = model.answer_multi_hop_query(
    query_type="1p",
    query=("http://www.benchmark.org/family#F9M167",
           ("http://www.benchmark.org/family#hasSibling",)),
    tnorm="min", k=3
)
print("Siblings:", q1)

# 2-hop: To whom is a sibling married?
q2 = model.answer_multi_hop_query(
    query_type="2p",
    query=("http://www.benchmark.org/family#F9M167",
           ("http://www.benchmark.org/family#hasSibling",
            "http://www.benchmark.org/family#married")),
    tnorm="min", k=3
)
print("Sibling spouses:", q2)

# 3-hop: What types of people are married to siblings?
q3 = model.answer_multi_hop_query(
    query_type="3p",
    query=("http://www.benchmark.org/family#F9M167",
           ("http://www.benchmark.org/family#hasSibling",
            "http://www.benchmark.org/family#married",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")),
    tnorm="min", k=5
)
print("Types:", q3)
```

### Batch Query Processing

```python
# Define multiple queries
queries = [
    ("1p", ("EntityA", ("relation1",))),
    ("2p", ("EntityB", ("relation2", "relation3"))),
    ("2i", (("EntityC", ("relation4",)), ("EntityD", ("relation5",)))),
]

results = {}
for query_type, query in queries:
    predictions = model.answer_multi_hop_query(
        query_type=query_type,
        query=query,
        tnorm="min",
        k=10
    )
    results[f"{query_type}_{query}"] = predictions

# Analyze results
for query_id, preds in results.items():
    print(f"\n{query_id}:")
    for entity, score in preds[:3]:
        print(f"  {entity}: {score:.4f}")
```

---

## Advanced Patterns

### T-norm Selection

The `tnorm` parameter controls how conjunctions are computed:

```python
# Minimum t-norm (default, most commonly used)
model.answer_multi_hop_query(..., tnorm="min")

# Product t-norm (multiplicative semantics)
model.answer_multi_hop_query(..., tnorm="prod")
```

**When to use:**
- **`min`**: Standard choice, preserves highest score in conjunction
- **`prod`**: More strict, all conditions must have high scores

### Filtering Results

```python
predictions = model.answer_multi_hop_query(
    query_type="2p",
    query=("EntityA", ("relation1", "relation2")),
    tnorm="min",
    k=100  # Get more candidates
)

# Filter by score threshold
high_confidence = [(e, s) for e, s in predictions if s > 0.8]

# Filter by entity type
def is_person(entity):
    return entity.startswith("Person")

people_only = [(e, s) for e, s in predictions if is_person(e)]
```

### Query Composition

```python
# Build complex queries programmatically
def find_common_friends(person1, person2, k=5):
    """Find people who are friends with both person1 and person2."""
    return model.answer_multi_hop_query(
        query_type="2i",
        query=((person1, ("hasFriend",)), 
               (person2, ("hasFriend",))),
        tnorm="min",
        k=k
    )

def find_friends_of_friends(person, k=5):
    """Find friends of friends."""
    return model.answer_multi_hop_query(
        query_type="2p",
        query=(person, ("hasFriend", "hasFriend")),
        tnorm="min",
        k=k
    )

# Use the functions
common = find_common_friends("Alice", "Bob")
fof = find_friends_of_friends("Alice")
```

---

## Performance Tips

### 1. Model Selection

**Best models for multi-hop reasoning:**
- ✅ `Keci` — Clifford algebra, strong reasoning
- ✅ `ComplEx` — Complex embeddings, good for symmetric relations
- ✅ `QMult` / `OMult` — Quaternion/Octonion, high capacity
- ⚠️ `TransE` — Simpler, may struggle with complex patterns

### 2. Training Configuration

```bash
# Use AllvsAll for better multi-hop performance
dicee --dataset_dir KGs/Family --model Keci --scoring_technique AllvsAll

# Higher embedding dimension for complex reasoning
dicee --dataset_dir KGs/Family --model Keci --embedding_dim 512

# More epochs for convergence
dicee --dataset_dir KGs/Family --model Keci --num_epochs 200
```

### 3. Query Optimization

```python
# Start with smaller k, expand if needed
predictions = model.answer_multi_hop_query(..., k=10)  # Fast
if not enough_results(predictions):
    predictions = model.answer_multi_hop_query(..., k=100)  # Slower

# Cache intermediate results for repeated queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_query(query_type, query_tuple, k):
    return tuple(model.answer_multi_hop_query(
        query_type=query_type,
        query=query_tuple,
        tnorm="min",
        k=k
    ))
```

### 4. Benchmarking

```python
import time

def benchmark_query(query_type, query, k=10, n_runs=10):
    """Benchmark query performance."""
    times = []
    for _ in range(n_runs):
        start = time.time()
        model.answer_multi_hop_query(
            query_type=query_type,
            query=query,
            tnorm="min",
            k=k
        )
        times.append(time.time() - start)
    
    print(f"Query type {query_type}:")
    print(f"  Mean: {np.mean(times):.4f}s")
    print(f"  Std:  {np.std(times):.4f}s")
    print(f"  Min:  {np.min(times):.4f}s")
    print(f"  Max:  {np.max(times):.4f}s")
```

---

## Common Patterns & Use Cases

### Social Network Analysis

```python
# Find mutual friends
mutual_friends = model.answer_multi_hop_query(
    query_type="2i",
    query=(("Alice", ("friend",)), ("Bob", ("friend",))),
    tnorm="min", k=10
)

# Find friend recommendations (friends of friends, excluding direct friends)
fof = model.answer_multi_hop_query(
    query_type="2p",
    query=("Alice", ("friend", "friend")),
    tnorm="min", k=20
)
```

### Knowledge Graph Completion

```python
# Find potential new facts
# Question: What companies might PersonX work for?
# Based on: PersonX's colleagues work for CompanyY
potential_employers = model.answer_multi_hop_query(
    query_type="2p",
    query=("PersonX", ("colleague", "worksFor")),
    tnorm="min", k=5
)
```

### Biomedical Reasoning

```python
# Find drugs that treat diseases via protein interactions
# Query: ?drug : ∃protein. treats(drug, protein) ∧ causedBy(disease, protein)
candidate_drugs = model.answer_multi_hop_query(
    query_type="pi",
    query=(("Drug", ("treats", "interactsWith")),
           ("Disease", ("causedBy", "interactsWith"))),
    tnorm="min", k=10
)
```

---

## Limitations & Considerations

### Limitations

1. **Negation not supported** — Cannot express "NOT" queries
2. **Numerical constraints not supported** — Cannot express "> 5" or "between X and Y"
3. **Recursive queries not supported** — Cannot express "transitive closure"
4. **Performance degrades with query complexity** — 3-hop queries slower than 1-hop

### Quality Considerations

- **Model quality matters** — Low training MRR → poor query answering
- **Query complexity vs accuracy tradeoff** — More hops = more error propagation
- **Relation coverage** — Rare relations in training → poor query performance
- **Entity frequency** — Rare entities → lower confidence scores

---

## Troubleshooting

### Low confidence scores

**Problem:** All predictions have scores < 0.1

**Solutions:**
1. Train longer or with better hyperparameters
2. Use AllvsAll scoring technique
3. Increase embedding dimension
4. Verify query syntax is correct

### No results returned

**Problem:** Empty result list

**Solutions:**
1. Check entity/relation names match training data exactly
2. Verify entities exist in knowledge graph
3. Increase `k` parameter
4. Check model was trained on relevant relations

### Slow query execution

**Problem:** Queries take too long

**Solutions:**
1. Reduce `k` parameter
2. Use simpler query types when possible
3. Ensure model is on GPU: `model.to("cuda")`
4. Cache repeated query patterns

---

## Further Reading

- **Tests:** [test_answer_multi_hop_query.py](../../tests/test_answer_multi_hop_query.py) — Complete examples
- **Research:** [Query2Box paper](https://arxiv.org/abs/2002.05969) — Theoretical foundation
- **Benchmarks:** [examples/multi_hop_query_answering/](../../examples/multi_hop_query_answering/) — Performance evaluation

---

## Summary

**Quick Reference:**

| Need | Use Query Type |
|------|----------------|
| Follow one relation | `1p` |
| Follow chain of relations | `2p`, `3p` |
| Find entities satisfying AND conditions | `2i`, `3i` |
| Find entities satisfying OR conditions | `2u` |
| Complex patterns | `ip`, `pi`, `up` |

**Best Practices:**
- ✅ Start with simple queries (1p, 2p)
- ✅ Use `tnorm="min"` as default
- ✅ Train with `AllvsAll` for best reasoning
- ✅ Use high `embedding_dim` (≥ 256)
- ✅ Validate query syntax carefully
- ✅ Benchmark on known ground truth first
