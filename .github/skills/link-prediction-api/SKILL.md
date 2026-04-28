---
name: link-prediction-api
description: "Use a pre-trained KGE model for inference via the KGE class. Use when: loading a trained model, predicting missing links (head/relation/tail), answering multi-hop EPFO queries (1p 2p 3p 2i 3i ip pi 2u up), predicting literal values, accessing raw embeddings, deploying with the Gradio UI."
argument-hint: "Describe what you want to predict: missing head, tail, relation, or a multi-hop query type"
---

# Use a Pre-Trained KGE Model for Inference

## When to Use
- Loading a trained model to predict missing triples
- Running 1p / 2p / 3p / intersection / union / negation queries
- Extracting entity or relation embeddings
- Predicting literal values (numerical attributes) with `train_literals`
- Deploying the model as a web UI via `index_serve.py`

## Reference Implementation
- [`dicee/knowledge_graph_embeddings.py`](../../../dicee/knowledge_graph_embeddings.py) — `KGE` class
- [`dicee/scripts/index_serve.py`](../../../dicee/scripts/index_serve.py) — Gradio deployment

---

## 1. Load a Pre-Trained Model

```python
from dicee import KGE

# Load from an experiment folder (must contain model.pt + configuration.json)
model = KGE(path="Experiments/2024-01-01_12-00/")

# Or load from a custom path
model = KGE(path="MyRun/")
```

The `path` directory must contain:
- `model.pt` — model weights
- `configuration.json` — training config
- `entity_to_idx.csv` — entity index mapping
- `relation_to_idx.csv` — relation index mapping

---

## 2. Predict Missing Links — `predict_topk`

`predict_topk` infers the missing element in a triple. Supply exactly two of `h`, `r`, `t`.

```python
# (h, r, ?) — predict missing TAIL entity
results = model.predict_topk(h=["Alice"], r=["knows"], topk=5)
# Returns: [[ ("Bob", 0.95), ("Carol", 0.88), ... ]]

# (?, r, t) — predict missing HEAD entity
results = model.predict_topk(r=["knows"], t=["Bob"], topk=5)

# (h, ?, t) — predict missing RELATION
results = model.predict_topk(h=["Alice"], t=["Bob"], topk=5)

# Batch query — pass lists of strings
results = model.predict_topk(h=["Alice", "Dave"], r=["knows", "friendOf"], topk=3)
# Returns list of B result lists (one per h-r pair)

# Restrict candidates with `within`
results = model.predict_topk(h=["Alice"], r=["knows"],
                              within=["Bob", "Carol", "Eve"], topk=3)
```

**Notes:**
- Scores are sigmoid-normalised (0–1)
- `batch_size` controls memory during inference (default: 1024)
- Input strings must match the KG's entity/relation vocabulary exactly

---

## 3. Check Vocabulary

```python
# Is an entity / relation in the model's vocabulary?
model.is_seen(entity="Alice")        # True / False
model.is_seen(relation="knows")      # True / False

# Sample random entities
model.sample_entity(n=10)             # List[str]

# Vocabulary sizes
len(model.entity_to_idx)              # number of entities
len(model.relation_to_idx)            # number of relations
```

---

## 4. Access Raw Embeddings

```python
# Get entity embeddings as list of floats
vecs = model.get_transductive_entity_embeddings(
    indices=["Alice", "Bob"],
    as_list=True       # default — List[List[float]]
)

# As numpy array
vecs = model.get_transductive_entity_embeddings(
    indices=["Alice", "Bob"],
    as_numpy=True
)

# As PyTorch tensor
vecs = model.get_transductive_entity_embeddings(
    indices=["Alice", "Bob"],
    as_pytorch=True
)
```

---

## 5. Multi-Hop Query Answering — `answer_multi_hop_query`

Answers EPFO (Existential Positive First-Order) queries including conjunctions, disjunctions, and negations.

### Supported Query Types

| `query_type` | Pattern | Meaning |
|-------------|---------|---------|
| `"1p"` | `(e, (r,))` | Single hop: `?x : r(e, x)` |
| `"2p"` | `(e, (r1, r2))` | Two hops: `?x : ∃y. r1(e,y) ∧ r2(y,x)` |
| `"3p"` | `(e, (r1, r2, r3))` | Three hops |
| `"2i"` | `((e1,(r1,)), (e2,(r2,)))` | Intersection: `?x : r1(e1,x) ∧ r2(e2,x)` |
| `"3i"` | three conjuncts | Three-way intersection |
| `"ip"` | `((2i_query), (r,))` | Intersection then hop |
| `"pi"` | `((2p_query), (e2,(r2,)))` | Hop then intersection |
| `"2in"` | with negation `"n"` | Intersection with negation |
| `"2u"` | with union `"u"` | Disjunction |
| `"up"` | union then hop | Union followed by projection |

### Examples

```python
# 1p: Who does Alice know? (?x : knows(Alice, x))
results = model.answer_multi_hop_query(
    query_type="1p",
    query=("Alice", ("knows",)),
    k=10, tnorm="min"
)

# 2p: Who are the friends-of-friends of Alice?
results = model.answer_multi_hop_query(
    query_type="2p",
    query=("Alice", ("knows", "knows")),
    k=10, tnorm="prod"
)

# 2i: Who is known by BOTH Alice and Bob?
results = model.answer_multi_hop_query(
    query_type="2i",
    query=(("Alice", ("knows",)), ("Bob", ("knows",))),
    k=10, tnorm="min"
)

# Batch of queries at once
results = model.answer_multi_hop_query(
    query_type="1p",
    queries=[
        ("Alice", ("knows",)),
        ("Bob", ("friendOf",)),
    ],
    k=5, tnorm="prod"
)
```

### Parameters

| Param | Type | Description |
|-------|------|-------------|
| `query_type` | str | One of: `1p 2p 3p 2i 3i ip pi 2in 3in inp pin pni 2u up` |
| `query` | tuple | Nested tuple encoding the query (see pattern table above) |
| `queries` | list | List of query tuples for batch evaluation |
| `tnorm` | str | T-norm for conjunction: `"prod"` (product) or `"min"` (Gödel) |
| `neg_norm` | str | Negation norm: `"standard"`, `"sugeno"`, or `"yager"` |
| `k` | int | Top-k answer entities to return |
| `use_logits` | bool | Use raw logits (default: `True`) vs sigmoid probabilities |

---

## 6. Literal Prediction

Train a linear regression head on top of entity embeddings to predict numerical attribute values.

```python
# Train the literal model (requires a TSV file: entity \t attribute \t value)
model.train_literals(train_file_path="KGs/DBpedia/literals_train.tsv")

# Predict attribute values
predictions = model.predict_literals(
    entity=["Berlin", "Paris"],
    attribute=["population", "area"]
)
# Returns: {'Berlin': {'population': 3_500_000, 'area': 891.8}, ...}
```

---

## 7. Evaluate on a Custom Dataset

```python
from dicee import KGE
model = KGE(path="MyRun/")

# Evaluate filtered link prediction metrics (MRR, HITS@k)
triples = [
    ("Alice", "knows", "Bob"),
    ("Carol", "livesIn", "Berlin"),
]
metrics = model.eval_lp_performance(dataset=triples, filtered=True)
print(metrics)  # {'MRR': ..., 'MR': ..., 'HITS@1': ..., 'HITS@3': ..., 'HITS@10': ...}
```

---

## 8. Move Model to Device

```python
model.to("cpu")
model.to("cuda")
model.to("cuda:1")
```

---

## 9. Deploy as a Web UI

Use the built-in Gradio server:

```bash
# Serve from an experiment folder
python -m dicee.scripts.index_serve --path "MyRun/"
```

This launches a Gradio interface at `http://localhost:7860` with:
- Subject / Predicate / Object fields
- 1vsAll scoring (fill two fields, leave one blank)
- Top-k entity scores displayed

---

## 10. Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError: 'Alice'` | Entity not in vocabulary | Check `model.is_seen(entity="Alice")` |
| `AssertionError: k <= num_entities` | `topk` exceeds vocab size | Reduce `topk` |
| `RuntimeError: model.pt not found` | Wrong path | Verify path contains `model.pt` |
| `FileNotFoundError: entity_to_idx` | Missing index files | Retrain or restore from backup |
| Query returns empty results | Wrong query structure | Check query tuple nesting against pattern table |
