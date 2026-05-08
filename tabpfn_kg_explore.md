# TabPFN as a Knowledge Graph Reasoner

## Overview

A **non-parametric, in-context link predictor** for knowledge graphs.
There are no learned embeddings, no gradient descent, no weight matrices trained
on the KG itself.  Instead, the structural topology of the KG is encoded as a
tabular feature matrix, and **TabPFN** — a pre-trained transformer that performs
Bayesian inference in-context — acts as the classifier at each reasoning stage.

---

## 1. Knowledge Graph Representation

A knowledge graph $\mathcal{G} = (\mathcal{E}, \mathcal{R}, \mathcal{T})$
consists of entities $\mathcal{E}$, relations $\mathcal{R}$, and observed triples
$\mathcal{T} \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E}$.

### 3-D Binary Adjacency Tensor

$$\mathbf{T} \in \{0,1\}^{|\mathcal{E}| \times |\mathcal{E}| \times |\mathcal{R}|},
\qquad
\mathbf{T}_{i,k,j} = \mathbf{1}\bigl[(e_i, r_j, e_k) \in \mathcal{T}\bigr]$$

Memory footprint: $|\mathcal{E}|^2 \cdot |\mathcal{R}|$ bytes (`uint8`).

### Entity Feature Vectors — Pluggable Strategy

Every downstream stage consumes only the entity feature matrix
$\Phi \in \mathbb{R}^{|\mathcal{E}| \times d}$, where row $i$ is
$\phi(e_i)$.  The feature strategy is fully interchangeable via the
`EntityFeaturiser` abstract interface and the `--features` CLI flag.

#### `DegreeFeaturiser` (`--features degree`, default)

Rather than flattening $\mathbf{T}[i,:,:]$, each entity is described by
its **relational degree profile**:

$$d^{\text{out}}_{i,j} = \sum_{k} \mathbf{T}_{i,k,j}
\qquad\text{(out-degree of } e_i \text{ under } r_j\text{)}$$

$$d^{\text{in}}_{i,j} = \sum_{k} \mathbf{T}_{k,i,j}
\qquad\text{(in-degree of } e_i \text{ under } r_j\text{)}$$

$$\phi(e_i) = \bigl[d^{\text{out}}_{i,\cdot} \;\|\; d^{\text{in}}_{i,\cdot}\bigr]
\in \mathbb{R}^{2|\mathcal{R}|}$$

This is a **structural fingerprint** — no entity names or semantic types are
used, only graph topology.  No external model required.

#### `KGEFeaturiser` (`--features kge`)

Uses the entity embedding table of a pre-trained **dicee** KGE model
(e.g., Keci, ComplEx, DistMult).  The feature dimensionality equals the
model's `embedding_dim`:

$$\phi(e_i) = \mathbf{E}[i] \in \mathbb{R}^{d_{\text{emb}}}$$

Requires `--kge_path` pointing to a dicee experiment folder
(`model.pt` + `configuration.json`).

#### Adding a new strategy

1. Subclass `EntityFeaturiser` and implement `fit_transform(T, e2i, i2e) -> np.ndarray`.
2. Register the name in `build_featuriser()`.
3. Pass it via `--features <name>`.

---

## 2. The Three-Stage Pipeline

### Stage 1 — Relation Predictor: $P(r_j \mid e_i)$

> *Given a head entity, which relations does it typically participate in?*

$$P(r_j \mid e_i) \approx \text{TabPFN}_{\text{rel}}\bigl(\phi(e_i)\bigr)$$

- **Input features:** $\phi(e_i) \in \mathbb{R}^{2|\mathcal{R}|}$
- **Target:** relation index $r_j$
- **Training set:** all triples $\{(\phi(e_i),\, r_j)\}_{(e_i,r_j,e_k)\in\mathcal{T}}$
- **Result on UMLS:** accuracy **0.22** (random baseline = 0.022 for 46 classes)

Because $|\mathcal{R}|$ can exceed TabPFN's class limit of 10, the label set is
partitioned into $\lceil |\mathcal{R}| / 10 \rceil$ chunks and one
`TabPFNClassifier` is trained per chunk (`MultiTabPFNClassifier`).

---

### Stage 2 — Tail Predictor: $P(e_k \mid e_i, r_j)$

> *Given a head entity and a fixed relation, who is the tail?*

$$P(e_k \mid e_i, r_j) \approx \text{TabPFN}_{\text{tail}}^{(j)}\bigl(\phi(e_i)\bigr)$$

- **Input features:** $\phi(e_i) \in \mathbb{R}^{2|\mathcal{R}|}$
- **Target:** tail entity $e_k$
- **Training set:** triples restricted to relation $r_j$:
  $\mathcal{T}_j = \{(e_i, e_k) \mid (e_i, r_j, e_k) \in \mathcal{T}\}$
- **One classifier per relation** — 43 / 46 trained on UMLS (3 skipped: too few
  unique tail classes or constant features)

---

### Stage 3 — Pair-to-Relation Predictor: $P(r_j \mid e_i, e_k)$

> *Given both endpoints of a potential triple, what relation connects them?*

$$P(r_j \mid e_i, e_k) \approx \text{TabPFN}_{\text{pair}}\bigl(\psi(e_i, e_k)\bigr),
\qquad \psi(e_i, e_k) = [\phi(e_i) \;\|\; \phi(e_k)] \in \mathbb{R}^{4|\mathcal{R}|}$$

- **Input features:** concatenated degree profiles of both endpoints
- **Target:** relation index $r_j$
- **Result on UMLS:** accuracy **0.58** — nearly 3× better than Stage 1

The large gap (0.22 → 0.58) shows that most UMLS relations are determined by
the *type combination* of both endpoints, not the head alone.

---

## 3. Inference Queries

| Query | Formula | CLI mode |
|---|---|---|
| Tail prediction (marginal) | $P(e_k \mid e_i) = \sum_j P(r_j \mid e_i)\, P(e_k \mid e_i, r_j)$ | `tail_predict` |
| Link prediction | $\text{score}(e_k) = P(r_j \mid e_i) \cdot P(e_k \mid e_i, r_j)$ | `link_predict` |
| Relation prediction | $P(r_j \mid e_i)$ from Stage 1 | `relation_predict` |
| Evaluation | MRR, Hits@1/3/10 on `test.txt` | `evaluate` |

### Law of Total Probability (Tail Prediction)

The marginal tail score integrates over all relations:

$$P(e_k \mid e_i) = \sum_{j=1}^{|\mathcal{R}|} P(r_j \mid e_i)\; P(e_k \mid e_i, r_j)$$

### Triple Score

$$\text{score}(e_i, r_j, e_k) = P(r_j \mid e_i) \cdot P(e_k \mid e_i, r_j)$$

### Ranking for Evaluation

For each test triple $(e_i, r_j, e_k)$, all $|\mathcal{E}|$ candidate tails are
scored and the true tail is ranked:

$$\text{rank}(e_k) = 1 + \bigl|\{ e' : \text{score}(e_i, r_j, e') > \text{score}(e_i, r_j, e_k) \}\bigr|$$

$$\text{MRR} = \frac{1}{|\mathcal{T}_{\text{test}}|} \sum_{(e_i,r_j,e_k)} \frac{1}{\text{rank}(e_k)}$$

$$\text{Hits@}k = \frac{1}{|\mathcal{T}_{\text{test}}|} \sum_{(e_i,r_j,e_k)} \mathbf{1}[\text{rank}(e_k) \leq k]$$

---

## 4. What TabPFN Brings

TabPFN is a **prior-fitted network** meta-trained on millions of synthetic
classification tasks to approximate a Bayesian posterior
$P(y \mid x, \mathcal{D}_{\text{train}})$ **without any gradient steps at test time**.
Fitting is a single transformer forward pass that treats the training set as
in-context examples.

Key properties exploited here:
- Works on **small, tabular, low-dimensional** datasets — exactly what degree
  feature slices of a small KG produce
- No hyperparameter tuning needed
- Naturally outputs calibrated probabilities

### `MultiTabPFNClassifier`

Wraps the 10-class limit by partitioning $C$ classes into
$B = \lceil C / 10 \rceil$ chunks, training one `TabPFNClassifier` per chunk,
and re-normalising:

$$P(c \mid x) = \frac{\tilde{p}_c}{\sum_{c'} \tilde{p}_{c'}},
\qquad \tilde{p}_c = f_{b(c)}(x)_c$$

---

## 5. Limitations & Improvement Plan

| # | Limitation | Root cause | Status |
|---|---|---|---|
| L1 | **Purely structural features** | $\phi(e_i)$ encodes only degree counts | ✅ **Fixed** — `KGEFeaturiser` adds `--features kge` |
| L2 | **Closed-world** | Unseen entities have no feature vector | Open — inductive extensions (e.g., GNN-derived features) |
| L3 | **Tail coverage gap** | Stage 2 skips 3/46 relations; entities outside seen classes score 0 | Open — `MultiTabPFNClassifier` already handles chunking |
| L4 | **No negatives** | Only positive triples are used for training | Open — add corrupted triples as negatives in Stage 2 |
| L5 | **Scale** | TabPFN's 10K sample cap | Open — subsample + ensemble, or scalable classifier for large KGs |

### Evaluation Results on UMLS

All numbers are raw (non-filtered) tail ranking on `test.txt`.

| Model | Features | MRR | MR | Hits@1 | Hits@3 | Hits@10 | Note |
|---|---|---|---|---|---|---|---|
| Random | — | ~0.007 | ~68 | ~0.007 | ~0.022 | ~0.074 | literature estimate |
| TransE | — | ~0.83 | — | — | — | ~0.99 | literature |
| **TabPFN** | `degree` | TBD | TBD | TBD | TBD | TBD | run full evaluate |
| **TabPFN** | `kge` (Keci 64d) | **0.211** | **6.8** | 0.000 | 0.300 | **0.600** | 10-triple sample |

> ⚠️ The KGE row is based on 10 randomly sampled test triples (`--n_test 10`).
> Run `--mode evaluate` without `--n_test` for the full 661-triple result.

### Run full evaluation

```bash
# Degree features
python tabpfn_kg_explore.py --dataset_dir KGs/UMLS --features degree \
    --save --cache umls_degree.pkl
python tabpfn_kg_explore.py --load --cache umls_degree.pkl \
    --mode evaluate --dataset_dir KGs/UMLS

# KGE (Keci) features
python tabpfn_kg_explore.py --load --cache umls_kge.pkl \
    --mode evaluate --dataset_dir KGs/UMLS
```

---

## 6. CLI Reference

### Feature strategy flags

| Flag | Value | Description |
|---|---|---|
| `--features` | `degree` (default) | Relational degree profile, no external model |
| `--features` | `kge` | Entity embeddings from a pre-trained dicee model |
| `--kge_path` | path | Required with `--features kge`; path to dicee experiment folder |

### Common workflows

```bash
# Train with default (degree) features and save
python tabpfn_kg_explore.py --dataset_dir KGs/UMLS --save --cache umls_degree.pkl

# Train with Keci KGE features
# Step 1: train a Keci model
dicee --dataset_dir KGs/UMLS --model Keci --embedding_dim 64 \
      --num_epochs 200 --scoring_technique KvsAll \
      --path_to_store_single_run Experiments/Keci_UMLS
# Step 2: build pipeline
python tabpfn_kg_explore.py --dataset_dir KGs/UMLS \
    --features kge --kge_path Experiments/Keci_UMLS \
    --save --cache umls_kge.pkl

# Evaluate (subsample for speed)
python tabpfn_kg_explore.py --load --cache umls_kge.pkl \
    --mode evaluate --dataset_dir KGs/UMLS --n_test 100

# Full evaluation
python tabpfn_kg_explore.py --load --cache umls_kge.pkl \
    --mode evaluate --dataset_dir KGs/UMLS

# P(e_k | e_i) marginalised over relations
python tabpfn_kg_explore.py --load --cache umls_kge.pkl \
    --mode tail_predict --query_entity "enzyme" --top_k 10

# P(r_j | e_i) — which relations does enzyme participate in?
python tabpfn_kg_explore.py --load --cache umls_kge.pkl \
    --mode relation_predict --query_entity "enzyme" --top_k 10

# P(e_k | e_i, r_j) — link prediction given (head, relation)
python tabpfn_kg_explore.py --load --cache umls_kge.pkl \
    --mode link_predict --query_entity "enzyme" \
    --query_relation "disrupts" --top_k 10
```
