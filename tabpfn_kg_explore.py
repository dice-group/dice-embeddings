"""
Exploration: TabPFN on Knowledge Graph Tensor Slices
=====================================================

Knowledge Graph Tensor Representation
--------------------------------------
A knowledge graph :math:`\\mathcal{G} = (\\mathcal{E}, \\mathcal{R}, \\mathcal{T})`
consists of entities :math:`\\mathcal{E}`, relations :math:`\\mathcal{R}`, and triples
:math:`\\mathcal{T} \\subseteq \\mathcal{E} \\times \\mathcal{R} \\times \\mathcal{E}`.

We encode it as a 3-D binary adjacency tensor
:math:`\\mathbf{T} \\in \\{0,1\\}^{|\\mathcal{E}| \\times |\\mathcal{E}| \\times |\\mathcal{R}|}`:

.. math::

    \\mathbf{T}_{i,k,j} = \\begin{cases} 1 & \\text{if } (e_i, r_j, e_k) \\in \\mathcal{T} \\\\ 0 & \\text{otherwise} \\end{cases}

Two-Stage TabPFN Pipeline
--------------------------
**Stage 1 — Relation predictor.**
Given a head entity :math:`e_i`, estimate the posterior over relations:

.. math::

    P(r_j \\mid e_i) \\approx \\text{TabPFN}_{\\text{rel}}\\bigl(\\phi(e_i)\\bigr)

where :math:`\\phi(e_i)` is an entity feature vector produced by the selected
:class:`EntityFeaturiser` (``--features degree`` or ``--features kge``).

**Stage 2 — Tail predictor.**
For each relation :math:`r_j`, estimate the posterior over tail entities:

.. math::

    P(e_k \\mid e_i, r_j) \\approx \\text{TabPFN}_{\\text{tail}}^{(j)}\\bigl(\\phi(e_i)\\bigr)

**Chaining (law of total probability).**
Marginalise out the latent relation to obtain an unconditional tail score:

.. math::

    P(e_k \\mid e_i) = \\sum_{j=1}^{|\\mathcal{R}|} P(r_j \\mid e_i)\\\\; P(e_k \\mid e_i, r_j)

Works best on small KGs: UMLS, Family, KINSHIP, Animals.
TabPFN is designed for :math:`\\leq 10{,}000` samples and :math:`\\leq 100` features.

Usage Examples
--------------
**Train all stages and save the pipeline to a cache file:**

.. code-block:: bash

    python tabpfn_kg_explore.py --dataset_dir KGs/UMLS --save --cache umls.pkl

**Load from cache and predict tail entities (marginalised over relations):**

.. code-block:: bash

    python tabpfn_kg_explore.py --load --cache umls.pkl \\
        --mode tail_predict --query_entity "enzyme" --top_k 10

**Load from cache and predict which relations a head entity participates in:**

.. code-block:: bash

    python tabpfn_kg_explore.py --load --cache umls.pkl \\
        --mode relation_predict --query_entity "enzyme" --top_k 10

**Link prediction — given (head, relation), predict the most likely tail entities:**

.. code-block:: bash

    python tabpfn_kg_explore.py --load --cache umls.pkl \\
        --mode link_predict --query_entity "enzyme" --query_relation "disrupts" --top_k 10

**Evaluate on test.txt — compute MRR, MR, Hits@1/3/10:**

.. code-block:: bash

    python tabpfn_kg_explore.py --load --cache umls.pkl \\
        --mode evaluate --dataset_dir KGs/UMLS

**Train without saving (one-shot run):**

.. code-block:: bash

    python tabpfn_kg_explore.py --dataset_dir KGs/UMLS --query_entity "enzyme"

Entity Feature Strategies
--------------------------
Select the feature strategy with ``--features``.

**Default — relational degree profile (no external model required):**

.. code-block:: bash

    python tabpfn_kg_explore.py --dataset_dir KGs/UMLS --features degree --save --cache umls_degree.pkl

**KGE embeddings — first train a Keci model with dicee, then use its embeddings:**

.. code-block:: bash

    # Step 1: train a Keci model and store it
    dicee --dataset_dir KGs/UMLS --model Keci --embedding_dim 64 \
          --num_epochs 200 --path_to_store_single_run Experiments/Keci_UMLS

    # Step 2: build the TabPFN pipeline using Keci entity embeddings
    python tabpfn_kg_explore.py \\
        --dataset_dir KGs/UMLS \\
        --features kge \\
        --kge_path Experiments/Keci_UMLS \\
        --save --cache umls_kge.pkl

    # Step 3: evaluate
    python tabpfn_kg_explore.py \\
        --load --cache umls_kge.pkl \\
        --mode evaluate --dataset_dir KGs/UMLS

Any dicee model (ComplEx, DistMult, TransE, …) can be used — just point
``--kge_path`` at the experiment folder that contains ``model.pt``.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from tqdm import tqdm

TABPFN_MAX_CLASSES = 10  # hard limit imposed by TabPFN


def has_feature_variance(X: np.ndarray) -> bool:
    """Return ``False`` if every feature column is constant.

    TabPFN internally removes zero-variance columns and raises an error when
    none remain.  A column :math:`j` has variance

    .. math::

        \\sigma_j^2 = \\frac{1}{N}\\sum_{i=1}^{N}\\bigl(X_{ij} - \\bar{X}_j\\bigr)^2

    This guard checks :math:`\\exists\\, j : \\sigma_j > 0`.
    """
    return bool(np.any(X.std(axis=0) > 0))


def filter_top_classes(X: np.ndarray, y: np.ndarray, max_classes: int = TABPFN_MAX_CLASSES):
    """Keep only samples whose label is among the top-``max_classes`` most frequent.

    TabPFN supports at most :math:`C_{\\max}` classes (default 10).  Given label
    counts :math:`n_c = |\\{i : y_i = c\\}|`, we retain the set

    .. math::

        \\mathcal{C}^* = \\operatorname*{arg\\,top-}_{C_{\\max}}\\{ n_c \\mid c \\in \\mathcal{C} \\}

    and discard all samples with :math:`y_i \\notin \\mathcal{C}^*`.
    Returns filtered ``(X, y)`` — both arrays may shrink.
    """
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) <= max_classes:
        return X, y
    top_labels = set(labels[np.argsort(counts)[-max_classes:]])
    mask = np.array([yi in top_labels for yi in y])
    return X[mask], y[mask]


class MultiTabPFNClassifier:
    """Ensemble of TabPFN classifiers that covers an arbitrary number of classes.

    TabPFN is limited to :math:`C_{\\max} = 10` classes per model.  When the
    true number of classes :math:`C > C_{\\max}`, we partition the label set
    into :math:`\\lceil C / C_{\\max} \\rceil` disjoint chunks
    :math:`\\mathcal{C}_1, \\ldots, \\mathcal{C}_B` and train one classifier
    per chunk:

    .. math::

        f_b : \\mathbb{R}^d \\to \\Delta^{|\\mathcal{C}_b|-1},
        \\qquad b = 1, \\ldots, B

    At inference the raw probability vectors are concatenated and
    re-normalised to form a single distribution over all :math:`C` classes:

    .. math::

        P(c \\mid x) = \\frac{\\tilde{p}_c}{\\sum_{c'} \\tilde{p}_{c'}},
        \\qquad \\tilde{p}_c = f_{b(c)}(x)_c

    where :math:`b(c)` denotes the chunk that contains class :math:`c`.
    This preserves all classes — no information is discarded.
    """

    def __init__(self, chunk_size: int = TABPFN_MAX_CLASSES):
        self.chunk_size = chunk_size
        self.classifiers_: list[TabPFNClassifier] = []
        self.chunks_: list[np.ndarray] = []   # list of class-label arrays per chunk
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiTabPFNClassifier":
        self.classes_ = np.unique(y)
        self.classifiers_ = []
        self.chunks_ = []

        for start in range(0, len(self.classes_), self.chunk_size):
            chunk_labels = self.classes_[start: start + self.chunk_size]
            mask = np.isin(y, chunk_labels)
            X_chunk, y_chunk = X[mask], y[mask]

            if len(np.unique(y_chunk)) < 2 or not has_feature_variance(X_chunk):
                # store None so predict_proba can assign uniform scores for this chunk
                self.classifiers_.append(None)
            else:
                clf = TabPFNClassifier()
                clf.fit(X_chunk, y_chunk)
                self.classifiers_.append(clf)

            self.chunks_.append(chunk_labels)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability matrix of shape ``(n_samples, n_classes)``."""
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes_)), dtype=np.float64)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for clf, chunk_labels in zip(self.classifiers_, self.chunks_):
            col_idxs = [class_to_idx[c] for c in chunk_labels]
            if clf is None:
                # uniform fallback for this chunk
                scores[:, col_idxs] = 1.0 / len(chunk_labels)
            else:
                chunk_proba = clf.predict_proba(X)   # (n_samples, |chunk|)
                # map clf.classes_ back to global positions
                for local_i, label in enumerate(clf.classes_):
                    scores[:, class_to_idx[label]] = chunk_proba[:, local_i]

        # Re-normalise rows
        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return scores / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ── Persistence helpers ───────────────────────────────────────────────────────

DEFAULT_CACHE = "tabpfn_kg_cache.pkl"


def save_pipeline(
    path: str,
    e2i: dict,
    r2i: dict,
    X_ent: np.ndarray,
    triples: np.ndarray,
    rel_clf,
    tail_clfs: dict,
    pair_rel_clf,
) -> None:
    """Serialise all pipeline artefacts to a single pickle file.

    Saved keys:

    * ``e2i``, ``r2i``          — entity / relation index maps
    * ``X_ent``                 — entity feature matrix :math:`\\Phi`
    * ``triples``               — indexed triple array
    * ``rel_clf``               — Stage 1 :class:`MultiTabPFNClassifier`
    * ``tail_clfs``             — Stage 2 dict of :class:`MultiTabPFNClassifier`
    * ``pair_rel_clf``          — Stage 3 :class:`MultiTabPFNClassifier`
    """
    payload = {
        "e2i": e2i,
        "r2i": r2i,
        "X_ent": X_ent,
        "triples": triples,
        "rel_clf": rel_clf,
        "tail_clfs": tail_clfs,
        "pair_rel_clf": pair_rel_clf,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[cache] Pipeline saved → {path}")


def load_pipeline(path: str) -> dict:
    """Deserialise the pipeline artefacts saved by :func:`save_pipeline`.

    Returns the same dict of keys described in :func:`save_pipeline`.
    Raises ``FileNotFoundError`` if ``path`` does not exist.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    print(f"[cache] Pipeline loaded ← {path}")
    return payload


# ── 1. Load a small KG ────────────────────────────────────────────────────────

def load_kg(dataset_dir: str) -> tuple[np.ndarray, dict, dict]:
    """Load train triples and build entity/relation index maps."""
    path = Path(dataset_dir) / "train.txt"
    df = pd.read_csv(path, sep="\t", header=None, names=["h", "r", "t"])

    entities = sorted(set(df["h"]) | set(df["t"]))
    relations = sorted(set(df["r"]))

    e2i = {e: i for i, e in enumerate(entities)}
    r2i = {r: i for i, r in enumerate(relations)}

    triples = df.apply(lambda row: (e2i[row.h], r2i[row.r], e2i[row.t]), axis=1).tolist()
    return np.array(triples, dtype=np.int32), e2i, r2i


# ── 2. Build the 3D binary adjacency tensor ───────────────────────────────────

def build_tensor(triples: np.ndarray, n_entities: int, n_relations: int) -> np.ndarray:
    """Build the 3-D binary adjacency tensor :math:`\\mathbf{T}`.

    .. math::

        \\mathbf{T} \\in \\{0,1\\}^{|\\mathcal{E}| \\times |\\mathcal{E}| \\times |\\mathcal{R}|},
        \\qquad \\mathbf{T}_{i,k,j} = \\mathbf{1}[(e_i, r_j, e_k) \\in \\mathcal{T}]

    Memory footprint: :math:`|\\mathcal{E}|^2 \\cdot |\\mathcal{R}|` bytes (``uint8``).
    """
    T = np.zeros((n_entities, n_entities, n_relations), dtype=np.uint8)
    for h, r, t in triples:
        T[h, t, r] = 1
    return T


# ── 3. Entity feature vectors from tensor slices ─────────────────────────────


class EntityFeaturiser(ABC):
    """Abstract base class for entity feature extraction.

    Any implementation must produce a matrix
    :math:`\\Phi \\in \\mathbb{R}^{|\\mathcal{E}| \\times d}` where row
    :math:`i` is the feature vector :math:`\\phi(e_i)` for entity :math:`e_i`.
    All downstream stages (training and inference) consume only this matrix,
    making the feature strategy fully interchangeable.

    To add a new strategy:

    1. Subclass :class:`EntityFeaturiser`.
    2. Implement :meth:`fit_transform`.
    3. Register the name in :func:`build_featuriser`.
    """

    @abstractmethod
    def fit_transform(
        self,
        T: np.ndarray,
        e2i: dict,
        i2e: dict,
    ) -> np.ndarray:
        """Compute and return the entity feature matrix.

        Parameters
        ----------
        T:
            3-D binary adjacency tensor :math:`\\mathbf{T}`.
        e2i:
            Entity name → integer index mapping.
        i2e:
            Integer index → entity name mapping.

        Returns
        -------
        np.ndarray
            Shape ``(|E|, d)``, dtype ``float32``.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier shown in CLI output."""


class DegreeFeaturiser(EntityFeaturiser):
    """Represent each entity by its relational degree profile.

    Out- and in-degree counts per relation are concatenated:

    .. math::

        d^{\\text{out}}_{i,j} = \\sum_{k} \\mathbf{T}_{i,k,j}, \\qquad
        d^{\\text{in}}_{i,j}  = \\sum_{k} \\mathbf{T}_{k,i,j}

    .. math::

        \\phi(e_i) = \\bigl[d^{\\text{out}}_{i,\\cdot}\\\\;\\|\\\\;
        d^{\\text{in}}_{i,\\cdot}\\bigr] \\in \\mathbb{R}^{2|\\mathcal{R}|}

    No external model required — works on any KG straight from the tensor.
    """

    @property
    def name(self) -> str:
        return "degree"

    def fit_transform(self, T: np.ndarray, e2i: dict, i2e: dict) -> np.ndarray:
        out_deg = T.sum(axis=1)   # (|E|, |R|)
        in_deg  = T.sum(axis=0)   # (|E|, |R|)
        return np.hstack([out_deg, in_deg]).astype(np.float32)


class KGEFeaturiser(EntityFeaturiser):
    """Represent each entity by its embedding from a pre-trained dicee KGE model.

    Loads the model via :class:`dicee.KGE` and calls
    :meth:`get_transductive_entity_embeddings` to extract the embedding
    matrix.  The feature dimensionality equals the model's ``embedding_dim``.

    .. math::

        \\phi(e_i) = \\mathbf{E}[i] \\in \\mathbb{R}^{d_{\\text{emb}}}

    where :math:`\\mathbf{E}` is the entity embedding table of the trained model.

    Parameters
    ----------
    kge_path:
        Path to an experiment folder produced by dicee (must contain
        ``model.pt`` and ``configuration.json``).
    """

    def __init__(self, kge_path: str):
        self.kge_path = kge_path
        self._model = None

    @property
    def name(self) -> str:
        return f"kge:{self.kge_path}"

    def fit_transform(self, T: np.ndarray, e2i: dict, i2e: dict) -> np.ndarray:
        from dicee import KGE  # lazy import — not required for degree mode
        self._model = KGE(path=self.kge_path)

        # Entities ordered by their integer index (same order as T rows)
        ordered_entities = [i2e[i] for i in range(len(i2e))]

        # Filter to entities known to the KGE model
        known = [e for e in ordered_entities if e in self._model.entity_to_idx]
        unknown = [e for e in ordered_entities if e not in self._model.entity_to_idx]
        if unknown:
            print(rf"[KGEFeaturiser] Warning: {len(unknown)} entities not in KGE model\; "
                  "using zero vectors for them.")

        d = self._model.model.entity_embeddings.embedding_dim
        X = np.zeros((len(ordered_entities), d), dtype=np.float32)

        if known:
            embs = self._model.get_transductive_entity_embeddings(
                known, as_pytorch=True
            ).detach().numpy().astype(np.float32)
            for i, name in enumerate(known):
                X[e2i[name]] = embs[i]

        return X


def build_featuriser(name: str, kge_path: str | None) -> EntityFeaturiser:
    """Factory that maps a strategy name to an :class:`EntityFeaturiser`.

    Available strategies
    --------------------
    ``degree``
        :class:`DegreeFeaturiser` — no external model needed.
    ``kge``
        :class:`KGEFeaturiser` — requires ``--kge_path``.

    Parameters
    ----------
    name:
        Strategy identifier, one of ``{"degree", "kge"}``.
    kge_path:
        Path to a dicee experiment folder.  Required when ``name == "kge"``.
    """
    if name == "degree":
        return DegreeFeaturiser()
    if name == "kge":
        if not kge_path:
            raise ValueError("--kge_path is required when --features kge")
        return KGEFeaturiser(kge_path)
    raise ValueError(f"Unknown --features value '{name}'. Choose from: degree, kge")


# ── 4. Stage 1 — Relation predictor ──────────────────────────────────────────

def build_relation_prediction_dataset(triples: np.ndarray, X_entities: np.ndarray):
    """Construct the dataset for Stage 1 (relation prediction).

    For each observed triple :math:`(e_i, r_j, e_k) \\in \\mathcal{T}` we form
    one training example:

    .. math::

        \\bigl(\\phi(e_i),\\; r_j\\bigr)

    so the classifier learns :math:`P(r_j \\mid \\phi(e_i))`.
    """
    X = X_entities[triples[:, 0]]   # head entity features
    y = triples[:, 1]                # relation label
    return X, y


def train_relation_predictor(triples, X_entities):
    X, y = build_relation_prediction_dataset(triples, X_entities)
    n_classes = len(np.unique(y))

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    max_train = 3000
    if len(X_tr) > max_train:
        idx = np.random.choice(len(X_tr), max_train, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    if not has_feature_variance(X_tr):
        print("[Stage 1] Skipped: all features are constant in training split")
        return None

    clf = MultiTabPFNClassifier()
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    print(f"[Stage 1] Relation prediction accuracy: {acc:.4f}  "
          f"(train={len(X_tr)}, test={len(X_te)}, n_relations={n_classes}, "
          f"n_chunks={len(clf.chunks_)})")
    return clf


# ── 5. Stage 2 — Tail predictor per relation ─────────────────────────────────

def build_tail_prediction_dataset(triples: np.ndarray, X_entities: np.ndarray, relation_id: int):
    """Construct the dataset for Stage 2 (tail prediction) for a single relation.

    For a fixed relation :math:`r_j`, restrict :math:`\\mathcal{T}` to the slice

    .. math::

        \\mathcal{T}_j = \\{(e_i, e_k) \\mid (e_i, r_j, e_k) \\in \\mathcal{T}\\}

    and build examples :math:`(\\phi(e_i),\\; e_k)` so the classifier learns
    :math:`P(e_k \\mid e_i, r_j)`.
    """
    mask = triples[:, 1] == relation_id
    sub  = triples[mask]
    if len(sub) == 0:
        return None, None
    X = X_entities[sub[:, 0]]
    y = sub[:, 2]
    return X, y


def train_tail_predictors(triples, X_entities, n_relations, max_per_rel=2000):
    """Train one :class:`TabPFNClassifier` per relation.

    Produces a family of classifiers
    :math:`\\{f_j\\}_{j=1}^{|\\mathcal{R}|}` where each

    .. math::

        f_j : \\mathbb{R}^{2|\\mathcal{R}|} \\to \\Delta^{C_j - 1}

    maps a head-entity feature vector to a probability simplex over the
    :math:`C_j \\leq C_{\\max}` most frequent tail entities for relation
    :math:`r_j`.

    Returns a dict ``{rel_id: clf}``.
    """
    clfs = {}
    skipped: list[tuple[int, str]] = []   # (relation_id, reason)
    for r in range(n_relations):
        X, y = build_tail_prediction_dataset(triples, X_entities, r)
        if X is None or len(np.unique(y)) < 2:
            skipped.append((r, "fewer than 2 unique tail classes"))
            continue

        # Restrict to top-TABPFN_MAX_CLASSES most frequent tail entities
        X, y = filter_top_classes(X, y)
        if len(np.unique(y)) < 2:
            skipped.append((r, "fewer than 2 classes after top-class filter"))
            continue

        if len(X) > max_per_rel:
            idx  = np.random.choice(len(X), max_per_rel, replace=False)
            X, y = X[idx], y[idx]

        if not has_feature_variance(X):
            skipped.append((r, "all head-entity features are constant"))
            continue

        clf = MultiTabPFNClassifier()
        clf.fit(X, y)
        clfs[r] = clf

    print(f"[Stage 2] Trained tail predictors for {len(clfs)}/{n_relations} relations")
    if skipped:
        print(f"[Stage 2] Skipped {len(skipped)} relation(s):")
        for r_id, reason in skipped:
            print(f"           relation id={r_id:>3}  reason: {reason}")
    return clfs


# ── 6. Chained inference: P(e_k | e_i) ───────────────────────────────────────


# ── Stage 3 — Relation predictor given (head, tail) pair ─────────────────────

def build_pair_relation_dataset(triples: np.ndarray, X_entities: np.ndarray):
    """Construct the dataset for Stage 3 (relation prediction given a pair).

    For each triple :math:`(e_i, r_j, e_k) \\in \\mathcal{T}` we concatenate
    the head and tail feature vectors:

    .. math::

        \\psi(e_i, e_k) = \\bigl[\\phi(e_i) \\\\;\\|\\\\; \\phi(e_k)\\bigr]
        \\in \\mathbb{R}^{4|\\mathcal{R}|}

    and use the relation :math:`r_j` as the target, so the classifier learns
    :math:`P(r_j \\mid e_i, e_k)`.
    """
    X = np.hstack([X_entities[triples[:, 0]], X_entities[triples[:, 2]]])
    y = triples[:, 1]
    return X, y


def train_pair_relation_predictor(triples: np.ndarray, X_entities: np.ndarray):
    """Train a :class:`MultiTabPFNClassifier` that predicts
    :math:`P(r_j \\mid e_i, e_k)` from the concatenated pair features
    :math:`\\psi(e_i, e_k)`.
    """
    X, y = build_pair_relation_dataset(triples, X_entities)
    n_classes = len(np.unique(y))

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    max_train = 3000
    if len(X_tr) > max_train:
        idx = np.random.choice(len(X_tr), max_train, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    if not has_feature_variance(X_tr):
        print("[Stage 3] Skipped: all features are constant")
        return None

    clf = MultiTabPFNClassifier()
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    print(f"[Stage 3] Pair→relation accuracy: {acc:.4f}  "
          f"(train={len(X_tr)}, test={len(X_te)}, n_relations={n_classes}, "
          f"n_chunks={len(clf.chunks_)})")
    return clf


def predict_relations_for_pair(
    head_entity_id: int,
    tail_entity_id: int,
    X_entities: np.ndarray,
    pair_rel_clf: MultiTabPFNClassifier,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Predict the most likely relations between a head–tail pair.

    .. math::

        P(r_j \\mid e_i, e_k) \\approx
        \\text{TabPFN}_{\\text{pair}}\\bigl(\\psi(e_i, e_k)\\bigr)

    Returns the top-``top_k`` ``(relation_id, probability)`` pairs.
    """
    x = np.hstack([
        X_entities[head_entity_id],
        X_entities[tail_entity_id],
    ]).reshape(1, -1)
    proba = pair_rel_clf.predict_proba(x)[0]
    ranked = sorted(zip(pair_rel_clf.classes_, proba), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


def score_triple(
    head_entity_id: int,
    relation_id: int,
    tail_entity_id: int,
    X_entities: np.ndarray,
    rel_clf: MultiTabPFNClassifier,
    tail_clfs: dict,
) -> float:
    """Compute the joint triple score using the two-stage factorisation.

    .. math::

        \\text{score}(e_i, r_j, e_k)
        = P(r_j \\mid e_i) \\cdot P(e_k \\mid e_i, r_j)

    Returns 0.0 if :math:`r_j` was not covered by Stage 1, or if no tail
    predictor exists for :math:`r_j`.
    """
    x = X_entities[head_entity_id].reshape(1, -1)

    p_rel = rel_clf.predict_proba(x)[0]
    rel_probs = dict(zip(rel_clf.classes_, p_rel))
    p_r = rel_probs.get(relation_id, 0.0)
    if p_r < 1e-9 or relation_id not in tail_clfs:
        return 0.0

    tail_clf = tail_clfs[relation_id]
    p_tail = tail_clf.predict_proba(x)[0]
    tail_probs = dict(zip(tail_clf.classes_, p_tail))
    p_t = tail_probs.get(tail_entity_id, 0.0)

    return float(p_r * p_t)


def predict_relations_given_head(
    head_entity_id: int,
    X_entities: np.ndarray,
    rel_clf: MultiTabPFNClassifier,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Return a ranked list of ``(relation_id, probability)`` pairs for a head entity.

    Directly exposes the Stage 1 posterior:

    .. math::

        P(r_j \\mid e_i) \\approx \\text{TabPFN}_{\\text{rel}}\\bigl(\\phi(e_i)\\bigr)

    Parameters
    ----------
    top_k:
        Number of top relations to return.  ``None`` returns all.
    """
    x = X_entities[head_entity_id].reshape(1, -1)
    proba = rel_clf.predict_proba(x)[0]
    ranked = sorted(zip(rel_clf.classes_, proba), key=lambda kv: kv[1], reverse=True)
    return ranked if top_k is None else ranked[:top_k]


def link_predict(
    head_entity_id: int,
    relation_id: int,
    X_entities: np.ndarray,
    rel_clf: MultiTabPFNClassifier,
    tail_clfs: dict,
    n_entities: int,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Predict a probability distribution over all tail entities given ``(e_i, r_j)``.

    The score for each candidate tail :math:`e_k` is the joint triple probability:

    .. math::

        \\text{score}(e_k) = P(r_j \\mid e_i) \\cdot P(e_k \\mid e_i, r_j)

    * :math:`P(r_j \\mid e_i)` is a scalar from Stage 1 (same for all :math:`e_k`).
    * :math:`P(e_k \\mid e_i, r_j)` comes from the Stage 2 classifier for :math:`r_j`.
      Entities not in the classifier's support receive score 0.

    Returns the top-``top_k`` ``(entity_id, score)`` pairs sorted by descending score.
    """
    x = X_entities[head_entity_id].reshape(1, -1)

    # P(r_j | e_i)
    p_rel = rel_clf.predict_proba(x)[0]
    rel_probs = dict(zip(rel_clf.classes_, p_rel))
    p_r = rel_probs.get(relation_id, 0.0)

    scores = np.zeros(n_entities, dtype=np.float64)
    if p_r > 1e-9 and relation_id in tail_clfs:
        clf = tail_clfs[relation_id]
        p_tail = clf.predict_proba(x)[0]          # distribution over seen tail classes
        for e_k, p_t in zip(clf.classes_, p_tail):
            scores[e_k] = p_r * p_t

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_idx]


def evaluate_link_prediction(
    test_triples: np.ndarray,
    X_entities: np.ndarray,
    rel_clf: MultiTabPFNClassifier,
    tail_clfs: dict,
    n_entities: int,
    n_test: int | None = None,
) -> dict:
    """Compute standard link prediction metrics on a held-out triple set.

    For each test triple :math:`(e_i, r_j, e_k)`, rank :math:`e_k` among all
    :math:`|\\mathcal{E}|` candidate tails using :func:`link_predict` scores:

    .. math::

        \\text{rank}(e_k) = 1 + \\bigl|\\{ e' : \\text{score}(e_i, r_j, e') > \\text{score}(e_i, r_j, e_k) \\}\\bigr|

    Reported metrics:

    .. math::

        \\text{MRR} = \\frac{1}{|\\mathcal{T}_{\\text{test}}|} \\sum_{(e_i,r_j,e_k)} \\frac{1}{\\text{rank}(e_k)}

    .. math::

        \\text{Hits@}k = \\frac{1}{|\\mathcal{T}_{\\text{test}}|}
        \\sum_{(e_i,r_j,e_k)} \\mathbf{1}[\\text{rank}(e_k) \\leq k]

    Parameters
    ----------
    n_test:
        If given, randomly subsample this many triples from ``test_triples``
        before evaluation.  Useful for a quick estimate on large test sets.
    """
    if n_test is not None and n_test < len(test_triples):
        idx = np.random.choice(len(test_triples), n_test, replace=False)
        test_triples = test_triples[idx]

    ranks = []
    for h, r, t in tqdm(test_triples, desc="Evaluating", unit="triple"):
        x = X_entities[h].reshape(1, -1)
        p_rel = rel_clf.predict_proba(x)[0]
        rel_probs = dict(zip(rel_clf.classes_, p_rel))
        p_r = rel_probs.get(int(r), 0.0)

        scores = np.zeros(n_entities, dtype=np.float64)
        if p_r > 1e-9 and int(r) in tail_clfs:
            clf = tail_clfs[int(r)]
            p_tail = clf.predict_proba(x)[0]
            for e_k, p_t in zip(clf.classes_, p_tail):
                scores[e_k] = p_r * p_t

        # rank of the true tail (1-based, lower is better)
        rank = int((scores > scores[t]).sum()) + 1
        ranks.append(rank)

    ranks = np.array(ranks, dtype=np.float64)
    return {
        "MRR":     float(np.mean(1.0 / ranks)),
        "Hits@1":  float(np.mean(ranks <= 1)),
        "Hits@3":  float(np.mean(ranks <= 3)),
        "Hits@10": float(np.mean(ranks <= 10)),
        "MR":      float(np.mean(ranks)),
        "n_triples": len(ranks),
    }

def predict_tails(
    head_entity_id: int,
    X_entities: np.ndarray,
    rel_clf: MultiTabPFNClassifier,
    tail_clfs: dict,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Compute marginal tail scores via the law of total probability.

    .. math::

        P(e_k \\mid e_i)
        = \\sum_{j=1}^{|\\mathcal{R}|} P(r_j \\mid e_i)\\\\; P(e_k \\mid e_i, r_j)

    where

    * :math:`P(r_j \\mid e_i)` comes from the Stage 1 :class:`MultiTabPFNClassifier`,
    * :math:`P(e_k \\mid e_i, r_j)` comes from the Stage 2 :class:`MultiTabPFNClassifier`
      ``tail_clfs[j]``.

    Terms with :math:`P(r_j \\mid e_i) < 10^{-6}` are skipped for efficiency.

    Returns the top-``top_k`` ``(tail_entity_id, score)`` pairs sorted by
    descending :math:`P(e_k \\mid e_i)`.
    """
    x = X_entities[head_entity_id].reshape(1, -1)

    # Stage 1: P(r_j | e_i) — use classes_ from MultiTabPFNClassifier
    p_rel = rel_clf.predict_proba(x)[0]                   # (n_relations,)
    rel_probs = dict(zip(rel_clf.classes_, p_rel))

    # Accumulate P(e_k | e_i) across relations
    tail_scores: dict[int, float] = {}
    for r_id, clf in tail_clfs.items():
        p_r = rel_probs.get(r_id, 0.0)
        if p_r < 1e-6:
            continue
        p_tail = clf.predict_proba(x)[0]                  # (n_tails,)
        for e_k, p_t in zip(clf.classes_, p_tail):
            tail_scores[e_k] = tail_scores.get(e_k, 0.0) + p_r * p_t

    ranked = sorted(tail_scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


# ── 7. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="KGs/Family", help="Path to KG folder")
    parser.add_argument("--query_entity", default=None, help="Head entity name")
    parser.add_argument("--query_relation", default=None, help="Relation name (for link_predict mode)")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=None, help="Randomly subsample this many test triples for evaluation (None = use all)")
    parser.add_argument(
        "--features",
        default="degree",
        choices=["degree", "kge"],
        help=(
            "degree  : relational degree profile phi(e_i) in R^{2|R|} (default, no external model)\n"
            "kge     : entity embeddings from a pre-trained dicee model (requires --kge_path)"
        ),
    )
    parser.add_argument(
        "--kge_path",
        default=None,
        help="Path to a dicee experiment folder (required when --features kge)",
    )
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="Path to cache file")
    parser.add_argument("--save", action="store_true", help="Save trained pipeline to --cache after training")
    parser.add_argument("--load", action="store_true", help="Load pipeline from --cache and skip training")
    parser.add_argument(
        "--mode",
        default="tail_predict",
        choices=["tail_predict", "relation_predict", "link_predict", "evaluate"],
        help=(
            "tail_predict   : P(e_k | e_i) marginalised over relations  [needs --query_entity]\n"
            "relation_predict: P(r_j | e_i) from Stage 1               [needs --query_entity]\n"
            "link_predict   : P(e_k | e_i, r_j) from Stage 2           [needs --query_entity + --query_relation]\n"
            "evaluate       : MRR / Hits@k on test.txt"
        ),
    )
    args = parser.parse_args()

    if args.load:
        # ── Load all artefacts from disk\; skip every training stage ───────────
        p = load_pipeline(args.cache)
        e2i, r2i       = p["e2i"], p["r2i"]
        X_ent, triples = p["X_ent"], p["triples"]
        rel_clf        = p["rel_clf"]
        tail_clfs      = p["tail_clfs"]
        pair_rel_clf   = p["pair_rel_clf"]
        i2e = {v: k for k, v in e2i.items()}
        i2r = {v: k for k, v in r2i.items()}
        n_e, n_r = len(e2i), len(r2i)
        print(f"  Entities: {n_e},  Relations: {n_r},  Triples: {len(triples)}")
    else:
        # ── Full training pipeline ────────────────────────────────────────────
        print(f"\n=== Loading KG: {args.dataset_dir} ===")
        triples, e2i, r2i = load_kg(args.dataset_dir)
        i2e = {v: k for k, v in e2i.items()}
        i2r = {v: k for k, v in r2i.items()}
        n_e, n_r = len(e2i), len(r2i)
        print(f"  Entities: {n_e},  Relations: {n_r},  Triples: {len(triples)}")

        print("\n=== Building 3D adjacency tensor ===")
        T = build_tensor(triples, n_e, n_r)
        print(f"  Tensor shape: {T.shape}  (density: {T.mean():.5f})")

        featuriser = build_featuriser(args.features, args.kge_path)
        print(f"\n=== Computing entity features  [{featuriser.name}] ===")
        X_ent = featuriser.fit_transform(T, e2i, i2e)
        print(f"  Entity feature matrix: {X_ent.shape}")

        print("\n=== Stage 1: Train relation predictor (head → relation) ===")
        rel_clf = train_relation_predictor(triples, X_ent)

        print("\n=== Stage 2: Train per-relation tail predictors ===")
        tail_clfs = train_tail_predictors(triples, X_ent, n_r)

        print("\n=== Stage 3: Train pair→relation predictor (head, tail) → relation ===")
        pair_rel_clf = train_pair_relation_predictor(triples, X_ent)

        if args.save:
            save_pipeline(
                args.cache, e2i, r2i, X_ent, triples,
                rel_clf, tail_clfs, pair_rel_clf,
            )

    if rel_clf is None:
        print(r"Stage 1 classifier unavailable\; cannot run inference.")
        raise SystemExit(1)

    # ──────────────────────────────────────────────────────────────────────
    # MODE: evaluate
    # ──────────────────────────────────────────────────────────────────────
    if args.mode == "evaluate":
        test_path = Path(args.dataset_dir) / "test.txt"
        if not test_path.exists():
            print(f"  test.txt not found at {test_path}")
            raise SystemExit(1)
        test_df = pd.read_csv(test_path, sep="\t", header=None, names=["h", "r", "t"])
        # keep only triples whose entities and relations were seen during training
        test_df = test_df[
            test_df["h"].isin(e2i) & test_df["r"].isin(r2i) & test_df["t"].isin(e2i)
        ]
        test_triples = np.array(
            [(e2i[row.h], r2i[row.r], e2i[row.t]) for _, row in test_df.iterrows()],
            dtype=np.int32,
        )
        print(f"\n=== Evaluation on test.txt ({len(test_triples)} triples{f', subsampling {args.n_test}' if args.n_test else ''}) ===")
        metrics = evaluate_link_prediction(test_triples, X_ent, rel_clf, tail_clfs, n_e, n_test=args.n_test)
        print(f"  MRR      : {metrics['MRR']:.4f}")
        print(f"  MR       : {metrics['MR']:.1f}")
        print(f"  Hits@1   : {metrics['Hits@1']:.4f}")
        print(f"  Hits@3   : {metrics['Hits@3']:.4f}")
        print(f"  Hits@10  : {metrics['Hits@10']:.4f}")
        raise SystemExit(0)

    # ──────────────────────────────────────────────────────────────────────
    # Query modes that need --query_entity
    # ──────────────────────────────────────────────────────────────────────
    if args.query_entity is None:
        print("  No --query_entity provided. Training complete. Use --query_entity to run inference.")
        raise SystemExit(0)

    if args.query_entity not in e2i:
        print(f"  Entity '{args.query_entity}' not found. Known entities (sample): {list(e2i)[:5]}")
        raise SystemExit(1)
    query_id = e2i[args.query_entity]
    print(f"\n  Query entity: '{args.query_entity}'  (id={query_id})")

    # ──────────────────────────────────────────────────────────────────────
    if args.mode == "relation_predict":
        # P(r_j | e_i)
        print(f"\n  P(relation | '{args.query_entity}'):")
        rel_results = predict_relations_given_head(query_id, X_ent, rel_clf, top_k=args.top_k)
        print(f"  {'Rank':<5} {'Relation':<45} {'P(rel|head)':>12}")
        print(f"  {'-'*4} {'-'*44} {'-'*12}")
        for rank, (r_id, prob) in enumerate(rel_results, 1):
            print(f"  {rank:<5} {i2r.get(r_id, str(r_id)):<45} {prob:>12.4f}")

    elif args.mode == "link_predict":
        # P(e_k | e_i, r_j)
        if args.query_relation not in r2i:
            print(f"  Relation '{args.query_relation}' not found. Known relations (sample): {list(r2i)[:5]}")
            raise SystemExit(1)
        rel_id = r2i[args.query_relation]
        print(f"  Query relation: '{args.query_relation}'  (id={rel_id})")
        p_r = dict(zip(rel_clf.classes_, rel_clf.predict_proba(X_ent[query_id].reshape(1, -1))[0]))
        print(f"  P(relation | head) = {p_r.get(rel_id, 0.0):.4f}")
        lp_results = link_predict(query_id, rel_id, X_ent, rel_clf, tail_clfs, n_e, top_k=args.top_k)
        print(f"\n  Top-{args.top_k} tails for ('{args.query_entity}', '{args.query_relation}', ?):")
        print(f"  {'Rank':<5} {'Tail entity':<45} {'score P(r|h)*P(t|h,r)':>22}")
        print(f"  {'-'*4} {'-'*44} {'-'*22}")
        for rank, (e_k, score) in enumerate(lp_results, 1):
            print(f"  {rank:<5} {i2e.get(e_k, str(e_k)):<45} {score:>22.6f}")

    else:  # tail_predict (default)
        # P(e_k | e_i) marginalised over relations, with triple scores
        results = predict_tails(
            head_entity_id=query_id,
            X_entities=X_ent,
            rel_clf=rel_clf,
            tail_clfs=tail_clfs,
            top_k=args.top_k,
        )
        print(f"\n  {'Rank':<5} {'Tail entity':<42} {'P(tail|head)':>13}  {'Top relation':<40} {'P(rel|pair)':>12}  {'Triple score':>13}")
        print(f"  {'-'*4} {'-'*41} {'-'*13}  {'-'*39} {'-'*12}  {'-'*13}")
        for rank, (e_k, tail_score) in enumerate(results, 1):
            if pair_rel_clf is not None:
                top_rel_id, top_rel_prob = predict_relations_for_pair(
                    query_id, e_k, X_ent, pair_rel_clf, top_k=1
                )[0]
                top_rel_name = i2r.get(top_rel_id, str(top_rel_id))
                triple_score = score_triple(query_id, top_rel_id, e_k, X_ent, rel_clf, tail_clfs)
            else:
                top_rel_name, top_rel_prob, triple_score = "N/A", 0.0, 0.0
            print(f"  {rank:<5} {i2e[e_k]:<42} {tail_score:>13.4f}  {top_rel_name:<40} {top_rel_prob:>12.4f}  {triple_score:>13.6f}")
