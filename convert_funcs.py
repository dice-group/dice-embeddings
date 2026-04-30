from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm 
from dicee.static_funcs import load_pickle
from dicee.evaluation.link_prediction import evaluate_link_prediction_performance
from collections import defaultdict


class _TabPFNForwardWrapper:
    """
    Expose a TabPFN classifier through the scoring interface expected by DiCEE.

    The link prediction evaluator expects a torch-like object with ``eval()``
    and ``forward_triples()`` methods. This wrapper converts indexed triples
    into concatenated embedding features and returns positive-class
    probabilities from a fitted ``TabPFNClassifier``.
    """
    def __init__(self, clf, entity_embeddings, relation_embeddings):
        self._clf = clf
        self._entity_embeddings = entity_embeddings
        self._relation_embeddings = relation_embeddings

    def eval(self):
        """Return ``self`` for API compatibility with torch modules."""
        # TabPFN classifier itself has no torch eval-mode switch.
        return self

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score indexed triples with the wrapped TabPFN classifier.

        Args:
            x: Tensor of shape ``(n, 3)`` containing ``(head, relation, tail)``
                indices.

        Returns:
            A 1D torch tensor with the predicted positive-class probability for
            each triple candidate.
        """
        # x: [num_candidates, 3] -> (h_idx, r_idx, t_idx)
        triple_idx = x.detach().cpu().numpy().astype(np.int64)
        h_emb = self._entity_embeddings[triple_idx[:, 0]]
        r_emb = self._relation_embeddings[triple_idx[:, 1]]
        t_emb = self._entity_embeddings[triple_idx[:, 2]]
        features = np.concatenate([h_emb, r_emb, t_emb], axis=1).astype(np.float32)
        probs = self._clf.predict_proba(features)[:, 1]
        return torch.tensor(probs, dtype=torch.float32)


class TabPFNLinkPredictionAdapter:
    """
    Adapt a fitted TabPFN model to the link-prediction evaluation interface.

    This adapter stores entity and relation index mappings and exposes a
    ``model`` attribute that behaves like the scorer expected by
    ``evaluate_link_prediction_performance``.
    """
    def __init__(self, clf, entity_to_idx, relation_to_idx, entity_embeddings, relation_embeddings):
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.num_entities = len(entity_to_idx)
        self.model = _TabPFNForwardWrapper(clf, entity_embeddings, relation_embeddings)

    def get_entity_index(self, x: str) -> int:
        """Return the integer index for an entity identifier."""
        return self.entity_to_idx[x]

    def get_relation_index(self, x: str) -> int:
        """Return the integer index for a relation identifier."""
        return self.relation_to_idx[x]

def kg_to_array(dataset_dir: str, split: str) -> list:
    """
    Load triples from a knowledge graph split file.

    Args:
        dataset_dir: Path to the dataset folder, for example ``"KGs/UMLS"``.
        split: Name of the split file without extension, usually
            ``"train"``, ``"valid"``, or ``"test"``.

    Returns:
        A list of ``(head, relation, tail)`` tuples.

    Example:
        >>> triples = kg_to_array("KGs/UMLS", "train")
        >>> triples[0]
        ('entity_a', 'relation_x', 'entity_b')
    """
    file_path = os.path.join(dataset_dir, f"{split}.txt")
    
    triples = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")  # or split() if space-separated
            if len(parts) == 3:
                head, relation, tail = parts
                triples.append((head, relation, tail))
    
    return triples


#Binary representation of entities based on their relations.
def build_entity_relation_matrix(triples):
    """
    Build a binary entity-relation feature matrix from triples.

    The returned matrix uses head entities as rows and relations as columns.
    A cell is set to ``1`` when the entity appears with that relation as a
    head entity in at least one triple; otherwise it is ``0``.

    Args:
        triples: Iterable of ``(head, relation, tail)`` triples.

    Returns:
        A pandas ``DataFrame`` indexed by entity names with one binary column
        per relation.

    Example:
        >>> triples = [("alice", "likes", "pizza"), ("alice", "knows", "bob")]
        >>> matrix = build_entity_relation_matrix(triples)
        >>> int(matrix.loc["alice", "likes"])
        1
    """
    entities = sorted(set(h for h, _, _ in triples))
    relations = sorted(set(r for _, r, _ in triples))

    matrix = pd.DataFrame(0, index=entities, columns=relations)

    for head, relation, tail in triples:
        matrix.at[head, relation] = 1

    return matrix


def generate_samples(triples, entity_matrix, neg_ratio=1, seed=42):
    """
    Generate labeled tabular samples from triples using binary relation features.

    Each positive triple contributes one positive row built from the binary
    feature vector of the head entity concatenated with the feature vector of
    the tail entity. Negative rows are created by corrupting the tail entity.

    Args:
        triples: Iterable of ``(head, relation, tail)`` triples.
        entity_matrix: Output of :func:`build_entity_relation_matrix`.
        neg_ratio: Number of negative samples to create per positive triple.
        seed: Random seed used for negative sampling.

    Returns:
        A pandas ``DataFrame`` whose columns are head features, tail features,
        and a ``label`` column.

    Example:
        >>> triples = [("alice", "likes", "pizza"), ("bob", "likes", "tea")]
        >>> matrix = build_entity_relation_matrix(triples)
        >>> samples = generate_samples(triples, matrix, neg_ratio=1, seed=0)
        >>> "label" in samples.columns
        True
    """
    rng = np.random.default_rng(seed)
    entities = list(entity_matrix.index)
    triple_set = set(triples)  # use passed triples, not matrix
    rows = []

    for h, r, t in triples:  # iterate over triples, not matrix
        if h not in entity_matrix.index or t not in entity_matrix.index:
            continue  # skip unseen entities (important for test/valid sets)
        
        h_feats = entity_matrix.loc[h].values
        t_feats = entity_matrix.loc[t].values
        rows.append((*h_feats, *t_feats, 1))

        for _ in range(neg_ratio):
            corrupt_t = t
            attempts = 0
            while (h, r, corrupt_t) in triple_set and attempts < 10:
                corrupt_t = rng.choice(entities)
                attempts += 1
            ct_feats = entity_matrix.loc[corrupt_t].values
            rows.append((*h_feats, *ct_feats, 0))

    relations = list(entity_matrix.columns)
    col_names = (
        [f"head_{r}" for r in relations]
        + [f"tail_{r}" for r in relations]
        + ["label"]
    )
    return pd.DataFrame(rows, columns=col_names)
#@TODO Add unitTests

def evaluate(clf, X, y, split_name):
    """
    Evaluate a classifier on a dataset split and print standard metrics.

    The function predicts probabilities in batches of 100 rows, converts them
    to binary labels using a threshold of ``0.5``, and prints accuracy,
    precision, recall, F1, AUC, and the number of samples.

    Args:
        clf: A fitted classifier implementing ``predict_proba``.
        X: Feature matrix.
        y: Ground-truth binary labels.
        split_name: Human-readable name shown in the printed output.

    Example:
        >>> # After training a classifier:
        >>> # evaluate(clf, X_valid, y_valid, "Valid set")
    """
    probs = []
    for i in range(0, len(X), 100):  
        prob = clf.predict_proba(X[i:i + 100])[:, 1]
        probs.append(prob)
    probs = np.concatenate(probs)
    preds = (probs >= 0.5).astype(int)

    print(f"{split_name}: "
          f"Acc={accuracy_score(y, preds):.4f} "
          f"Prec={precision_score(y, preds):.4f} "
          f"Rec={recall_score(y, preds):.4f} "
          f"F1={f1_score(y, preds):.4f} "
          f"AUC={roc_auc_score(y, probs):.4f} "
          f"samples={len(y)}")

def load_embeddings(model_path: str):
    """
    Load entity and relation embeddings from a trained KGE model.

    Args:
        model_path: Path to the experiment directory containing ``model.pt``,
            ``entity_to_idx.csv``, and ``relation_to_idx.csv``.

    Returns:
        A tuple ``(entity_embeddings, relation_embeddings, entity_to_idx,
        relation_to_idx)``.

    Example:
        >>> entity_emb, relation_emb, entity_to_idx, relation_to_idx = \
        ...     load_embeddings("Experiments/TransE_UMLS")
        >>> entity_emb.shape[0] == len(entity_to_idx)
        True
    """
    weights = torch.load(f"{model_path}/model.pt", map_location="cpu") # dicee could be used 
    entity_embeddings = weights["entity_embeddings.weight"].numpy()
    relation_embeddings = weights["relation_embeddings.weight"].numpy()

    entity_to_idx = pd.read_csv(f"{model_path}/entity_to_idx.csv", index_col=0)
    entity_to_idx = {row["entity"]: idx for idx, row in entity_to_idx.iterrows()}

    relation_to_idx = pd.read_csv(f"{model_path}/relation_to_idx.csv", index_col=0)
    relation_to_idx = {row["relation"]: idx for idx, row in relation_to_idx.iterrows()}

    return entity_embeddings, relation_embeddings, entity_to_idx, relation_to_idx


def generate_embedding_samples(triples, entity_to_idx, relation_to_idx,
                                entity_embeddings, relation_embeddings,
                                neg_ratio=1, seed=42):
    """
    Create labeled samples by concatenating head, relation, and tail embeddings.

    For each valid triple ``(h, r, t)``, the function creates one positive
    sample ``[h_emb, r_emb, t_emb]`` with label ``1`` and ``neg_ratio``
    negative samples by replacing the tail entity with a corrupted one and
    assigning label ``0``.

    Args:
        triples: Iterable of ``(head, relation, tail)`` triples.
        entity_to_idx: Mapping from entity identifier to embedding row index.
        relation_to_idx: Mapping from relation identifier to embedding row index.
        entity_embeddings: Array of entity embeddings.
        relation_embeddings: Array of relation embeddings.
        neg_ratio: Number of negative samples per positive triple.
        seed: Random seed used for negative sampling.

    Returns:
        A tuple ``(X, y)`` where ``X`` is a NumPy array of concatenated
        embeddings and ``y`` contains binary labels.

    Example:
        >>> triples = [("alice", "likes", "pizza")]
        >>> entity_to_idx = {"alice": 0, "pizza": 1}
        >>> relation_to_idx = {"likes": 0}
        >>> entity_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> relation_embeddings = np.array([[0.5, 0.5]])
        >>> X, y = generate_embedding_samples(
        ...     triples,
        ...     entity_to_idx,
        ...     relation_to_idx,
        ...     entity_embeddings,
        ...     relation_embeddings,
        ...     neg_ratio=0,
        ... )
        >>> X.shape
        (1, 6)
    """
    rng = np.random.default_rng(seed)
    triple_set = set(triples)
    entities = list(entity_to_idx.keys())
    rows = []
    labels = []

    for h, r, t in triples:
        if h not in entity_to_idx or t not in entity_to_idx or r not in relation_to_idx:
            continue

        h_emb = entity_embeddings[entity_to_idx[h]]
        r_emb = relation_embeddings[relation_to_idx[r]]
        t_emb = entity_embeddings[entity_to_idx[t]]

        rows.append(np.concatenate([h_emb, r_emb, t_emb]))
        labels.append(1)

        for _ in range(neg_ratio):
            corrupt_t = t
            attempts = 0
            while (h, r, corrupt_t) in triple_set and attempts < 10:
                corrupt_t = rng.choice(entities)
                attempts += 1
            ct_emb = entity_embeddings[entity_to_idx[corrupt_t]]
            rows.append(np.concatenate([h_emb, r_emb, ct_emb]))
            labels.append(0)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y

def evaluate_link_prediction_tabpfn_wrapper(dataset_dir, experiment_dir, clf, entity_to_idx, relation_to_idx, entity_embeddings, relation_embeddings):
    """
    Train a small TabPFN classifier and run DiCEE link-prediction evaluation.

    The function loads triples and pretrained embeddings, fits TabPFN on
    embedding-based binary samples, wraps the classifier in the DiCEE
    evaluation interface, and reports filtered link-prediction metrics on the
    test split.

    Args:
        dataset_dir: Path to the knowledge-graph dataset directory.
        experiment_dir: Path to the trained embedding experiment directory.
        clf: Pretrained TabPFN classifier.
        entity_to_idx: Mapping from entity identifier to embedding row index.
        relation_to_idx: Mapping from relation identifier to embedding row index.
        entity_embeddings: Array of entity embeddings.
        relation_embeddings: Array of relation embeddings.

    Example:
        >>> # evaluate_link_prediction_tabpfn(Path("KGs/UMLS"), Path("Experiments/TransE_UMLS"))
    """

    test_triples = kg_to_array(str(dataset_dir), "test")

    # Keep debugging responsive: by default, evaluate only a small subset.
    # Set DEBUG_LP_TRIPLES=all to run the full test set.
    debug_triples = os.getenv("DEBUG_LP_TRIPLES", "10")
    if debug_triples != "all":
        test_triples = test_triples[: int(debug_triples)]

    wrapped_model = TabPFNLinkPredictionAdapter(
        clf=clf,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
    )

    er_vocab = load_pickle(str(experiment_dir + "/er_vocab.p"))
    re_vocab = load_pickle(str(experiment_dir + "/re_vocab.p"))


    print(f"Triples used for evaluate run: {len(test_triples)}")
    metrics = evaluate_link_prediction_performance(
        model=wrapped_model,
        triples=test_triples,
        er_vocab=er_vocab,
        re_vocab=re_vocab,
    )
    print("Metrics from TabPFN-wrapped evaluate_link_prediction_performance:")
    print(metrics)

def evaluate_link_prediction_tabpfn_batched(
    clf,
    test_triples: list,
    entity_to_idx: dict,
    relation_to_idx: dict,
    entity_embeddings: np.ndarray,
    relation_embeddings: np.ndarray,
    er_vocab: dict,
    re_vocab: dict, 
    k_values: list = [1, 3, 10],
    max_triples: int = None,
    triple_batch_size: int = 10,
) -> dict:
    """
    Batched version of link prediction evaluation.

    Instead of calling predict_proba once per triple (135 candidates each),
    this batches multiple triples together into one predict_proba call:
    triple_batch_size=10 means 10 × 135 = 1350 candidates per call,
    reducing the number of TabPFN forward passes significantly.

    Args:
        clf: Fitted TabPFN classifier.
        test_triples: List of (head, relation, tail) tuples.
        entity_to_idx: Entity name to embedding index mapping.
        relation_to_idx: Relation name to embedding index mapping.
        entity_embeddings: Shape (num_entities, emb_dim).
        relation_embeddings: Shape (num_relations, emb_dim).
        er_vocab: (head, relation) -> list of known correct tails.
        re_vocab: (relation, tail) -> list of known correct heads.
        k_values: List of K values for Hits@K.
        max_triples: Limit evaluation to first N triples. None = all.
        triple_batch_size: How many triples to score per predict_proba call.
            Higher = faster but more memory.

    Returns:
        Dictionary with MRR and H@K metrics.
    """
    entities = list(entity_to_idx.keys())
    num_entities = len(entities)
    emb_dim = entity_embeddings.shape[1]

    # shape = (32, 1350, 3*emb_dim) for triple_batch_size=10 and num_entities=135
    all_tail_embs = np.stack(
        [entity_embeddings[entity_to_idx[e]] for e in entities], axis=0
    ).astype(np.float32)

    eval_triples = test_triples[:max_triples] if max_triples else test_triples
    eval_triples = [
        (h, r, t) for h, r, t in eval_triples
        if h in entity_to_idx and t in entity_to_idx and r in relation_to_idx
    ]

    ranks = []

    for batch_start in tqdm(range(0, len(eval_triples), triple_batch_size),
                         desc="Evaluating link prediction"):
        batch = eval_triples[batch_start: batch_start + triple_batch_size]
        batch_size = len(batch)

        # Build candidate matrices for BOTH tail and head prediction
        # Tail: (h, r, ?) — vary tail
        # Head: (?, r, t) — vary head
        tail_candidates = np.zeros((batch_size * num_entities, 3 * emb_dim), dtype=np.float32)
      #  head_candidates = np.zeros((batch_size * num_entities, 3 * emb_dim), dtype=np.float32)

        for i, (h, r, t) in enumerate(batch):
            h_emb = entity_embeddings[entity_to_idx[h]]
            r_emb = relation_embeddings[relation_to_idx[r]]
            t_emb = entity_embeddings[entity_to_idx[t]]
            start = i * num_entities
            end = start + num_entities

            # Tail prediction: fix h and r, vary t (Memory usage: num_of_unique_entities * emb_dim * 3)
            tail_candidates[start:end, :emb_dim] = h_emb
            tail_candidates[start:end, emb_dim:2*emb_dim] = r_emb
            tail_candidates[start:end, 2*emb_dim:] = all_tail_embs

            # Head prediction: vary h, fix r and t
            # head_candidates[start:end, :emb_dim] = all_tail_embs  # reuse same entity embs
            # head_candidates[start:end, emb_dim:2*emb_dim] = r_emb
            # head_candidates[start:end, 2*emb_dim:] = t_emb

        # Two predict_proba calls per batch — one for tail, one for head
        #combined = np.concatenate([tail_candidates, head_candidates], axis=0)
        combined = np.concatenate([tail_candidates], axis=0)
        combined_scores = clf.predict_proba(combined)[:, 1]
        tail_scores_all = combined_scores[:batch_size * num_entities]
        #head_scores_all = combined_scores[batch_size * num_entities:]

        for i, (h, r, t) in enumerate(batch):
            start = i * num_entities
            
            # --- Tail prediction (h, r, ?) ---
            tail_scores = tail_scores_all[start:start + num_entities].copy()
            true_tail_idx = entity_to_idx[t]
            filt_tail = [
                entity_to_idx[kt]
                for kt in er_vocab.get((h, r), [])
                if kt != t and kt in entity_to_idx
            ]
            tail_scores[filt_tail] = -np.inf
            tail_rank = 1 + int(np.sum(tail_scores > tail_scores[true_tail_idx]))

            # --- Head prediction (?, r, t) ---
            #  head_scores = head_scores_all[start:start + num_entities].copy()
            # true_head_idx = entity_to_idx[h]
            # filt_head = [
            #     entity_to_idx[kh]
            #     for kh in re_vocab.get((r, t), [])
            #     if kh != h and kh in entity_to_idx
            # ]
            # head_scores[filt_head] = -np.inf
            # head_rank = 1 + int(np.sum(head_scores > head_scores[true_head_idx]))

            ranks.append(tail_rank)
            # ranks.append(head_rank)

    ranks = np.array(ranks)
    num_triples = len(eval_triples)
    results = {"MRR": float(np.mean(1.0 / ranks))}
    for k in k_values:
        results[f"H@{k}"] = float(np.mean(ranks <= k))

    print(f"\nLink Prediction Results ({num_triples} triples, tail only):")
    print(f"  MRR:  {results['MRR']:.4f}")
    for k in k_values:
        print(f"  H@{k}: {results[f'H@{k}']:.4f}")
    return results

def build_er_vocab(train_triples, valid_triples, test_triples) -> dict:
    """
    Build entity-relation to tail vocabulary from all splits combined.

    Used for filtered link prediction evaluation. Including all splits
    ensures that known correct answers from any split are masked out
    during ranking, following the standard filtered evaluation protocol.

    Args:
        train_triples: Training split triples.
        valid_triples: Validation split triples.
        test_triples: Test split triples.

    Returns:
        Dictionary mapping ``(head, relation)`` to a list of all known
        correct tail entities across all splits.

    Example:
        >>> er_vocab = build_er_vocab(train_triples, valid_triples, test_triples)
        >>> er_vocab[("Demir", "BornIn")]
        ['Istanbul']
    """
    er_vocab = defaultdict(list)
    for h, r, t in train_triples + valid_triples + test_triples:
        er_vocab[(h, r)].append(t)
    return dict(er_vocab)

def build_re_vocab(train_triples, valid_triples, test_triples) -> dict:
    """
    Build relation-entity to head vocabulary from all splits combined.

    Used for filtered link prediction evaluation. Including all splits
    ensures that known correct answers from any split are masked out
    during ranking, following the standard filtered evaluation protocol.

    Args:
        train_triples: Training split triples.
        valid_triples: Validation split triples.
        test_triples: Test split triples.

    Returns:
        Dictionary mapping ``(relation, tail)`` to a list of all known
        correct head entities across all splits.

    Example:
        >>> re_vocab = build_re_vocab(train_triples, valid_triples, test_triples)
        >>> re_vocab[("BornIn", "Istanbul")]
        ['Demir']
    """
    re_vocab = defaultdict(list)
    for h, r, t in train_triples + valid_triples + test_triples:
        re_vocab[(r, t)].append(h)
    return dict(re_vocab)