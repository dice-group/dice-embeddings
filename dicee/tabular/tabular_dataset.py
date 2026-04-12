from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class EntityCentricDataset:
    """Represent a KG as an entity-centric tabular dataset."""

    def __init__(self, separator="\t", not_applicable="NotApplicable"):
        self.separator = separator
        self.not_applicable = not_applicable
        self.all_relations = set()
        self.entity_relations = defaultdict(lambda: defaultdict(list))

    def read_triples(self, file_path: str) -> List[Tuple[str, str, str]]:
        """Read triples from a TSV or CSV file."""
        if self.separator in {r"\s+", "\\s+", None}:
            triples = pd.read_csv(file_path, sep=r"\s+", engine="python", header=None)
        elif self.separator in {"\t", ","}:
            triples = pd.read_csv(file_path, sep=self.separator, header=None)
        else:
            raise ValueError("Triple reader expects a whitespace, TSV, or CSV separator.")

        if triples.shape[1] < 3:
            raise ValueError(f"Expected at least 3 columns in {file_path}.")

        return list(triples.iloc[:, :3].itertuples(index=False, name=None))

    def build_entity_centric_structure(self, triples: List[Tuple[str, str, str]]) -> None:
        # Group targets by source entity and relation so we can later expand them into rows.
        self.all_relations = set()
        self.entity_relations = defaultdict(lambda: defaultdict(list))
        for h, r, t in triples:
            self.all_relations.add(r)
            self.entity_relations[h][r].append(t)

    def triples_to_entity_centric_tabular(
        self, triples: List[Tuple[str, str, str]], labels: List[int] = None
    ) -> pd.DataFrame:
        self.build_entity_centric_structure(triples)
        rows = []
        sorted_relations = sorted(self.all_relations)
        triple_to_label = {}

        if labels is not None:
            for index, triple in enumerate(triples):
                triple_to_label[triple] = labels[index]

        for entity in self.entity_relations:
            entity_rels = self.entity_relations[entity]
            relation_values = {}
            for rel in sorted_relations:
                relation_values[rel] = entity_rels.get(rel, [self.not_applicable])

            # An entity may have multiple tails for one relation, so we emit multiple rows if needed.
            max_values = max(len(values) for values in relation_values.values())
            for index in range(max_values):
                row = {"Entity": entity}
                for rel in sorted_relations:
                    values = relation_values[rel]
                    row[rel] = values[index] if index < len(values) else values[-1]

                label = 1.0
                if labels is not None:
                    found = False
                    for rel in sorted_relations:
                        if row[rel] == self.not_applicable:
                            continue
                        triple = (entity, rel, row[rel])
                        if triple in triple_to_label:
                            label = triple_to_label[triple]
                            found = True
                            break
                    if not found:
                        label = 1.0

                row["Label"] = label
                rows.append(row)

        return pd.DataFrame(rows)

    def _generate_negative_samples(
        self, positive_triples: List[Tuple[str, str, str]], num_negative: int
    ) -> List[Tuple[str, str, str]]:
        # Corrupt heads or tails while avoiding duplicates and known positives.
        existing_triples = set(positive_triples)
        negative_triples = []
        all_entities = set()

        for h, _, t in positive_triples:
            all_entities.add(h)
            all_entities.add(t)

        entities_list = list(all_entities)
        attempts = 0
        max_attempts = num_negative * 10

        while len(negative_triples) < num_negative and attempts < max_attempts:
            h, r, t = positive_triples[np.random.randint(len(positive_triples))]
            if np.random.random() < 0.5:
                corrupted = (np.random.choice(entities_list), r, t)
            else:
                corrupted = (h, r, np.random.choice(entities_list))

            if corrupted not in existing_triples:
                negative_triples.append(corrupted)
                existing_triples.add(corrupted)
            attempts += 1

        return negative_triples

    def generate_entity_centric_dataset(
        self, train_file: str, valid_file: str = None, test_file: str = None, negative_ratio: float = 1.0
    ) -> Dict:
        # Each split is converted independently after adding synthetic negatives.
        result = {}

        train_triples = self.read_triples(train_file)
        train_negative = self._generate_negative_samples(train_triples, int(len(train_triples) * negative_ratio))
        all_train_triples = train_triples + train_negative
        train_labels = [1] * len(train_triples) + [0] * len(train_negative)
        result["train"] = self.triples_to_entity_centric_tabular(all_train_triples, train_labels)

        if valid_file:
            valid_triples = self.read_triples(valid_file)
            valid_negative = self._generate_negative_samples(valid_triples, int(len(valid_triples) * negative_ratio))
            all_valid_triples = valid_triples + valid_negative
            valid_labels = [1] * len(valid_triples) + [0] * len(valid_negative)
            result["valid"] = self.triples_to_entity_centric_tabular(all_valid_triples, valid_labels)

        if test_file:
            test_triples = self.read_triples(test_file)
            test_negative = self._generate_negative_samples(test_triples, int(len(test_triples) * negative_ratio))
            all_test_triples = test_triples + test_negative
            test_labels = [1] * len(test_triples) + [0] * len(test_negative)
            result["test"] = self.triples_to_entity_centric_tabular(all_test_triples, test_labels)

        return result


class TripleCentricDataset:
    """Represent KG triples as a triple-centric tabular dataset."""

    def __init__(self, separator="\t"):
        self.separator = separator
        self.entity_to_idx = {}
        self.relation_to_idx = {}
        self.head_to_relations = defaultdict(set)
        self.head_relation_to_tails = defaultdict(set)
        self.tail_to_relations = defaultdict(set)
        self.tail_relation_to_heads = defaultdict(set)

    def read_triples(self, file_path: str) -> List[Tuple[str, str, str]]:
        """Read triples from a TSV or CSV file."""
        if self.separator in {r"\s+", "\\s+", None}:
            triples = pd.read_csv(file_path, sep=r"\s+", engine="python", header=None)
        elif self.separator in {"\t", ","}:
            triples = pd.read_csv(file_path, sep=self.separator, header=None)
        else:
            raise ValueError("Triple reader expects a whitespace, TSV, or CSV separator.")

        if triples.shape[1] < 3:
            raise ValueError(f"Expected at least 3 columns in {file_path}.")

        return list(triples.iloc[:, :3].itertuples(index=False, name=None))

    def build_vocabulary(self, triples: List[Tuple[str, str, str]]) -> None:
        entities = {entity for h, _, t in triples for entity in (h, t)}
        relations = {relation for _, relation, _ in triples}

        for index, entity in enumerate(sorted(entities)):
            self.entity_to_idx[entity] = index

        for index, relation in enumerate(sorted(relations)):
            self.relation_to_idx[relation] = index

    def build_graph_structure(self, triples: List[Tuple[str, str, str]]) -> None:
        # Store only first-hop neighborhood statistics used by the feature builder.
        self.head_to_relations = defaultdict(set)
        self.head_relation_to_tails = defaultdict(set)
        self.tail_to_relations = defaultdict(set)
        self.tail_relation_to_heads = defaultdict(set)

        for h, r, t in triples:
            self.head_to_relations[h].add(r)
            self.head_relation_to_tails[(h, r)].add(t)
            self.tail_to_relations[t].add(r)
            self.tail_relation_to_heads[(t, r)].add(h)

    def compute_first_hop_features(self, entity: str) -> Dict[str, float]:
        # These features summarize how an entity participates in the training graph.
        features = {}
        out_relations = self.head_to_relations.get(entity, set())
        features["num_out_relations"] = len(out_relations)
        out_neighbors = sum(len(self.head_relation_to_tails.get((entity, r), set())) for r in out_relations)
        features["out_degree"] = out_neighbors
        features["avg_out_neighbors"] = out_neighbors / len(out_relations) if out_relations else 0

        in_relations = self.tail_to_relations.get(entity, set())
        features["num_in_relations"] = len(in_relations)
        in_neighbors = sum(len(self.tail_relation_to_heads.get((entity, r), set())) for r in in_relations)
        features["in_degree"] = in_neighbors
        features["avg_in_neighbors"] = in_neighbors / len(in_relations) if in_relations else 0
        features["total_degree"] = features["out_degree"] + features["in_degree"]
        return features

    def triples_to_tabular(
        self, triples: List[Tuple[str, str, str]], labels: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        relation_counts = defaultdict(int)
        entity_feature_cache = {}
        num_triples = len(triples)
        # The feature layout is fixed, so we can allocate the matrix once.
        x_data = np.empty((num_triples, 16), dtype=np.float32)

        for _, relation, _ in triples:
            relation_counts[relation] += 1

        for row_index, (h, r, t) in enumerate(triples):
            # Reuse entity-level graph features across many triples in the same split.
            h_features = entity_feature_cache.setdefault(h, self.compute_first_hop_features(h))
            t_features = entity_feature_cache.setdefault(t, self.compute_first_hop_features(t))
            x_data[row_index] = (
                self.entity_to_idx[h],
                self.relation_to_idx[r],
                self.entity_to_idx[t],
                h_features["out_degree"],
                h_features["in_degree"],
                h_features["num_out_relations"],
                h_features["num_in_relations"],
                h_features["avg_out_neighbors"],
                h_features["total_degree"],
                t_features["out_degree"],
                t_features["in_degree"],
                t_features["num_out_relations"],
                t_features["num_in_relations"],
                t_features["avg_in_neighbors"],
                t_features["total_degree"],
                relation_counts[r] / num_triples,
            )

        y_data = np.array(labels, dtype=np.int32) if labels is not None else np.ones(len(triples), dtype=np.int32)
        return x_data, y_data

    def filter_unseen_triples(
        self, triples: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        """Drop triples that reference entities or relations unseen in training."""
        # The tabular features rely on vocabularies built from the training split only.
        return [
            (h, r, t)
            for h, r, t in triples
            if h in self.entity_to_idx and r in self.relation_to_idx and t in self.entity_to_idx
        ]

    def generate_negative_samples(
        self, positive_triples: List[Tuple[str, str, str]], num_negative: int = None, corruption_mode: str = "both"
    ) -> List[Tuple[str, str, str]]:
        # Build binary-classification negatives by corrupting one side of a positive triple.
        if num_negative is None:
            num_negative = len(positive_triples)

        existing_triples = set(positive_triples)
        negative_triples = []
        entities = list(self.entity_to_idx.keys())
        attempts = 0
        max_attempts = num_negative * 10

        while len(negative_triples) < num_negative and attempts < max_attempts:
            h, r, t = positive_triples[np.random.randint(len(positive_triples))]
            if corruption_mode == "head":
                corrupted = (np.random.choice(entities), r, t)
            elif corruption_mode == "tail":
                corrupted = (h, r, np.random.choice(entities))
            else:
                corrupted = (np.random.choice(entities), r, t) if np.random.random() < 0.5 else (h, r, np.random.choice(entities))

            if corrupted not in existing_triples:
                negative_triples.append(corrupted)
                existing_triples.add(corrupted)
            attempts += 1

        return negative_triples

    def load_and_convert(
        self, train_file: str, valid_file: str = None, test_file: str = None, negative_ratio: float = 1.0
    ) -> Dict:
        # Training triples define both the vocabularies and the graph-derived features.
        train_triples = self.read_triples(train_file)
        self.build_vocabulary(train_triples)
        self.build_graph_structure(train_triples)

        result = {}
        train_negative = self.generate_negative_samples(train_triples, int(len(train_triples) * negative_ratio))
        all_train_triples = train_triples + train_negative
        train_labels = [1] * len(train_triples) + [0] * len(train_negative)
        x_train, y_train = self.triples_to_tabular(all_train_triples, train_labels)
        result["train"] = {"X": x_train, "y": y_train}

        if valid_file:
            valid_triples = self.read_triples(valid_file)
            valid_triples = self.filter_unseen_triples(valid_triples)
            valid_negative = self.generate_negative_samples(valid_triples, int(len(valid_triples) * negative_ratio))
            all_valid_triples = valid_triples + valid_negative
            valid_labels = [1] * len(valid_triples) + [0] * len(valid_negative)
            x_valid, y_valid = self.triples_to_tabular(all_valid_triples, valid_labels)
            result["valid"] = {"X": x_valid, "y": y_valid}

        if test_file:
            test_triples = self.read_triples(test_file)
            test_triples = self.filter_unseen_triples(test_triples)
            test_negative = self.generate_negative_samples(test_triples, int(len(test_triples) * negative_ratio))
            all_test_triples = test_triples + test_negative
            test_labels = [1] * len(test_triples) + [0] * len(test_negative)
            x_test, y_test = self.triples_to_tabular(all_test_triples, test_labels)
            result["test"] = {"X": x_test, "y": y_test}

        return result
