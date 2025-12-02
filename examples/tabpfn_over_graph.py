"""
Knowledge Graph to Tabular Data Conversion and TabPFN Training

This module provides complete functionality for:
1. Converting knowledge graphs to tabular features
2. Training TabPFN for link prediction
3. Evaluating model performance

Based on: https://github.com/dice-group/dice-embeddings/issues/352
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tabpfn import TabPFNClassifier


class KGToTabularConverter:
    """
    Converts Knowledge Graph triples into tabular features for TabPFN.
    
    The conversion process:
    1. Read triples (h, r, t) from text files
    2. Create entity and relation vocabularies
    3. Compute first-hop neighborhood features
    4. Generate tabular features for each triple
    """
    
    def __init__(self, separator='\t'):
        """
        Initialize the converter.
        
        Parameters
        ----------
        separator : str
            Delimiter used in the triple files (default: tab)
        """
        self.separator = separator
        self.entity_to_idx = {}
        self.relation_to_idx = {}
        self.idx_to_entity = {}
        self.idx_to_relation = {}
        
        # Graph structure for neighborhood computation
        self.head_to_relations = defaultdict(set)  # h -> set of r
        self.head_relation_to_tails = defaultdict(set)  # (h,r) -> set of t
        self.tail_to_relations = defaultdict(set)  # t -> set of r
        self.tail_relation_to_heads = defaultdict(set)  # (t,r) -> set of h
        
        # Statistics
        self.num_entities = 0
        self.num_relations = 0
        
    def read_triples(self, file_path: str) -> List[Tuple[str, str, str]]:
        """
        Read triples from a file.
        
        Parameters
        ----------
        file_path : str
            Path to the triple file
            
        Returns
        -------
        List[Tuple[str, str, str]]
            List of (head, relation, tail) tuples
        """
        triples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(self.separator)
                if len(parts) == 3:
                    h, r, t = parts
                    triples.append((h.strip(), r.strip(), t.strip()))
        return triples
    
    def build_vocabulary(self, triples: List[Tuple[str, str, str]]) -> None:
        """
        Build entity and relation vocabularies from triples.
        
        Parameters
        ----------
        triples : List[Tuple[str, str, str]]
            List of (head, relation, tail) tuples
        """
        entities = set()
        relations = set()
        
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
        
        # Create mappings
        for idx, entity in enumerate(sorted(entities)):
            self.entity_to_idx[entity] = idx
            self.idx_to_entity[idx] = entity
            
        for idx, relation in enumerate(sorted(relations)):
            self.relation_to_idx[relation] = idx
            self.idx_to_relation[idx] = relation
        
        self.num_entities = len(entities)
        self.num_relations = len(relations)
        
        print(f"Vocabulary built: {self.num_entities} entities, {self.num_relations} relations")
    
    def build_graph_structure(self, triples: List[Tuple[str, str, str]]) -> None:
        """
        Build graph structure for computing neighborhood features.
        
        Parameters
        ----------
        triples : List[Tuple[str, str, str]]
            List of (head, relation, tail) tuples
        """
        for h, r, t in triples:
            # Outgoing edges from head
            self.head_to_relations[h].add(r)
            self.head_relation_to_tails[(h, r)].add(t)
            
            # Incoming edges to tail
            self.tail_to_relations[t].add(r)
            self.tail_relation_to_heads[(t, r)].add(h)
        
        print(f"Graph structure built with {len(triples)} triples")
    
    def compute_first_hop_features(self, entity: str) -> Dict[str, float]:
        """
        Compute first-hop neighborhood features for an entity.
        
        Features include:
        - out_degree: Number of outgoing edges
        - in_degree: Number of incoming edges
        - num_out_relations: Number of unique outgoing relation types
        - num_in_relations: Number of unique incoming relation types
        - avg_out_neighbors: Average number of neighbors per outgoing relation
        
        Parameters
        ----------
        entity : str
            Entity name
            
        Returns
        -------
        Dict[str, float]
            Dictionary of feature name to value
        """
        features = {}
        
        # Outgoing edges
        out_relations = self.head_to_relations.get(entity, set())
        features['num_out_relations'] = len(out_relations)
        
        out_neighbors = 0
        for r in out_relations:
            out_neighbors += len(self.head_relation_to_tails.get((entity, r), set()))
        features['out_degree'] = out_neighbors
        features['avg_out_neighbors'] = out_neighbors / len(out_relations) if out_relations else 0
        
        # Incoming edges
        in_relations = self.tail_to_relations.get(entity, set())
        features['num_in_relations'] = len(in_relations)
        
        in_neighbors = 0
        for r in in_relations:
            in_neighbors += len(self.tail_relation_to_heads.get((entity, r), set()))
        features['in_degree'] = in_neighbors
        features['avg_in_neighbors'] = in_neighbors / len(in_relations) if in_relations else 0
        
        # Total connectivity
        features['total_degree'] = features['out_degree'] + features['in_degree']
        features['total_relations'] = features['num_out_relations'] + features['num_in_relations']
        
        return features
    
    def triples_to_tabular(self, triples: List[Tuple[str, str, str]], 
                          labels: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert triples to tabular features for TabPFN.
        
        Features generated for each triple (h, r, t):
        - h_idx: Head entity index
        - r_idx: Relation index
        - t_idx: Tail entity index
        - h_out_degree: Outgoing degree of head
        - h_in_degree: Incoming degree of head
        - h_num_out_relations: Number of unique outgoing relations from head
        - h_num_in_relations: Number of unique incoming relations to head
        - t_out_degree: Outgoing degree of tail
        - t_in_degree: Incoming degree of tail
        - relation_frequency: How often this relation appears in the graph
        
        Parameters
        ----------
        triples : List[Tuple[str, str, str]]
            List of (head, relation, tail) tuples
        labels : List[int], optional
            Binary labels (1 for positive, 0 for negative triples)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Features (X) and labels (y) as numpy arrays
        """
        features_list = []
        
        # Pre-compute relation frequencies
        relation_counts = defaultdict(int)
        for h, r, t in triples:
            relation_counts[r] += 1
        
        for h, r, t in triples:
            # Basic indices
            h_idx = self.entity_to_idx[h]
            r_idx = self.relation_to_idx[r]
            t_idx = self.entity_to_idx[t]
            
            # Head features
            h_features = self.compute_first_hop_features(h)
            
            # Tail features
            t_features = self.compute_first_hop_features(t)
            
            # Relation features
            r_frequency = relation_counts[r] / len(triples)
            
            # Combine all features
            feature_vector = [
                h_idx,
                r_idx,
                t_idx,
                h_features['out_degree'],
                h_features['in_degree'],
                h_features['num_out_relations'],
                h_features['num_in_relations'],
                h_features['avg_out_neighbors'],
                h_features['total_degree'],
                t_features['out_degree'],
                t_features['in_degree'],
                t_features['num_out_relations'],
                t_features['num_in_relations'],
                t_features['avg_in_neighbors'],
                t_features['total_degree'],
                r_frequency
            ]
            
            features_list.append(feature_vector)
        
        X = np.array(features_list, dtype=np.float32)
        
        if labels is not None:
            y = np.array(labels, dtype=np.int32)
        else:
            y = np.ones(len(triples), dtype=np.int32)  # All positive by default
        
        return X, y
    
    def generate_negative_samples(self, positive_triples: List[Tuple[str, str, str]], 
                                 num_negative: int = None,
                                 corruption_mode: str = 'both') -> List[Tuple[str, str, str]]:
        """
        Generate negative triples by corrupting positive ones.
        
        Parameters
        ----------
        positive_triples : List[Tuple[str, str, str]]
            List of positive triples
        num_negative : int, optional
            Number of negative samples to generate (default: same as positive)
        corruption_mode : str
            'head': corrupt only head entity
            'tail': corrupt only tail entity
            'both': corrupt either head or tail
            
        Returns
        -------
        List[Tuple[str, str, str]]
            List of negative triples
        """
        if num_negative is None:
            num_negative = len(positive_triples)
        
        # Create set of existing triples for filtering
        existing_triples = set(positive_triples)
        
        negative_triples = []
        entities = list(self.entity_to_idx.keys())
        
        attempts = 0
        max_attempts = num_negative * 10
        
        while len(negative_triples) < num_negative and attempts < max_attempts:
            # Sample a positive triple
            h, r, t = positive_triples[np.random.randint(len(positive_triples))]
            
            # Corrupt it
            if corruption_mode == 'head':
                corrupted = (np.random.choice(entities), r, t)
            elif corruption_mode == 'tail':
                corrupted = (h, r, np.random.choice(entities))
            else:  # both
                if np.random.random() < 0.5:
                    corrupted = (np.random.choice(entities), r, t)
                else:
                    corrupted = (h, r, np.random.choice(entities))
            
            # Only add if it's not an existing triple
            if corrupted not in existing_triples:
                negative_triples.append(corrupted)
                existing_triples.add(corrupted)
            
            attempts += 1
        
        if len(negative_triples) < num_negative:
            print(f"Warning: Only generated {len(negative_triples)} negative samples out of {num_negative} requested")
        
        return negative_triples
    
    def load_and_convert(self, train_file: str, valid_file: str = None, 
                        test_file: str = None, negative_ratio: float = 1.0) -> Dict:
        """
        Complete pipeline: Load KG and convert to tabular data.
        
        Parameters
        ----------
        train_file : str
            Path to training triples
        valid_file : str, optional
            Path to validation triples
        test_file : str, optional
            Path to test triples
        negative_ratio : float
            Ratio of negative to positive samples
            
        Returns
        -------
        Dict
            Dictionary with 'train', 'valid', 'test' keys, each containing
            'X' (features) and 'y' (labels)
        """
        print("Loading knowledge graph...")
        
        # Load training triples
        train_triples = self.read_triples(train_file)
        print(f"Loaded {len(train_triples)} training triples")
        
        # Build vocabulary from training data
        self.build_vocabulary(train_triples)
        
        # Build graph structure
        self.build_graph_structure(train_triples)
        
        result = {}
        
        # Convert training data
        print("\nGenerating negative samples for training...")
        num_negative = int(len(train_triples) * negative_ratio)
        train_negative = self.generate_negative_samples(train_triples, num_negative)
        
        print("Converting training data to tabular format...")
        all_train_triples = train_triples + train_negative
        train_labels = [1] * len(train_triples) + [0] * len(train_negative)
        X_train, y_train = self.triples_to_tabular(all_train_triples, train_labels)
        
        result['train'] = {'X': X_train, 'y': y_train}
        print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Convert validation data if provided
        if valid_file:
            valid_triples = self.read_triples(valid_file)
            print(f"\nLoaded {len(valid_triples)} validation triples")
            
            # Filter triples with unknown entities/relations
            valid_triples = [(h, r, t) for h, r, t in valid_triples 
                           if h in self.entity_to_idx and r in self.relation_to_idx and t in self.entity_to_idx]
            print(f"Filtered to {len(valid_triples)} validation triples with known entities")
            
            valid_negative = self.generate_negative_samples(valid_triples, 
                                                           int(len(valid_triples) * negative_ratio))
            all_valid_triples = valid_triples + valid_negative
            valid_labels = [1] * len(valid_triples) + [0] * len(valid_negative)
            X_valid, y_valid = self.triples_to_tabular(all_valid_triples, valid_labels)
            
            result['valid'] = {'X': X_valid, 'y': y_valid}
            print(f"Validation data: {X_valid.shape[0]} samples, {X_valid.shape[1]} features")
        
        # Convert test data if provided
        if test_file:
            test_triples = self.read_triples(test_file)
            print(f"\nLoaded {len(test_triples)} test triples")
            
            # Filter triples with unknown entities/relations
            test_triples = [(h, r, t) for h, r, t in test_triples 
                          if h in self.entity_to_idx and r in self.relation_to_idx and t in self.entity_to_idx]
            print(f"Filtered to {len(test_triples)} test triples with known entities")
            
            test_negative = self.generate_negative_samples(test_triples, 
                                                          int(len(test_triples) * negative_ratio))
            all_test_triples = test_triples + test_negative
            test_labels = [1] * len(test_triples) + [0] * len(test_negative)
            X_test, y_test = self.triples_to_tabular(all_test_triples, test_labels)
            
            result['test'] = {'X': X_test, 'y': y_test}
            print(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        return result


def load_kg_for_tabpfn(dataset_dir: str, negative_ratio: float = 1.0) -> Dict:
    """
    Convenience function to load a knowledge graph dataset for TabPFN.
    
    Parameters
    ----------
    dataset_dir : str
        Path to directory containing train.txt, valid.txt, test.txt
    negative_ratio : float
        Ratio of negative to positive samples
        
    Returns
    -------
    Dict
        Dictionary with train/valid/test splits containing X and y
        
    Example
    -------
    >>> data = load_kg_for_tabpfn('KGs/UMLS')
    >>> X_train, y_train = data['train']['X'], data['train']['y']
    >>> X_test, y_test = data['test']['X'], data['test']['y']
    """
    import os
    
    converter = KGToTabularConverter()
    
    train_file = os.path.join(dataset_dir, 'train.txt')
    valid_file = os.path.join(dataset_dir, 'valid.txt')
    test_file = os.path.join(dataset_dir, 'test.txt')
    
    # Check which files exist
    files_to_use = {
        'train': train_file if os.path.exists(train_file) else None,
        'valid': valid_file if os.path.exists(valid_file) else None,
        'test': test_file if os.path.exists(test_file) else None
    }
    
    if files_to_use['train'] is None:
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    return converter.load_and_convert(
        train_file=files_to_use['train'],
        valid_file=files_to_use['valid'],
        test_file=files_to_use['test'],
        negative_ratio=negative_ratio
    )


def train_and_evaluate(dataset_dir='KGs/UMLS', negative_ratio=1.0, device='cpu', 
                      max_train_samples=1000, n_estimators=8):
    """
    Complete pipeline for training TabPFN on KG data.
    
    Parameters
    ----------
    dataset_dir : str
        Path to KG dataset directory
    negative_ratio : float
        Ratio of negative to positive samples
    device : str
        Device to use for TabPFN ('cpu' or 'cuda')
    max_train_samples : int
        Maximum training samples (TabPFN limitation)
    n_estimators : int
        Number of ensemble estimators for TabPFN
        
    Returns
    -------
    Dict
        Dictionary containing test and validation metrics
    """
    
    print("="*80)
    print("TabPFN Link Prediction on Knowledge Graph")
    print("="*80)
    
    # Step 1: Load and convert data
    print("\n[1/4] Loading and converting knowledge graph data...")
    data = load_kg_for_tabpfn(dataset_dir, negative_ratio=negative_ratio)
    
    X_train, y_train = data['train']['X'], data['train']['y']
    
    if 'valid' not in data or 'test' not in data:
        print("Warning: Missing validation or test data")
        return None
        
    X_valid, y_valid = data['valid']['X'], data['valid']['y']
    X_test, y_test = data['test']['X'], data['test']['y']
    
    print(f"\nDataset statistics:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_valid.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    
    # Step 2: Initialize TabPFN
    print(f"\n[2/4] Initializing TabPFN classifier (device={device})...")
    clf = TabPFNClassifier(device=device, n_estimators=n_estimators, 
                          ignore_pretraining_limits=True)
    
    # Step 3: Train
    print("\n[3/4] Training TabPFN...")
    print("Note: TabPFN is pre-trained, so 'training' is actually just fitting to the data")
    
    # TabPFN has a limit on training samples
    if X_train.shape[0] > max_train_samples:
        print(f"\nWarning: Training set has {X_train.shape[0]} samples")
        print(f"TabPFN works best with <={max_train_samples} samples")
        print(f"Randomly sampling {max_train_samples} samples for training...")
        
        indices = np.random.choice(X_train.shape[0], max_train_samples, replace=False)
        X_train_sample = X_train[indices]
        y_train_sample = y_train[indices]
    else:
        X_train_sample = X_train
        y_train_sample = y_train
    
    try:
        clf.fit(X_train_sample, y_train_sample)
        print("✓ Training complete")
    except Exception as e:
        print(f"Error during training: {e}")
        print("\nNote: TabPFN has limits on dataset size and features.")
        print("Try reducing negative_ratio or feature dimensions.")
        return None
    
    # Step 4: Evaluate
    print("\n[4/4] Evaluating on test set...")
    
    # Predict on test set
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*80)
    print("Test Results:")
    print("="*80)
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print("="*80)
    
    # Validation set evaluation
    print("\n[Bonus] Evaluating on validation set...")
    y_valid_pred = clf.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    print(f"Validation Accuracy: {valid_accuracy:.4f}")
    
    return {
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_auc': auc,
        'valid_accuracy': valid_accuracy,
        'model': clf
    }


def experiment_with_different_ratios(dataset_dir='KGs/UMLS', ratios=None):
    """
    Experiment with different negative sampling ratios.
    
    Parameters
    ----------
    dataset_dir : str
        Path to KG dataset directory
    ratios : List[float], optional
        List of negative ratios to test (default: [0.5, 1.0, 2.0])
    """
    if ratios is None:
        ratios = [0.5, 1.0, 2.0]
    
    print("\n" + "="*80)
    print("Experimenting with Different Negative Sampling Ratios")
    print("="*80)
    
    results = {}
    
    for ratio in ratios:
        print(f"\n\n{'='*80}")
        print(f"Testing with negative_ratio = {ratio}")
        print('='*80)
        
        try:
            metrics = train_and_evaluate(dataset_dir=dataset_dir, negative_ratio=ratio)
            results[ratio] = metrics
        except Exception as e:
            print(f"Failed with ratio {ratio}: {e}")
            results[ratio] = None
    
    # Summary
    print("\n\n" + "="*80)
    print("Summary of Results")
    print("="*80)
    print(f"{'Ratio':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-"*80)
    for ratio in ratios:
        if results[ratio]:
            m = results[ratio]
            print(f"{ratio:<10.1f} {m['test_accuracy']:<10.4f} {m['test_precision']:<10.4f} "
                  f"{m['test_recall']:<10.4f} {m['test_f1']:<10.4f} {m['test_auc']:<10.4f}")
        else:
            print(f"{ratio:<10.1f} {'Failed'}")
    print("="*80)
    
    return results


def convert_only(dataset_dir='KGs/UMLS', negative_ratio=1.0):
    """
    Only convert KG data to tabular format without training.
    
    Parameters
    ----------
    dataset_dir : str
        Path to KG dataset directory
    negative_ratio : float
        Ratio of negative to positive samples
    """
    print("="*80)
    print("Knowledge Graph to Tabular Data Conversion for TabPFN")
    print("="*80)
    
    # Load dataset
    data = load_kg_for_tabpfn(dataset_dir, negative_ratio=negative_ratio)
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    for split in ['train', 'valid', 'test']:
        if split in data:
            X, y = data[split]['X'], data[split]['y']
            print(f"\n{split.capitalize()}:")
            print(f"  Shape: {X.shape}")
            print(f"  Positive samples: {np.sum(y == 1)}")
            print(f"  Negative samples: {np.sum(y == 0)}")
            print(f"  Feature names: h_idx, r_idx, t_idx, h_out_deg, h_in_deg, ...")
    
    print("\n" + "="*80)
    print("Data is ready for TabPFN!")
    print("="*80)
    
    return data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Knowledge Graph to Tabular Conversion and TabPFN Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert only (no training)
  python kg_to_tabular.py --convert-only --dataset KGs/UMLS
  
  # Train TabPFN
  python kg_to_tabular.py --train --dataset KGs/UMLS --negative_ratio 1.0
  
  # Run experiments with different ratios
  python kg_to_tabular.py --experiment --dataset KGs/UMLS
        """
    )
    
    parser.add_argument('--dataset', type=str, default='KGs/UMLS',
                       help='Path to KG dataset directory (default: KGs/UMLS)')
    parser.add_argument('--negative_ratio', type=float, default=1.0,
                       help='Ratio of negative to positive samples (default: 1.0)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for TabPFN (default: cpu)')
    parser.add_argument('--max_train_samples', type=int, default=1000,
                       help='Maximum training samples for TabPFN (default: 1000)')
    parser.add_argument('--n_estimators', type=int, default=8,
                       help='Number of TabPFN ensemble estimators (default: 8)')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--convert-only', action='store_true',
                           help='Only convert to tabular format, do not train')
    mode_group.add_argument('--train', action='store_true',
                           help='Train TabPFN model')
    mode_group.add_argument('--experiment', action='store_true',
                           help='Run experiments with different negative ratios')
    
    args = parser.parse_args()
    
    # Default to convert-only if no mode specified
    if not (args.convert_only or args.train or args.experiment):
        args.convert_only = True
    
    if args.convert_only:
        convert_only(args.dataset, args.negative_ratio)
    elif args.experiment:
        experiment_with_different_ratios(args.dataset)
    elif args.train:
        train_and_evaluate(
            dataset_dir=args.dataset,
            negative_ratio=args.negative_ratio,
            device=args.device,
            max_train_samples=args.max_train_samples,
            n_estimators=args.n_estimators
        )
