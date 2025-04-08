import random
import dspy
from dspy.teleprompt import * 
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from dicee.static_funcs_training import evaluate_lp_k_vs_all
from ..abstract import AbstractBaseLinkPredictorClass
from dicee.knowledge_graph import KG
import torch
from ..schemas import PredictionItem


class LM_Call_Signature(dspy.Signature):
    source: str = dspy.InputField(description="The source entity")
    relation: str = dspy.InputField(description="The relation")
    target_entities: List[str] = dspy.InputField(
        description="The list of target entities"
    )
    predictions: List[PredictionItem] = dspy.OutputField(
        description="The list of predicted entities with scores"
    )


class DSPy_RCL(AbstractBaseLinkPredictorClass):

    def __init__(
        self,
        knowledge_graph: KG = None,
        base_url: str = None,
        api_key: str = None,
        llm_model: str = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_relation_examples: int = 2000,
        use_val: bool = True,
        exclude_source: bool = False,
    ) -> None:
        super().__init__(knowledge_graph, name="DSPy_RCL")
        assert base_url is not None and isinstance(base_url, str)
        self.base_url = base_url
        self.api_key = api_key
        self.llm_model = llm_model
        self.temperature = temperature
        self.seed = seed
        self.max_relation_examples = max_relation_examples
        self.exclude_source = exclude_source
        # hardcoded for now
        self.lm = dspy.LM(
            model="openai/tentris", api_key=self.api_key, base_url=self.base_url, cache=False
        )
        dspy.configure(lm=self.lm)
        self.model = dspy.ChainOfThought(LM_Call_Signature, max_tokens=4000)

        # Training dataset
        self.train_set: List[Tuple[str]] = [
            (
                self.idx_to_entity[idx_h],
                self.idx_to_relation[idx_r],
                self.idx_to_entity[idx_t],
            )
            for idx_h, idx_r, idx_t in self.kg.train_set.tolist()
        ]
        # Validation dataset
        self.val_set: List[Tuple[str]] = [
            (
                self.idx_to_entity[idx_h],
                self.idx_to_relation[idx_r],
                self.idx_to_entity[idx_t],
            )
            for idx_h, idx_r, idx_t in self.kg.valid_set.tolist()
        ]

        triples = self.train_set + self.val_set if use_val else self.train_set
        self.triples = triples

        self.relation_to_idx = {v: k for k, v in self.idx_to_relation.items()}
        # Add this after self.idx_to_relation is initialized
        self.relation_to_idx = {v: k for k, v in self.idx_to_relation.items()}

        # Create a mapping from relation to all triples using that relation
        self.relation_to_triples = {}
        for s, p, o in triples:
            if p not in self.relation_to_triples:
                self.relation_to_triples[p] = []
            self.relation_to_triples[p].append((s, p, o))

        self.target_entities = list(sorted(self.entity_to_idx.keys()))
    
    def f1_score_metric(self, example: dspy.Example, pred: dspy.Prediction, trace=None):
        ground_truth = set(item.entity if hasattr(item, 'entity') else item for item in example.predictions)
        predicted_entities = set(item.entity for item in pred.predictions)

        true_positives = len(ground_truth & predicted_entities)
        precision = true_positives / len(predicted_entities) if predicted_entities else 0.0
        recall = true_positives / len(ground_truth) if ground_truth else 0.0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    

    def generate_examples(self):
        """
        Generate examples directly from knowledge graph triples.
        Includes target_entities to match the signature requirements.
        """
        examples = []
        # Use actual train/valid triples from the KG
        for idx_h, idx_r, idx_t in self.kg.train_set.tolist():
            source = self.idx_to_entity[idx_h]
            relation = self.idx_to_relation[idx_r]
            target = self.idx_to_entity[idx_t]
            
            # Create a list of target entities including the correct one
            # and a few random ones to match your signature requirements
            all_entities = list(self.entity_to_idx.keys())
            random_entities = random.sample(
                [e for e in all_entities if e != target], 
                min(4, len(all_entities)-1)
            )
            target_entities = [target] + random_entities
            
            example = dspy.Example(
                source=source,
                relation=relation,
                target_entities=target_entities,  # Add this field to match your signature
                predictions=[PredictionItem(entity=target, score=1.0)]  # For metric calculation
            ).with_inputs("source", "relation", "target_entities")  # Include all required inputs
            
            examples.append(example)
        return examples

    def generate_train_test_split(self, examples):
        """
        Split the examples into training and testing sets based on the knowledge graph's train and test sets.

        Args:
            examples (List[dspy.Example]): A list of DSPy examples to split.

        Returns:
            Tuple[List[dspy.Example], List[dspy.Example]]: A tuple containing the training and testing examples.
        """
        # Create dictionaries to map (source, relation) pairs to examples
        example_map = {}
        for example in examples:
            key = (example.source, example.relation)
            example_map[key] = example
        
        # Initialize train and test example lists
        train_examples = []
        test_examples = []
        
        # Use the knowledge graph's train and test sets to split examples
        for idx_h, idx_r, idx_t in self.kg.train_set.tolist():
            h, r = self.idx_to_entity[idx_h], self.idx_to_relation[idx_r]
            key = (h, r)
            if key in example_map:
                train_examples.append(example_map[key])
        
        for idx_h, idx_r, idx_t in self.kg.test_set.tolist():
            h, r = self.idx_to_entity[idx_h], self.idx_to_relation[idx_r]
            key = (h, r)
            if key in example_map:
                test_examples.append(example_map[key])
        
        return train_examples, test_examples

 

    def train_labeledFewShot(self, train_set, few_shot_k):
        lfs_optimizer = LabeledFewShot(k=few_shot_k)
        lfs_model = lfs_optimizer.compile(self.model, trainset=train_set)
        self.model = lfs_model
        lfs_model.save("./lfs_model.json")
        return lfs_model

    def train_COPRO(self, train_set):
        copro_optimizer = COPRO(
            metric=self.metric,
            prompt_model=self.lm,
            breadth=5,
            depth=3,
            init_temperature=1.4,
        )
        copro_model = copro_optimizer.compile(
            self.model,
            trainset=train_set,
            eval_kwargs={"num_threads": 32, "display_progress": True},
        )
        self.model = copro_model
        copro_model.save("./copro_model.json")
        return copro_model

    def train_MIPROv2(self, train_set, test_set):
        mipro_optimizer = MIPROv2(
            metric=self.metric, 
            prompt_model=self.lm, 
            num_candidates=10
        )
        mipro_model = mipro_optimizer.compile(
            self.model, 
            trainset=train_set, 
            valset=test_set,
            minibatch_size=24
        )
        self.model = mipro_model
        mipro_model.save("./mipro_model.json")
        return mipro_model

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        idx_h, idx_r = x.tolist()[0]
        h, r = self.idx_to_entity[idx_h], self.idx_to_relation[idx_r]
        pred = self.model(source=h, relation=r, target_entities=self.target_entities)
        scores = torch.zeros(len(self.target_entities))
        for item in pred.predictions:
            entity_idx = self.entity_to_idx.get(item.entity, -1)
            if entity_idx >= 0:
                scores[entity_idx] = item.score
        return scores

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        raise NotImplementedError("DSPy_RCL needs to implement it")
    
    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass that returns scores for all entities given head entity and relation.
        
        Args:
            x: Tensor of shape [batch_size, 2] containing (head_idx, relation_idx) pairs
            
        Returns:
            Tensor of shape [batch_size, num_entities] with scores for all entities
        """
        batch_size = x.size(0)
        num_entities = len(self.entity_to_idx)
        
        # Initialize tensor of correct shape with -Inf
        batch_output = torch.full((batch_size, num_entities), -np.Inf, dtype=torch.float)
        
        for i, item in enumerate(x.tolist()):
            idx_h, idx_r = item
            h, r = self.idx_to_entity[idx_h], self.idx_to_relation[idx_r]
            
            pred = self.model(
                source=h, relation=r, target_entities=self.target_entities
            )
            
            for pred_item in pred.predictions:
                entity_idx = self.entity_to_idx.get(pred_item.entity, -1)
                if entity_idx >= 0:
                    batch_output[i, entity_idx] = float(pred_item.score)
        
        return batch_output

    def metric(self, example: dspy.Example, pred: dspy.Prediction, trace=None):
        """
        Calculate MRR metric using the same filtering and ranking as evaluate_lp_k_vs_all.
        
        Args:
            example: DSPy example containing source, relation, and ground truth
            pred: DSPy prediction with predicted entities and scores
            
        Returns:
            MRR (Mean Reciprocal Rank) score
        """
        source = example.source
        relation = example.relation
        
        # Convert source and relation to indices
        id_e = self.entity_to_idx[source]
        id_r = self.relation_to_idx[relation]
        
        filt = self.kg.er_vocab[(id_e, id_r)]
        
        if hasattr(example, 'target'):
            ground_truth_entity = example.target
        else:
            ground_truth_entity = next((item.entity for item in example.predictions if item.score == 1.0), None)
        
        if not ground_truth_entity:
            return 0.0  # no correct target to evaluate
        
        id_e_target = self.entity_to_idx[ground_truth_entity]
        
        num_entities = len(self.entity_to_idx)
        predictions_tensor = torch.full((num_entities,), -np.Inf, dtype=torch.float)
        
        valid_predictions = 0
        for pred_item in pred.predictions:
            try:
                entity_idx = self.entity_to_idx[pred_item.entity]
                predictions_tensor[entity_idx] = float(pred_item.score)
                valid_predictions += 1
            except KeyError:
                continue
        
        if valid_predictions == 0:
            return 0.0
        
        target_value = predictions_tensor[id_e_target].item()
        
        # Filter predictions 
        predictions_tensor[filt] = -np.Inf
        
        # Restore target value
        predictions_tensor[id_e_target] = target_value
        
        # Sort values in descending order
        sort_values, sort_idxs = torch.sort(predictions_tensor, dim=0, descending=True)
        
        # Find rank (1-indexed)
        rank_positions = torch.where(sort_idxs == id_e_target)[0]
        if len(rank_positions) == 0:
            return 0.0
            
        rank = rank_positions[0].item() + 1
        
        # Compute MRR
        mrr = 1.0 / rank
        print("\n")
        print("mrr", mrr)
        return mrr

    def manual_evaluation(self, test_set):
        """
        Manually evaluate the model using the exact same triples as evaluate_lp_k_vs_all.
        
        Args:
            test_set: Numpy array of shape [num_triples, 3] containing (head_idx, relation_idx, tail_idx)
            
        Returns:
            Dictionary of evaluation metrics (H@1, H@3, H@10, MRR)
        """
        ranks = []
        hits_range = [1, 3, 10]
        hits = {i: 0 for i in hits_range}
        num_triples = len(test_set)
        
        # Use tqdm for progress bar
        for triple in tqdm(test_set, desc="Evaluating test triples", unit="triple"):
            id_e, id_r, id_e_target = triple
            
            # Convert indices to entity/relation names
            source = self.idx_to_entity[id_e]
            relation = self.idx_to_relation[id_r]
            target = self.idx_to_entity[id_e_target]
            
            # Get model predictions
            pred = self.model(
                source=source, relation=relation, target_entities=self.target_entities
            )
            
            # Create an example with the correct target entity
            example = dspy.Example(
                source=source,
                relation=relation,
                target=target  # Store target directly for metric function
            )
            
            # Calculate MRR
            rank_reciprocal = self.metric(example, pred)
            if rank_reciprocal > 0:
                rank = int(1.0 / rank_reciprocal)
                ranks.append(rank)
                
                # Calculate hits@k
                for k in hits_range:
                    if rank <= k:
                        hits[k] += 1
            else:
                # Handle edge case where metric returns 0
                ranks.append(float('inf'))
        
        # Calculate metrics in the same way as evaluate_lp_k_vs_all
        hit_1 = hits[1] / num_triples if num_triples > 0 else 0
        hit_3 = hits[3] / num_triples if num_triples > 0 else 0
        hit_10 = hits[10] / num_triples if num_triples > 0 else 0
        
        # Calculate MRR, handling infinite ranks
        finite_ranks = [r for r in ranks if r < float('inf')]
        mean_reciprocal_rank = np.mean(1.0 / np.array(finite_ranks)) if finite_ranks else 0.0
        
        results = {
            'H@1': hit_1,
            'H@3': hit_3,
            'H@10': hit_10,
            'MRR': mean_reciprocal_rank
        }
        
        return results

# Update the __main__ block to test both evaluation methods
if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Run DSPy_RCL model for link prediction")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file path for results")
    parser.add_argument("--base_url", type=str, default="http://harebell.cs.upb.de:8501/v1", 
                        help="Base URL for the LLM API")
    parser.add_argument("--api_key", type=str, default="xxx", 
                        help="API key for the LLM API")
    
    args = parser.parse_args()
    
    try:
        print(f"Loading knowledge graph from {args.dataset_dir}...")
        kg = KG(
            dataset_dir=args.dataset_dir,
            separator="\s+",
            eval_model="train_value_test",
            add_reciprocal=False,
        )
        
        print("Initializing DSPy_RCL model...")
        model = DSPy_RCL(
            knowledge_graph=kg,
            base_url=args.base_url,
            api_key=args.api_key,
        )

        # Generate examples
        print("Generating examples...")
        examples = model.generate_examples()
        train_examples, test_examples = model.generate_train_test_split(examples)
        
        print(f"Training MIPROv2 model on {len(train_examples)} examples...")
        model.train_MIPROv2(train_examples, test_examples)

        print(f"Evaluating model on {len(kg.test_set)} test triples...")
        eval_results = evaluate_lp_k_vs_all(
            model=model, 
            triple_idx=kg.test_set,  # Use the actual test set from the KG
            er_vocab=kg.er_vocab, 
            info=f'Eval KvsAll on {os.path.basename(args.dataset_dir)}', 
            batch_size=1
        )
        
        # Create comprehensive results dictionary
        results = {
            "dataset": os.path.basename(args.dataset_dir),
            "timestamp": datetime.now().isoformat(),
            "model": "DSPy_RCL",
            "metrics": {
                "MRR": eval_results.get("MRR", 0.0),
                "H@1": eval_results.get("H@1", 0.0),
                "H@3": eval_results.get("H@3", 0.0),
                "H@10": eval_results.get("H@10", 0.0)
            },
            "parameters": {
                "base_url": args.base_url
            }
        }
        
        # Save results to JSON file
        print(f"Saving results to {args.out}...")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results successfully saved to {args.out}")
        print(f"Metrics: MRR={results['metrics']['MRR']:.4f}, H@1={results['metrics']['H@1']:.4f}, H@3={results['metrics']['H@3']:.4f}, H@10={results['metrics']['H@10']:.4f}")
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        # Save error information to output file
        error_results = {
            "dataset": os.path.basename(args.dataset_dir) if args.dataset_dir else "unknown",
            "timestamp": datetime.now().isoformat(),
            "model": "DSPy_RCL",
            "error": str(e)
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump(error_results, f, indent=2)
        except Exception as write_error:
            print(f"Failed to write error to output file: {str(write_error)}", file=sys.stderr)
        sys.exit(1)