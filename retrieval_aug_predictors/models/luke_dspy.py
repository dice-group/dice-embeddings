import dspy
from dspy.teleprompt import * 
from typing import List, Tuple

from tqdm import tqdm
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
            model="openai/tentris", api_key=self.api_key, base_url=self.base_url
        )
        dspy.configure(lm=self.lm)
        self.model = dspy.ChainOfThought(LM_Call_Signature)

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

        # Create a mapping from relation to all triples using that relation
        self.relation_to_triples = {}
        for s, p, o in triples:
            if p not in self.relation_to_triples:
                self.relation_to_triples[p] = []
            self.relation_to_triples[p].append((s, p, o))

        self.target_entities = list(sorted(self.entity_to_idx.keys()))

    def metric(self, example: dspy.Example, pred: dspy.Prediction, trace=None):
        """
        Calculate Mean Reciprocal Rank (MRR) for the predictions.

        Args:
            example (dspy.Example): The input example containing the ground truth .predictions attribute.
            pred (dspy.Prediction): The model's prediction object containing the .predictions attribute.
            trace: Optional trace information

        Returns:
            float: The MRR score
        """
        try:
            ground_truth_items = example.predictions
        except AttributeError:
            print(
                "Warning: Could not find 'predictions' attribute in ground truth example. Trying 'output'."
            )
            return 0.0

        if not isinstance(ground_truth_items, (list, tuple)) or not all(
            hasattr(item, "entity") for item in ground_truth_items
        ):
            print(
                f"Warning: Unexpected ground truth format: {ground_truth_items}. Expected list of items with 'entity' attribute."
            )
            if isinstance(ground_truth_items, (list, tuple)) and all(
                isinstance(item, str) for item in ground_truth_items
            ):
                ground_truth_entities = ground_truth_items
            else:
                return 0.0
        else:
            ground_truth_entities = [item.entity for item in ground_truth_items]

        # Ensure pred has the predictions attribute and it's a list
        if not hasattr(pred, "predictions") or not isinstance(pred.predictions, list):
            print(
                f"Warning: Prediction object 'pred' lacks a valid 'predictions' list attribute: {pred}"
            )
            return 0.0

        # Sort predictions by score in descending order
        valid_predictions = [
            p for p in pred.predictions if hasattr(p, "score") and hasattr(p, "entity")
        ]
        if len(valid_predictions) != len(pred.predictions):
            print(
                f"Warning: Some prediction items lack 'score' or 'entity'. Original: {len(pred.predictions)}, Valid: {len(valid_predictions)}"
            )

        sorted_predictions = sorted(
            valid_predictions, key=lambda x: x.score, reverse=True
        )

        ranks = []
        found_entities = set()
        for gt_entity in ground_truth_entities:
            if gt_entity in found_entities:
                continue
            for i, pred_item in enumerate(sorted_predictions):
                if pred_item.entity == gt_entity:
                    ranks.append(i + 1)
                    found_entities.add(gt_entity)
                    break

        if not ranks:
            return 0.0

        mrr = sum(1.0 / rank for rank in ranks) / len(ground_truth_entities)
        return mrr
    
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
        Generate DSPy examples for training the model.

        Returns:
            List[dspy.Example]: A list of DSPy examples for training.
        """
        examples = []

        # Iterate through each relation
        for relation, triples in self.relation_to_triples.items():
            # Group triples by head entity
            head_to_tails = {}
            for s, p, o in triples:
                if s not in head_to_tails:
                    head_to_tails[s] = []
                head_to_tails[s].append(o)

            # Create examples for each head entity
            for source, targets in head_to_tails.items():
                # Convert target entities to PredictionItem objects with score 1.0
                prediction_items = [
                    PredictionItem(entity=target, score=1.0) for target in targets
                ]

                # Create a DSPy example with the input being the head entity and relation
                # and the output being all correct tail entities as PredictionItem objects
                example = dspy.Example(
                    source=source,
                    relation=relation,
                    target_entities=targets,
                    predictions=prediction_items,
                ).with_inputs("source", "relation", "target_entities")
                examples.append(example)

        return examples

    def generate_train_test_split(self, examples, test_size=0.8):
        """
        Split the examples into training and testing sets.

        Args:
            examples (List[dspy.Example]): A list of DSPy examples to split.
            test_size (float): The proportion of examples to include in the test set.

        Returns:
            Tuple[List[dspy.Example], List[dspy.Example]]: A tuple containing the training and testing examples.
        """
        import random

        random.seed(self.seed)

        # Shuffle the examples
        shuffled_examples = examples.copy()
        random.shuffle(shuffled_examples)

        # Calculate the split point
        split_idx = int(len(shuffled_examples) * (1 - test_size))

        # Split the examples
        train_examples = shuffled_examples[:split_idx]
        test_examples = shuffled_examples[split_idx:]

        return train_examples, test_examples

    def manual_evaluation(self, examples):
        """
        Manually evaluate the model on a list of examples using the metric method.

        Args:
            examples (List[dspy.Example]): A list of DSPy examples to evaluate.

        Returns:
            float: The average metric score across all examples.
        """
        total_score = 0.0
        num_evaluated = 0
        # Use tqdm for progress bar
        for example in tqdm(
            examples, desc="Evaluating examples", unit="ex", ncols=100, leave=True
        ):
            # Extract the input values from the example
            # These keys must match the input fields defined in your Signature
            # and present in the dspy.Example object.
            try:
                source = example.source
                relation = example.relation
                target_entities = example.target_entities
            except AttributeError as e:
                print(
                    f"Skipping example due to missing attribute: {e}. Example: {example}"
                )
                continue

            # Get model predictions (this should return a dspy.Prediction object)
            pred = self.model(
                source=source, relation=relation, target_entities=target_entities
            )

            # Calculate score using the metric function, passing the original example (ground truth)
            # and the prediction object.
            score = self.f1_score_metric(example, pred)
            total_score += score
            num_evaluated += 1

        # Return the average score
        return total_score / num_evaluated if num_evaluated else 0.0

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
            metric=self.f1_score_metric, prompt_model=self.lm, num_candidates=10
        )
        mipro_model = mipro_optimizer.compile(
            self.model, trainset=train_set, valset=test_set
        )
        self.model = mipro_model
        mipro_model.save("./mipro_model.json")
        return mipro_model

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        idx_h, idx_r = x.tolist()[0]
        h, r = self.idx_to_entity[idx_h], self.idx_to_relation[idx_r]
        pred = self.model(source=h, relation=r, target_entities=self.target_entities)
        return pred.predictions

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        batch_output = []
        for i in x.tolist():
            idx_h, idx_r = i
            h, r = self.idx_to_entity[idx_h], self.idx_to_relation[idx_r]
            pred = self.model(
                source=h, relation=r, target_entities=self.target_entities
            )
            batch_output.append(pred.predictions)
        return torch.FloatTensor(batch_output)

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        raise NotImplementedError("DSPy_RCL needs to implement it")

# test the dspy model -> remove later
if __name__ == "__main__":
    kg = KG(
        dataset_dir="KGs/Countries-S1",
        separator="\s+",
        eval_model="train_value_test",
        add_reciprocal=False,
    )
    model = DSPy_RCL(
        knowledge_graph=kg,
        base_url="http://harebell.cs.upb.de:8501/v1",
        api_key="token-tentris-upb",
    )

    examples = model.generate_examples()
    train_examples, test_examples = model.generate_train_test_split(
        examples, test_size=0.8
    )

    # Train the model
    model.train_MIPROv2(train_examples, test_examples)

    # eval model
    print(model.manual_evaluation(test_examples))