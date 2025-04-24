"""
python -m retrieval_aug_predictors.models.RALP --dataset_dir KGs/Countries-S1 --out "countries_s1_results.json" && cat countries_s1_results.json
{
    "H@1": 1.0,
    "H@3": 1.0,
    "H@10": 1.0,
    "MRR": 1.0
}
python -m retrieval_aug_predictors.models.RALP --dataset_dir KGs/Countries-S2 --out "countries_s2_results.json" && cat countries_s2_results.json
{
    "H@1": 1.0,
    "H@3": 1.0,
    "H@10": 1.0,
    "MRR": 1.0
}
python -m retrieval_aug_predictors.models.RALP --dataset_dir KGs/Countries-S3 --out "countries_s3_results.json" && cat countries_s3_results.json
{
    "H@1": 0.9166666666666666,
    "H@3": 1.0,
    "H@10": 1.0,
    "MRR": 0.951388888888889
}
"""
import time
import dspy
import torch
import json
from typing import List, Tuple, Callable, Protocol, Any
from retrieval_aug_predictors.models import KG, AbstractBaseLinkPredictorClass
from retrieval_aug_predictors.arguments import parser
from retrieval_aug_predictors.utils import sanity_checking
from dicee.evaluator import evaluate_lp_k_vs_all
from dotenv import load_dotenv
import tiktoken
load_dotenv()

class MultiLabelLinkPredictionWithScores(dspy.Signature):
    examples = dspy.InputField(
        desc="Few-shot examples of (subject, predicate) -> [{'entity': entity1, 'score': score1}, ...].")
    subject:str = dspy.InputField(desc="The subject entity.")
    predicate:str = dspy.InputField(desc="The relationship type.")

    # Updated OutputField requesting JSON
    objects_with_scores = dspy.OutputField(
        desc="A JSON string representing a list of objects. "
             "Each object in the list should be a dictionary with 'entity' (string) and 'score' (float, 0.0-1.0) keys.")


class MultiLabelLinkPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(MultiLabelLinkPredictionWithScores)

    def forward(self, subject, predicate, few_shot_examples) -> List[Tuple[str, float]]:
        # () Format examples more structured with clearer JSON expectations
        example_str = "Given a subject entity and predicate, predict all possible object entities with confidence scores.\n\n"
        example_str += "Examples:\n"
        # few_shot_examples = self._sample_examples_m2(subject, predicate, few_shot_examples)
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # () Add more context to each example
        for idx, ((s, p), o_list) in enumerate(few_shot_examples.items()):
            formatted_objects = [{"entity": o, "score": 1.0} for o in
                                 o_list]  # Assuming high confidence in training examples
            example_str += f"Example {idx + 1}:\n"
            example_str += f"Input: subject='{s}', predicate='{p}'\n"
            example_str += f"Output: {json.dumps(formatted_objects)}\n\n"
            if len(encoding.encode(example_str)) > 25000:
                break

        # () Add more explicit instructions for the model
        example_str += "Rules:\n"
        example_str += "1. Return all relevant entities with confidence scores between 0.0 and 1.0\n"
        example_str += "2. Higher scores indicate higher confidence\n"
        example_str += "3. Sort results by confidence score in descending order\n"
        example_str += "4. Use knowledge of real-world relationships when appropriate\n\n"

        # () Log the prompt for analysis (optional, can be removed in production)
        if hasattr(self, 'debug') and self.debug:
            print(f"PROMPT:\n{example_str}\n")
            print(f"QUERY: subject='{subject}', predicate='{predicate}'")

        # print(f"{len(encoding.encode(example_str))}: {subject}\t{predicate}")
        # () Make the prediction
        dspy_pred = self.predictor(examples=example_str, subject=subject, predicate=predicate)

        try:
            results = json.loads(dspy_pred.objects_with_scores)
            # () Sort by score descending
            return [(i["entity"], i["score"]) for i in sorted(results, key=lambda x: x["score"], reverse=True)]
        except json.JSONDecodeError:
            # () Handle malformed JSON gracefully
            print(f"Warning: Received malformed JSON: {dspy_pred.objects_with_scores}")
            return []

    def _sample_examples_m1(self, subject, predicate, few_shot_examples):
        """Sample triples that goes into the few shot examples.
        Results includes:
           1. triples where 'subject' is the subject,
           2. triples where o is the subject for o \in the list of objects for the given subject-predicate pair,
           3. every other triple where o is the object."""
        sampled_few_shot_examples = dict()
        # () get all triples where the subject is 'subject', undependable of the predicate
        items_for_s = {key: value for key, value in few_shot_examples.items() if key[0] == subject}
        sampled_few_shot_examples.update(items_for_s)
        # for i in range(k):
        for o in few_shot_examples.get((subject, predicate), []):
            # (1) where o is a subject
            items_i = {key: value for key, value in few_shot_examples.items() if key[0] == o}
            sampled_few_shot_examples.update(items_i)
            # (2) where o is an object
            items2_i = {key: value for key, value in few_shot_examples.items() if o in value}
            # (3) if not already added, add only the triple that contains o (what may be worthy for us)
            for key, value in items2_i.items():
                if key not in few_shot_examples:
                    sampled_few_shot_examples[key] = [o]

        return sampled_few_shot_examples


class RALP(AbstractBaseLinkPredictorClass):
    """Retrieval-Augmented Link Prediction (ex DemirEnsemble).
    Ensemble approach combining multiple prediction strategies"""

    def __init__(self, knowledge_graph, base_url, api_key, temperature, seed, llm_model, use_val: bool = False):
        super().__init__(knowledge_graph, name="RALP")
        self.temperature = temperature
        self.seed = seed
        # () Initialize ensemble.
        self.ensemble=[]
        for i in range(0, 9):
            temperature_coefficient=i*0.1
            self.ensemble.append(dspy.LM(model=f"openai/{llm_model}", api_key=api_key,
                    api_base=base_url, seed=seed, temperature=temperature_coefficient,
                    cache=True, cache_in_memory=True))
        # Initialize data same as original
        self.train_set = [(self.idx_to_entity[idx_h],
                           self.idx_to_relation[idx_r],
                           self.idx_to_entity[idx_t]) for idx_h, idx_r, idx_t in
                          self.kg.train_set.tolist()]

        self.val_set = [(self.idx_to_entity[idx_h],
                         self.idx_to_relation[idx_r],
                         self.idx_to_entity[idx_t]) for idx_h, idx_r, idx_t in
                        self.kg.valid_set.tolist()]

        self.triples = self.train_set + self.val_set if use_val else self.train_set

        # Create more sophisticated knowledge structures
        self.entity_relation_to_entities = dict()
        for s, p, o in self.triples:
            self.entity_relation_to_entities.setdefault((s, p), []).append(o)

        # Create models for ensemble
        self.predictors = self._create_ensemble_predictors()

        # Add pattern-based predictor for common relationship types
        self.pattern_predictor = self._create_pattern_predictor()

        # Add statistical predictor
        self.statistical_predictor = self._create_statistical_predictor()

    def _create_ensemble_predictors(self):
        """Create multiple predictors with different configurations"""
        predictors = []
        for i in self.ensemble:
            # Standard predictor
            dspy.configure(lm=i)
            predictors.append(MultiLabelLinkPredictor())
        return predictors

    def _create_pattern_predictor(self):
        """Create a pattern-based predictor for common relationship types"""
        patterns = {}

        # Analyze patterns in data
        for (s, p), o_list in self.entity_relation_to_entities.items():
            if p not in patterns:
                patterns[p] = {}

            # Look for patterns in object lists
            if len(o_list) >= 3:  # Need enough examples to infer patterns
                # Example pattern: location hierarchy
                if all(o.startswith("location") for o in o_list):
                    patterns[p]["type"] = "location_hierarchy"
                # More patterns can be added here

        return patterns

    def _create_statistical_predictor(self):
        """Create statistical predictors based on co-occurrence"""
        stats = {}

        # For each relation, calculate entity co-occurrence
        for (s, p), o_list in self.entity_relation_to_entities.items():
            if p not in stats:
                stats[p] = {"co_occurrence": {}}

            # Count co-occurrences of entities for this relation
            for o in o_list:
                if o not in stats[p]["co_occurrence"]:
                    stats[p]["co_occurrence"][o] = 0
                stats[p]["co_occurrence"][o] += 1

        return stats

    class ProcessingType(Protocol):
        """Specifies the type of the callable argument for `process_scores` method"""
        def __call__(self, x: List[float], y: List, **kwargs: Any) -> None: ...

    def process_scores(self, x: torch.LongTensor, process: ProcessingType,
                       storage: List = None) -> List:
        """Run the predictors to predict tail entities, score the prediction and
        further process the scores depending on the `process` argument.

        Args:
            x (torch.LongTensor): Batch of (h, r) indices to predict triples for.
            process (ProcessingType): Callable function to further process the scores.
            storage (List): List to store the processed scores.

        Returns:
            List: a list of processed scores.
        """
        seen_set = set()
        if storage is None:
            storage = []
        for hr in x.tolist():
            idx_h, idx_r = hr
            if (idx_h, idx_r) in seen_set:
                continue
            else:
                seen_set.add((idx_h, idx_r))

            h, r = self.idx_to_entity[idx_h], self.idx_to_relation[idx_r]

            # Get predictions from multiple models
            ensemble_predictions = []
            for predictor in self.predictors:
                dspy_pred = predictor.forward(
                    subject=h,
                    predicate=r,
                    few_shot_examples=self.entity_relation_to_entities)
                ensemble_predictions.append(dspy_pred)

            # Initialize scores vector
            scores = [0.0] * len(self.idx_to_entity)

            # Combine scores from ensemble
            for predictions in ensemble_predictions:
                for entity, score in predictions:
                    try:
                        idx_entity = self.entity_to_idx[entity]
                        # Accumulate scores from different predictors
                        scores[idx_entity] += float(score)
                    except (KeyError, ValueError):
                        continue

            # Add statistical bias based on co-occurrence
            if r in self.statistical_predictor:
                for entity, count in self.statistical_predictor[r]["co_occurrence"].items():
                    if entity in self.entity_to_idx:
                        idx_entity = self.entity_to_idx[entity]
                        # Add a small bias based on frequency
                        scores[idx_entity] += 0.1 * (
                                    count / (1 + max(self.statistical_predictor[r]["co_occurrence"].values())))

            # Normalize scores to [0, 1]
            max_score = max(scores)
            if max_score > 0:
                scores = [s / max_score for s in scores]

            process(scores, storage, h=h, r=r)

        return storage

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:

        def process(scores, storage, **kwargs):
            storage.append(scores)

        batch_predictions = self.process_scores(x, process)

        return torch.FloatTensor(batch_predictions)

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        raise NotImplementedError("RCL needs to implement it")

    def get_predicted_triples(self, x: torch.LongTensor, top_k: int = 1):
        """
        Retrieve the top predicted triples for the given batch of (h, r) indices.
        Args:
            x (torch.LongTensor): Batch of (h, r) indices to predict triples for.
            top_k (int): Number of top entities to return per (h, r).
        Returns:
            List[Tuple[str, str, List[Tuple[str, float]]]]:
                A list of (subject, relation, [(entity, score)]), where the scores are
                sorted to get the top_k predicted triples.
        """

        def process(scores, storage, **kwargs):
            # Extract top-k entities (along with scores)
                rng = range(len(scores))
                top_entity_indices = sorted(rng, key=lambda i: scores[i], reverse=True)[:top_k]
                top_entities = [(self.idx_to_entity[idx], scores[idx]) for idx in top_entity_indices]

                # Append formatted triple: (subject, predicate, [(entity, score)])
                storage.append((kwargs.get('h'), kwargs.get('r'), top_entities))

        predicted_triples = self.process_scores(x, process)

        return predicted_triples


# test the dspy model -> remove later
if __name__ == "__main__":
    args=parser.parse_args()
    # Important: add_reciprocal=False in KvsAll implies that inverse relation has been introduced.
    # Therefore, The link prediction results are based on the missing tail rankings only!
    kg = KG(dataset_dir=args.dataset_dir, separator="\s+", eval_model=args.eval_model, add_reciprocal=False)
    sanity_checking(args,kg)
    model = RALP(knowledge_graph=kg, base_url=args.base_url, api_key=args.api_key,
                              llm_model=args.llm_model_name, temperature=args.temperature, seed=args.seed)
    start = time.time()
    if not args.print_top_predictions:
        results:dict = evaluate_lp_k_vs_all(model=model, triple_idx=kg.test_set[:args.eval_size],
                             er_vocab=kg.er_vocab, info='Eval KvsAll Starts', batch_size=args.batch_size)
    else:
        x = kg.train_set[:, [0, 1]]
        results = model.get_predicted_triples(x, args.k)
    elapsed_time = time.time() - start
    print("Results: {}".format(results))
    print(f"Task completed successfully in {elapsed_time:.2f} seconds")
    if args.out and results:
        # Writing the dictionary to a JSON file
        with open(args.out, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        print(f"Results has been saved to {args.out}")