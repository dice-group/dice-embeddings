"""
python -m retrieval_aug_predictors.models.Demir --dataset_dir KGs/Countries-S1 --out "countries_s1_results.json" && cat countries_s1_results.json
{
    "H@1": 1.0,
    "H@3": 1.0,
    "H@10": 1.0,
    "MRR": 1.0
}

python -m retrieval_aug_predictors.models.Demir --dataset_dir KGs/Countries-S2 --out "countries_s2_results.json" && cat countries_s2_results.json
{
    "H@1": 0.625,
    "H@3": 0.9583333333333334,
    "H@10": 0.9583333333333334,
    "MRR": 0.7921296296296297
}
python -m retrieval_aug_predictors.models.Demir --dataset_dir KGs/Countries-S3 --out "countries_s3_results.json" && cat countries_s3_results.json
{
    "H@1": 0.7083333333333334,
    "H@3": 0.9583333333333334,
    "H@10": 0.9583333333333334,
    "MRR": 0.8337962962962964
}
"""

import dspy
import torch
import json
from typing import List, Tuple
from retrieval_aug_predictors.models import KG, AbstractBaseLinkPredictorClass
from retrieval_aug_predictors.arguments import parser
from retrieval_aug_predictors.utils import sanity_checking
from dicee.evaluator import evaluate_lp_k_vs_all
from dotenv import load_dotenv
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
        self.predictor = dspy.Predict(MultiLabelLinkPredictionWithScores)

    def forward(self, subject, predicate, few_shot_examples) -> List[Tuple[str, float]]:
        # Format examples more structured with clearer JSON expectations
        example_str = "Given a subject entity and relationship, predict all possible object entities with confidence scores.\n\n"
        example_str += "Examples:\n"

        # Add more context to each example
        for idx, ((s, p), o_list) in enumerate(few_shot_examples.items()):
            formatted_objects = [{"entity": o, "score": 1.0} for o in
                                 o_list]  # Assuming high confidence in training examples
            example_str += f"Example {idx + 1}:\n"
            example_str += f"Input: subject='{s}', predicate='{p}'\n"
            example_str += f"Output: {json.dumps(formatted_objects)}\n\n"

        # Add more explicit instructions for the model
        example_str += "Rules:\n"
        example_str += "1. Return all relevant entities with confidence scores between 0.0 and 1.0\n"
        example_str += "2. Higher scores indicate higher confidence\n"
        example_str += "3. Sort results by confidence score in descending order\n"
        example_str += "4. Use knowledge of real-world relationships when appropriate\n\n"

        # Log the prompt for analysis (optional, can be removed in production)
        if hasattr(self, 'debug') and self.debug:
            print(f"PROMPT:\n{example_str}\n")
            print(f"QUERY: subject='{subject}', predicate='{predicate}'")

        # Make the prediction
        dspy_pred = self.predictor(examples=example_str, subject=subject, predicate=predicate)

        try:
            results = json.loads(dspy_pred.objects_with_scores)
            # Sort by score descending
            return [(i["entity"], i["score"]) for i in sorted(results, key=lambda x: x["score"], reverse=True)]
        except json.JSONDecodeError:
            # Handle malformed JSON gracefully
            print(f"Warning: Received malformed JSON: {dspy_pred.objects_with_scores}")
            return []

class Demir(AbstractBaseLinkPredictorClass):
    def __init__(self,knowledge_graph, base_url,api_key,temperature, seed,llm_model,use_val:bool=False):
        super().__init__(knowledge_graph,name="Demir")
        self.temperature = temperature
        self.seed = seed
        self.lm = dspy.LM(model=f"openai/{llm_model}", api_key=api_key,
                          api_base=base_url,
                          seed=seed,
                          temperature=temperature,
                          cache=True,cache_in_memory=True)
        dspy.configure(lm=self.lm)
        self.train_set: List[Tuple[str]] = [(self.idx_to_entity[idx_h],
                                             self.idx_to_relation[idx_r],
                                             self.idx_to_entity[idx_t]) for idx_h, idx_r, idx_t in
                                            self.kg.train_set.tolist()]
        # Validation dataset
        self.val_set: List[Tuple[str]] = [(self.idx_to_entity[idx_h],
                                           self.idx_to_relation[idx_r],
                                           self.idx_to_entity[idx_t]) for idx_h, idx_r, idx_t in
                                          self.kg.valid_set.tolist()]
        self.triples = self.train_set + self.val_set if use_val else self.train_set

        self.entity_relation_to_entities=dict()
        for s,p,o in self.triples:
            self.entity_relation_to_entities.setdefault((s,p),[]).append(o)
        # @TODO: rename entity_relation_to_entities to something else
        # @TODO: given (s,p), we need to return relevant triples, not the whole graph.
        self.scoring_func = MultiLabelLinkPredictor()

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        raise NotImplementedError("RCL needs to implement it")
    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        batch_predictions=[]
        for hr in x.tolist():
            idx_h, idx_r = hr
            h, r = self.idx_to_entity[idx_h], self.idx_to_relation[idx_r]
            predictions = self.scoring_func.forward(
                subject=h,
                predicate=r,
                few_shot_examples=self.entity_relation_to_entities)
            scores=[-100]*len(self.idx_to_entity)
            for entity,score in predictions:
                try:
                    idx_entity=self.entity_to_idx[entity]
                except KeyError:
                    print(f"Entity:{entity} not found")
                    continue
                scores[idx_entity]=score
            batch_predictions.append(scores)
        return torch.FloatTensor(batch_predictions)


# test the dspy model -> remove later
if __name__ == "__main__":
    args=parser.parse_args()
    # Important: add_reciprocal=False in KvsAll implies that inverse relation has been introduced.
    # Therefore, The link prediction results are based on the missing tail rankings only!
    kg = KG(dataset_dir=args.dataset_dir, separator="\s+", eval_model=args.eval_model, add_reciprocal=False)
    sanity_checking(args,kg)
    model = Demir(knowledge_graph=kg, base_url=args.base_url, api_key=args.api_key, llm_model=args.llm_model_name, temperature=args.temperature, seed=args.seed)
    results:dict = evaluate_lp_k_vs_all(model=model, triple_idx=kg.test_set[:args.eval_size],
                         er_vocab=kg.er_vocab, info='Eval KvsAll Starts', batch_size=args.batch_size)
    if args.out and results:
        # Writing the dictionary to a JSON file
        print(results)
        with open(args.out, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        print(f"Results has been saved to {args.out}")