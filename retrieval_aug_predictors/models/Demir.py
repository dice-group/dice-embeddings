import dspy
import torch
import json
from typing import List, Tuple
from retrieval_aug_predictors.models import KG, AbstractBaseLinkPredictorClass
from openai import OpenAI

# 1. Define the Signature
class KGLikelihood(dspy.Signature):
    """Assess the likelihood that a triple (subject, predicate, candidate_object) is true,
    given some context triples. Output a score between 0.0 and 1.0."""

    context = dspy.InputField(desc="Known knowledge graph triples.")
    subject = dspy.InputField(desc="The subject entity.")
    predicate = dspy.InputField(desc="The relationship type.")
    candidate_object = dspy.InputField(desc="The candidate object entity to score.")

    score = dspy.OutputField(desc="A likelihood score between 0.0 and 1.0.")


class MultiLabelLinkPredictionWithScores(dspy.Signature):
    """Given a subject entity and a predicate (relation), predict a list of
    object entities that satisfy the relation, along with a likelihood score for each.
    Use the provided examples as a guide.
    Output a JSON formatted list of objects, where each object has an 'entity' (string)
    and a 'score' (float between 0.0 and 1.0) key."""

    examples = dspy.InputField(
        desc="Few-shot examples of (subject, predicate) -> [{'entity': entity1, 'score': score1}, ...].")
    subject = dspy.InputField(desc="The subject entity.")
    predicate = dspy.InputField(desc="The relationship type.")

    # Updated OutputField requesting JSON
    objects_with_scores = dspy.OutputField(
        desc="A JSON string representing a list of objects. "
             "Each object in the list should be a dictionary with 'entity' (string) and 'score' (float, 0.0-1.0) keys.")

class MultiLabelLinkPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(MultiLabelLinkPredictionWithScores)
    def forward(self, subject, predicate, few_shot_examples)->List[Tuple[str, float]]:
        example_str = ""
        for (s, p), o_list in few_shot_examples.items():
            example_str += f"({s}, {p})\n{', '.join(o_list)}\n---\n"
        # @TODO: CD: Also keep track of LLM cost
        dspy_pred:dspy.primitives.prediction.Prediction=self.predictor(examples=example_str, subject=subject, predicate=predicate)
        return [ (i["entity"],i["score"])for i in json.loads(dspy_pred.objects_with_scores)]

class Demir(AbstractBaseLinkPredictorClass):
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

    def __init__(self,knowledge_graph, base_url,api_key,temperature, seed,llm_model,use_val:bool=False):
        super().__init__(knowledge_graph,name="Demir")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.temperature = temperature
        self.seed = seed

        self.lm = dspy.LM(model=f"openai/{llm_model}", api_key=api_key,
                          api_base=base_url,
                          seed=seed,
                          temperature=temperature,
                          cache=True,cache_in_memory=True,
                          kwargs={"extra_body":{"truncate_prompt_tokens": 32_000}})
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
        from collections import OrderedDict
        for s,p,o in self.triples:
            self.entity_relation_to_entities.setdefault((s,p),[]).append(o)

        # 4. Instantiate your predictor
        self.scoring_func = MultiLabelLinkPredictor()
        self.entities:List[str]=list(sorted(self.entity_to_idx.keys()))
