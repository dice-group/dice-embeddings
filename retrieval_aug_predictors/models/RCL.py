import torch
import json
from typing import List, Tuple
from retrieval_aug_predictors.models import KG, AbstractBaseLinkPredictorClass
from retrieval_aug_predictors.schemas import PredictionResponse
from openai import OpenAI

class RCL(AbstractBaseLinkPredictorClass):
    """ Relation-based Context Learning to predict missing entities.

    (h, r, t) ∈ G_test

    1. Use all triples from G_train involving relation r to create context.
    2. Generate a prompt based on these triples and (h,r) to assign scores for all e ∈ E.
    """

    def __init__(self, knowledge_graph: KG = None, base_url: str = None, api_key: str = None, llm_model: str = None,
                 temperature: float = 0.0, seed: int = 42, max_relation_examples: int = 2000, use_val: bool = True,
                 exclude_source: bool = False) -> None:
        super().__init__(knowledge_graph, name="RCL")
        assert base_url is not None and isinstance(base_url, str)
        self.base_url = base_url
        self.api_key = api_key
        self.llm_model = llm_model
        self.temperature = temperature
        self.seed = seed
        self.max_relation_examples = max_relation_examples
        self.exclude_source = exclude_source
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        # Training dataset
        self.train_set: List[Tuple[str]] = [(self.idx_to_entity[idx_h],
                                             self.idx_to_relation[idx_r],
                                             self.idx_to_entity[idx_t]) for idx_h, idx_r, idx_t in
                                            self.kg.train_set.tolist()]
        # Validation dataset
        self.val_set: List[Tuple[str]] = [(self.idx_to_entity[idx_h],
                                           self.idx_to_relation[idx_r],
                                           self.idx_to_entity[idx_t]) for idx_h, idx_r, idx_t in
                                          self.kg.valid_set.tolist()]

        triples = self.train_set + self.val_set if use_val else self.train_set

        # Create a mapping from relation to all triples using that relation
        self.relation_to_triples = {}
        for s, p, o in triples:
            if p not in self.relation_to_triples:
                self.relation_to_triples[p] = []
            self.relation_to_triples[p].append((s, p, o))

        self.target_entities = list(sorted(self.entity_to_idx.keys()))

    def _create_prompt_based_on_relation(self, source: str, relation: str) -> str:
        # Get all triples with the current relation
        relation_triples = []
        if relation in self.relation_to_triples:
            relation_triples = self.relation_to_triples[relation]

            # Exclude triples where the source entity is the current one if flag is set
            if self.exclude_source:
                relation_triples = [triple for triple in relation_triples if triple[0] != source]

            # Limit examples if too many
            if len(relation_triples) > self.max_relation_examples:
                relation_triples = relation_triples[:self.max_relation_examples]

        relation_context = "Examples of how the relation is used in the knowledge base:\n"
        for s, p, o in sorted(relation_triples):
            relation_context += f"- {s} {p} {o}\n"
        relation_context += "\n"

        prompt = f"""
    Task:
    Predict the most likely target entities for the query using the provided knowledge base examples.

    Input:
    - Source Entity: {source}
    - Relation: {relation}
    - Query: ({source}, {relation}, ?)

    Knowledge Base Examples:
    {relation_context}

    Instructions:
    1. From the provided list of target entities: {self.target_entities}, select up to {min(len(self.target_entities), 15)} plausible targets.
    2. Rank these entities in order of likelihood for being the correct target.
    3. Assign each entity a likelihood score as a floating point number between 0 and 1.
    4. Consider geographic factors (e.g., location, regional classifications, political associations) when applicable.
    5. Only include entities from the provided list that are plausible for the given relation.
    6. If no entity seems plausible, return an empty list.

    Output:
    Return a valid JSON object **only** in the following format (without any additional text) and float_number ∈ [0,1]:

    {{
    "predictions": [
        {{"entity": "entity_name", "score": float_number}},
        ...
    ]
    }}
    """
        return prompt

    def forward_triples(self, x: torch.LongTensor):
        raise NotImplementedError("RCL needs to implement it")

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        batch_output = []
        # Iterate over batch of subject and relation pairs
        for i in x.tolist():
            # index of an entity and index of a relation.
            idx_h, idx_r = i
            # String representations of an entity and a relation, respectively.
            h, r = self.idx_to_entity[idx_h], self.idx_to_relation[idx_r]
            llm_response = self.client.chat.completions.create(
                model=self.llm_model, temperature=self.temperature, seed=self.seed,
                messages=[{"role": "user",
                           "content": "You are a knowledgeable assistant that helps with link prediction tasks.\n" +
                                      self._create_prompt_based_on_relation(source=h, relation=r)}],
                extra_body={"guided_json": PredictionResponse.model_json_schema(),
                            "truncate_prompt_tokens": 30_000,
                            }).choices[0].message.content

            prediction_response = PredictionResponse(**json.loads(llm_response))
            # Initialize scores for all entities
            scores_for_all_entities = [-1.0 for _ in range(len(self.idx_to_entity))]
            for pred in prediction_response.predictions:
                try:
                    scores_for_all_entities[self.entity_to_idx[pred.entity]] = pred.score
                except KeyError:
                    print(f"For {h},{r}, {pred} not found\tPrediction Size: {len(prediction_response.predictions)}")
                    continue
            batch_output.append(scores_for_all_entities)
        return torch.FloatTensor(batch_output)
