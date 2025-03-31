from pydantic import BaseModel, Field
from .abstract import AbstractBaseLinkPredictorClass
from typing import List, Tuple
from dicee.knowledge_graph import KG
import torch
from openai import OpenAI
import json
import re
import igraph
class PredictionItem(BaseModel):
    """Individual prediction item with entity name and confidence score."""
    entity: str = Field(..., description="Name of the predicted entity")
    score: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)

class PredictionResponse(BaseModel):
    """Response model containing a list of entity predictions."""
    predictions: List[PredictionItem] = Field(..., description="List of predicted entities with scores")

class GCL(AbstractBaseLinkPredictorClass):
    """ in-context Learning on neighbouring triples to predict missing entities.

    (h, r, t) \in G_test

    1. Get all nodes that are n=3 hop around h.
    2. Get all triples from G_train involving (1).
    3. Generate a prompt based on (2) and (h,r) to assign scores for all e \in E.

    @TODO:CD: We should write a regression test on the Countries S1 dataset.
    @TODO:CD: We should ensure that the input tokens do not exceed the allowed limit.

    """

    def __init__(self, knowledge_graph: KG = None, base_url: str = None, api_key: str = None, llm_model: str = None,
                 temperature: float = 0.0, seed: int = 42, num_of_hops: int = 3, use_val: bool = True) -> None:
        super().__init__(knowledge_graph, name="GCL")
        # @TODO: CD: input arguments should be passed onto the abstract class
        assert base_url is not None and isinstance(base_url, str)
        self.base_url = base_url
        self.api_key = api_key
        self.llm_model = llm_model
        self.temperature = temperature
        self.seed = seed
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
        self.igraph = self.build_igraph(triples)
        self.str_entity_to_igraph_vertice = {i["name"]: i for i in self.igraph.vs}
        self.str_rel_to_igraph_edges = {i["label"]: i for i in self.igraph.es}

        # Mapping from an entity to relevant triples.
        # A relevant triple contains a entity that is num_of_hops around of a given entity
        self.node_to_relevant_triples = dict()
        for entity, entity_node_object in self.str_entity_to_igraph_vertice.items():
            neighboring_nodes = self.igraph.neighborhood(entity_node_object, order=num_of_hops)
            subgraph = self.igraph.subgraph(neighboring_nodes)
            self.node_to_relevant_triples[entity] = {
                (subgraph.vs[edge.source]["name"], edge["label"], subgraph.vs[edge.target]["name"]) for edge in
                subgraph.es}

        self.target_entities = list(sorted(self.entity_to_idx.keys()))

    def _create_prompt_based_on_neighbours(self, source: str, relation: str) -> str:
        # Get relevant triples for the source entity
        relevant_triples = []
        if source in self.node_to_relevant_triples:
            relevant_triples = list(self.node_to_relevant_triples[source])

        assert len(relevant_triples) > 0
        # @TODO:CD:Potential improvement by trade offing the test runtime:
        #  @TODO: Finding an some triples from relevant_triples while the prediction is being invariant to it
        # @TODO: Prediction does not change but the input size decreases
        # @TODO: The removed triples can be seen as noise
        triples_context = "Here are some known facts about the source entity that might be relevant:\n"
        for s, p, o in sorted(relevant_triples):
            triples_context += f"- {s} {p} {o}\n"
        triples_context += "\n"

        # Important: Grouping relations is important to reach MRR 1.0
        similar_relations = []
        for s, p, o in relevant_triples:
            if p == relation and s != source:
                similar_relations.append((s, p, o))

        similar_relations_context = "Here are examples of similar relations in the knowledge base:\n"
        for s, p, o in similar_relations:
            similar_relations_context += f"- {s} {p} {o}\n"
        similar_relations_context += "\n"

        base_prompt = f"""
        I'm trying to predict the most likely target entities for the following query:
        Source entity: {source}
        Relation: {relation}
        Query: ({source}, {relation}, ?)

        Subgraph Graph:
        {triples_context}
        {similar_relations_context}

        Please provide a ranked list of at most {min(len(self.target_entities), 15)} likely target entities from the following list, along with likelihoods for each: {self.target_entities}

        Provide your answer in the following JSON format: {{"predictions": [{{"entity": "entity_name", "score": float_number}}]}}

        Notes:
        1. Use the provided knowledge about the source entity and similar relations to inform your predictions.       
        1. Only include entities that are plausible targets for this relation.
        2. For geographic entities, consider geographic location, regional classifications, and political associations.
        3. Rank the entities by likelihood of being the correct target.
        4. ONLY INCLUDE entities from the provided list in your predictions.
        5. If certain entities are not suitable for this relation, don't include them.
        6. Return a valid JSON output.
        7. Make sure scores are floating point numbers between 0 and 1, not strings.
        8. A score can only be between 0 and 1, i.e. score ∈ [0, 1]. They can never be negative or greater than 1!
        """
        return base_prompt

    @staticmethod
    def build_igraph(graph: List[Tuple[str, str, str]]):
        ig_graph = igraph.Graph(directed=True)
        # Extract unique vertices from all quadruples
        vertices = set()
        edges = []
        labels = []
        for s, p, o in graph:
            vertices.add(s)
            vertices.add(o)
            # ORDER MATTERS!
            edges.append((s, o))
            labels.append(p)

        # Add all unique vertices at once
        ig_graph.add_vertices(list(vertices))
        # Add edges with labels
        ig_graph.add_edges(edges)
        ig_graph.es["label"] = labels
        # Validate edge count
        assert len(edges) == len(ig_graph.es), "Edge mismatch after graph construction!"
        extracted_triples = [(ig_graph.vs[edge.source]["name"], edge["label"], ig_graph.vs[edge.target]["name"]) for
                             edge in ig_graph.es]
        # Not only the number but even the order must match
        assert extracted_triples == [triple for triple in graph]
        return ig_graph

    def forward_triples(self, x: torch.LongTensor):
        raise NotImplementedError("GraphContextLearner needs to implement it")

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
                                      self._create_prompt_based_on_neighbours(source=h, relation=r)}],
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

class RALP(AbstractBaseLinkPredictorClass):
    def __init__(self, knowledge_graph: KG = None,
                 name="ralp-1.0",
                 base_url="http://tentris-ml.cs.upb.de:8501/v1",
                 api_key=None,
                 llm_model="tentris",
                 temperature: float = 1, seed: int = 42) -> None:
        super().__init__(knowledge_graph, name)
        # @TODO: CD: input arguments should be passed onto the abstract class

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.llm_model = llm_model
        self.temperature = temperature
        # @TODO:CD: Use the seed
        self.seed = seed

    def extract_float(self, text):
        """Extract the float number from a string. Used mainly to filter the LLM-output for the scoring task."""
        pattern = r"-?\d*\.\d+|-?\d+\.\d*"
        match = re.search(pattern, text)
        return float(match.group()) if match else 0.0

    def ru(self, entity):
        """Remove underscore from the entity (as str)."""
        return entity.replace("_", " ")

    def get_score(self, triple: tuple, triples_h: str) -> float:
        system_prompt = """You are an expert in knowledge graphs and link prediction. Your task is to assign a plausibility score (from 0 to 1) to a given triple (subject, predicate, object) based on a set of known training triples for the same subject. 

        - A score of 1.0 means the triple is highly likely to be true.  
        - A score of 0.0 means the triple is highly unlikely to be true.  
        - Intermediate values (e.g., 0.4, 0.7) reflect varying levels of plausibility.


        **Guidelines for scoring:**
        1. **Exact Match:** If the triple already exists in the training set or if the facts clearly state that the triple must be true assign a score close to 1.0.
        2. **Pattern Matching:** If the predicate-object pair frequently occurs for the given subject, assign a high score.
        3. **Semantic Similarity:** If the object is semantically close to known objects for the subject-predicate pair, assign a moderate to high score.
        4. **Rare or Unseen Combinations:** If the triple does not follow the learned patterns, assign a low score.
        5. **Contradictions:** If the triple contradicts existing facts (perform your own reasoning), assign a very low score.

        You must analyze the given triple and the training triples, apply the reasoning above, and output only a single **floating-point score** between **0.0 and 1.0**, without any explanation or additional text.
        Do not depend only on triples provided to you, also use your own knowledge as an AI assistant to reason about the truthness of the given triple as a fact.
        You are strictly required to provide only the score as an answer and do not explain it."""

        user_prompt = f"""Here is the triple we want to evaluate:
        (subject: {triple[0]}, predicate: {triple[1]}, object: {triple[2]})

        Here are the known training triples for the subject "{triple[0]}":
        {triples_h}

        Assign a score to the given triple based on the provided training triples.
        """
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            seed=42,
            temperature=self.temperature
        )

        # Extract the response content
        content = response.choices[0].message.content
        return self.extract_float(content)

    def forward_k_vs_all(self, x):
        raise NotImplementedError("RALP needs to implement it")

    def forward_triples(self, indexed_triples: torch.LongTensor):
        n, d = indexed_triples.shape
        # For the time being
        assert d == 3
        assert n == 1
        scores = []
        for triple in indexed_triples.tolist():
            idx_h, idx_r, idx_t = triple
            h, r, t = self.idx_to_entity[idx_h], self.idx_to_relation[idx_r], self.idx_to_entity[idx_t]

            # Retrieve triples where 'h' is a subject or an object
            triples_h = [trp for trp in self.kg.train_set if (trp[0] == idx_h or trp[2] == idx_h)]

            # Format the triples into structured string output that will be used in the prompt.
            triples_h_str = ""
            for trp in triples_h:
                triples_h_str += f'- ("{self.ru(self.idx_to_entity[trp[0]])}", "{self.ru(self.idx_to_relation[trp[1]])}", "{self.ru(self.idx_to_entity[trp[2]])}") \n'

            # Get the score from the LLM
            score = self.get_score((h, r, t), triples_h_str)
            scores.append([score])
        return torch.FloatTensor(scores)

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