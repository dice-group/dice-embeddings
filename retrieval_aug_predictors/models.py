from pydantic import BaseModel, Field
from .abstract import AbstractBaseLinkPredictorClass
from typing import List, Tuple
from dicee.knowledge_graph import KG
import torch
from openai import OpenAI
import json
import re
import igraph
from typing import Tuple, Dict
import dspy
from tqdm import tqdm
from dspy.teleprompt import LabeledFewShot
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
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.llm_model = llm_model
        self.temperature = temperature
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

class LM_Call_Signature(dspy.Signature):
    source: str = dspy.InputField(description="The source entity")
    relation: str = dspy.InputField(description="The relation")
    target_entities: List[str] = dspy.InputField(description="The list of target entities")
    predictions: List[PredictionItem] = dspy.OutputField(description="The list of predicted entities with scores")

class DSPy_RCL(AbstractBaseLinkPredictorClass):

    def __init__(self, knowledge_graph: KG = None, base_url: str = None, api_key: str = None, llm_model: str = None,
                 temperature: float = 0.0, seed: int = 42, max_relation_examples: int = 2000, use_val: bool = True,
                 exclude_source: bool = False) -> None:
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
        self.lm = dspy.LM(model="openai/tentris", api_key=self.api_key, base_url=self.base_url)
        dspy.configure(lm=self.lm)
        self.model = dspy.ChainOfThought(LM_Call_Signature)

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
        self.triples = triples

        # Create a mapping from relation to all triples using that relation
        self.relation_to_triples = {}
        for s, p, o in triples:
            if p not in self.relation_to_triples:
                self.relation_to_triples[p] = []
            self.relation_to_triples[p].append((s, p, o))

        self.target_entities = list(sorted(self.entity_to_idx.keys()))

    def metric(self, example, pred, trace=None):
        # Calculate MRR
        mrr = 0
        for i, (h, r, t) in enumerate(example):
            # Check if the target entity is in the list of predicted entities
            if t in [p.entity for p in pred]:
                mrr += 1 / (i + 1)
        mrr /= len(example)
        return mrr

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
                prediction_items = [PredictionItem(entity=target, score=1.0) for target in targets]
                
                # Create a DSPy example with the input being the head entity and relation
                # and the output being all correct tail entities as PredictionItem objects
                example = dspy.Example(
                    source=source,
                    relation=relation,
                    target_entities=self.target_entities,
                    predictions=prediction_items
                ).with_inputs("source", "relation", "target_entities")
                examples.append(example)
        
        return examples
    
    def generate_train_test_split(self, examples, test_size=0.2):
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
        for example in tqdm(examples, desc="Evaluating examples", unit="ex", ncols=100, leave=True):
            # Extract the input values from the example
            source = example.source
            relation = example.relation
            target_entities = example.target_entities
            # Get model predictions
            pred = self.model(source=source, relation=relation, target_entities=target_entities)
            formatted_example = [(source, relation, item.entity) for item in example.predictions]
            score = self.metric(formatted_example, pred.predictions)
            total_score += score
        # Return the average score
        return total_score / len(examples) if examples else 0.0

    def train_labeledFewShot(self, train_set, few_shot_k): 
        lfs_optimizer = LabeledFewShot(k=few_shot_k)
        lfs_model = lfs_optimizer.compile(self.model, trainset=train_set)
        self.model = lfs_model
        lfs_model.save("./lfs_model.json")
        return lfs_model
    
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
            pred = self.model(source=h, relation=r, target_entities=self.target_entities)
            batch_output.append(pred.predictions)
        return torch.FloatTensor(batch_output)
    
    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        raise NotImplementedError("DSPy_RCL needs to implement it")

# test the dspy model -> remove later 
if __name__ == "__main__":
    kg = KG(dataset_dir="KGs/Countries-S1", separator="\s+", eval_model="train_value_test", add_reciprocal=False)
    model = DSPy_RCL(knowledge_graph=kg, base_url="http://harebell.cs.upb.de:8501/v1", api_key=":)")
    
    examples = model.generate_examples()
    train_examples, test_examples = model.generate_train_test_split(examples, test_size=0.2)
    
    # Train the model
    model.train_labeledFewShot(train_examples, few_shot_k=3)

    # eval model 
    print(model.manual_evaluation(test_examples))
    
