import igraph
import torch
import json
from typing import List, Tuple
from dicee.knowledge_graph import KG
from ..abstract import AbstractBaseLinkPredictorClass
from ..schemas import PredictionResponse
from openai import OpenAI

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
