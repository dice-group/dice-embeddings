"""
Additional dependencies
pip install openai==1.66.3
pip install igraph==0.11.8
pip install jellyfish==1.1.3

# @TODO:CD:@Luke I guess writing few regression tests would help us to ensure that our modifications would not break the model
python retrieval_augmented_link_predictor.py --dataset_dir "KGs/Countries-S1" --model "GCL" --base_url "http://harebell.cs.upb.de:8501/v1" --num_of_hops 1
@TODO: CD: There is some randomness on this setup. I dunno whay
{'H@1': 0.7916666666666666, 'H@3': 0.875, 'H@10': 0.9583333333333334, 'MRR': 0.8472644080996884}

python retrieval_augmented_link_predictor.py --dataset_dir "KGs/Countries-S1" --model "GCL" --base_url "http://harebell.cs.upb.de:8501/v1" --num_of_hops 2
{'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}

python retrieval_augmented_link_predictor.py --dataset_dir "KGs/Countries-S2" --model "GCL" --base_url "http://harebell.cs.upb.de:8501/v1" --num_of_hops 2
@TODO: CD: There is some randomness on this setup. I dunno whay
{'H@1': 0.875, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 0.9305555555555555}
{'H@1': 0.9166666666666666, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 0.9583333333333334}

"""
import argparse
import networkx as nx
import rdflib
from rdflib import URIRef
from openai import OpenAI
import os
from typing import List, Dict, Tuple
import json
from dicee.knowledge_graph import KG
from dicee.evaluator import evaluate_lp, evaluate_lp_k_vs_all
from abc import ABC, abstractmethod
import torch
import re
import igraph
import os
import json
from typing import List, Tuple, Dict
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

class KnowledgeGraphPredictor:
    """
    A class for predicting missing relations in knowledge graphs using LLMs.
    """

    def __init__(self, api_key="super-secure-key", base_url="http://tentris-ml.cs.upb.de:8501/v1", model="tentris"):
        """
        Initialize the KnowledgeGraphPredictor.
        
        Parameters:
        -----------
        api_key : str, optional
        base_url : str, optional
        model : str, optional
        """
        # Set OpenAI API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please provide an API key or set the OPENAI_API_KEY environment variable.")

        self.base_url = base_url
        self.model = model

        # Initialize OpenAI client
        if self.base_url:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self.client = OpenAI(api_key=self.api_key)

    def build_knowledge_graph(self, rdf_file_path: str, format: str = "ttl") -> nx.DiGraph:
        """
        Build a directed NetworkX graph from an RDF file.
        
        Parameters:
        -----------
        rdf_file_path : str
            Path to the RDF file
        format : str, optional
            Format of the RDF file (default: "ttl")
            
        Returns:
        --------
        G : networkx.DiGraph
            The directed NetworkX graph built from the RDF data
        """
        # Load RDF graph
        g = rdflib.Graph()
        # NOTE: Currently uses RDF/TTL format! 
        g.parse(rdf_file_path, format=format)

        # Create a NetworkX directed graph
        G = nx.DiGraph()

        # Process the graph data
        for s, p, o in g:
            # NOTE: For now the borders relation is hardcoded! 
            if p == URIRef('http://dbpedia.org/ontology/borders'):
                # Get readable labels
                s_label = str(s).split('/')[-1].replace('>', '').replace('_', ' ')
                o_label = str(o).split('/')[-1].replace('>', '').replace('_', ' ')

                # Add edge to graph with relation as attribute
                G.add_edge(s_label, o_label, relation="borders")

        return G

    def extract_entity_neighborhood(self, G: nx.DiGraph, entity: str, k: int = 2) -> nx.DiGraph:
        """
        Extract the k-order neighborhood of an entity in the graph.
        
        Parameters:
        -----------
        G : networkx.DiGraph
            The knowledge graph
        entity : str
            The entity to extract neighborhood for
        k : int, optional
            The order of the neighborhood (default: 2)
            
        Returns:
        --------
        G_hr : networkx.DiGraph
            Subgraph containing the k-order neighborhood
        """
        # Initialize with the entity itself
        nodes = {entity}

        # Current frontier is the entity
        frontier = {entity}

        # Expand neighborhood k times
        for _ in range(k):
            new_frontier = set()

            for node in frontier:
                # Add outgoing neighbors
                out_neighbors = set(G.successors(node))
                new_frontier.update(out_neighbors)

                # Add incoming neighbors
                in_neighbors = set(G.predecessors(node))
                new_frontier.update(in_neighbors)

            # Update nodes and frontier
            nodes.update(new_frontier)
            frontier = new_frontier

        # Create subgraph with the collected nodes
        G_hr = G.subgraph(nodes).copy()

        return G_hr

    def extract_relation_neighborhood(self, G: nx.DiGraph, relation: str) -> nx.DiGraph:
        """
        Extract all triples with a specific relation.
        
        Parameters:
        -----------
        G : networkx.DiGraph
            The knowledge graph
        relation : str
            The relation to extract
            
        Returns:
        --------
        G_hr : networkx.DiGraph
            Subgraph containing all triples with the relation
        """
        # Get all edges with the specified relation
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == relation]

        # Create a new graph with these edges
        G_hr = nx.DiGraph()
        G_hr.add_edges_from(edges, relation=relation)

        return G_hr

    def generate_prompt(self, G_hr: nx.DiGraph, head: str, relation: str, candidates: List[str]) -> str:
        """
        Generate a prompt for the LLM to score candidates.
        
        Parameters:
        -----------
        G_hr : networkx.DiGraph
            The subgraph containing relevant context
        head : str
            The head entity
        relation : str
            The relation
        candidates : List[str]
            List of candidate tail entities
            
        Returns:
        --------
        prompt : str
            The prompt for the LLM
        """
        # Extract triples from the subgraph
        triples = []
        for u, v, d in G_hr.edges(data=True):
            triples.append(f"- {u} {d.get('relation', 'relates to')} {v}.")

        context = "\n".join(triples)

        # Create the prompt with emphasis on clean JSON response
        prompt = f"""Context:
{context}

Question: Which entities are most likely to have the relation '{relation}' with '{head}'? 
Assign scores between 0 and 1 to each candidate, where 1 means definitely related and 0 means definitely not related.

The context given might be incomplete - thus information not in the context can still result in a score of 1.0.
Use reasoning based on the context and your knowledge of the domain to justify your scores.

Candidates:
{', '.join(candidates)}

Respond with a clean, properly formatted JSON object using this exact format:
{{
    "reasoning": "Brief step-by-step reasoning for the scores",
    "scores": {{
        "{candidates[0]}": score1,
        "{candidates[1]}": score2,
        ...
    }}
}}

Important: Ensure your response is valid JSON without any markdown formatting or code blocks."""
        return prompt

    def query_openai(self, prompt: str) -> Dict:
        """
        Query the OpenAI API with a prompt.
        
        Parameters:
        -----------
        prompt : str
            The prompt for the LLM
            
        Returns:
        --------
        response : Dict
            The parsed response from the LLM
        """
        # Call the OpenAI API using the client
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that analyzes knowledge graphs and predicts missing relations. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
            )

            # Extract the response content
            content = response.choices[0].message.content.strip()

            # Try to extract JSON from the response
            try:
                # First try direct JSON parsing
                return json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown code blocks
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_str = content.split("```")[1].strip()
                else:
                    json_str = content

                # Clean up the string
                json_str = json_str.replace('\n', ' ').replace('\r', '')
                json_str = ' '.join(json_str.split())  # Normalize whitespace

                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    print(f"Cleaned JSON string: {json_str}")
                    return {"reasoning": "Error parsing response", "scores": {}}

        except Exception as e:
            print(f"Error querying LLM: {e}")
            return {"reasoning": "Error querying model", "scores": {}}
        
    def predict_missing_tails(self, head: str, relation: str, candidates: List[str], rdf_file_path: str) -> List[Tuple[str, float]]:
        """
        Predict missing tail entities for a given head and relation.
        
        Parameters:
        -----------
        head : str
            The head entity
        relation : str
            The relation
        candidates : List[str]
            List of candidate tail entities
        rdf_file_path : str
            Path to the RDF file
            
        Returns:
        --------
        ranked_candidates : List[Tuple[str, float]]
            List of candidates ranked by their scores
        """
        # Build the knowledge graph
        G = self.build_knowledge_graph(rdf_file_path)

        # Extract k-order neighborhood of the head entity
        G_entity = self.extract_entity_neighborhood(G, head, k=2)

        # Extract neighborhood of the relation
        G_relation = self.extract_relation_neighborhood(G, relation)

        # Combine the two subgraphs
        G_hr = nx.DiGraph()
        G_hr.add_edges_from(G_entity.edges(data=True))
        G_hr.add_edges_from(G_relation.edges(data=True))

        # Generate prompt for the LLM
        prompt = self.generate_prompt(G_hr, head, relation, candidates)

        # Query the LLM
        response = self.query_openai(prompt)
        print(response)
        # Extract scores
        scores = response.get("scores", {})

        # Rank candidates by scores
        ranked_candidates = [(candidate, scores.get(candidate, 0)) for candidate in candidates]
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)

        return ranked_candidates
class AbstractBaseLinkPredictorClass(ABC):
    def __init__(self, knowledge_graph: KG = None, name="dummy"):
        assert knowledge_graph is not None
        assert name is not None
        self.kg = knowledge_graph
        self.name = name

        # Create dictionaries
        #
        self.idx_to_entity = self.kg.entity_to_idx.set_index(self.kg.entity_to_idx.index)['entity'].to_dict()
        self.entity_to_idx = {idx: entity for entity, idx in self.idx_to_entity.items()}
        #
        self.idx_to_relation = self.kg.relation_to_idx.set_index(self.kg.relation_to_idx.index)['relation'].to_dict()
        self.relation_idx = {idx: rel for rel, idx in self.idx_to_relation.items()}

    def eval(self):
        pass

    @abstractmethod
    def forward_triples(self,x:torch.LongTensor) -> torch.FloatTensor:
        pass

    @abstractmethod
    def forward_k_vs_all(self,x:torch.LongTensor) -> torch.FloatTensor:
        pass

    def __call__(self, x: torch.LongTensor | Tuple[torch.LongTensor, torch.LongTensor], y_idx: torch.LongTensor = None):
        """Predicting missing triples """

        if isinstance(x, tuple):
            # x, y_idx = x
            raise NotImplementedError(
                "Currently, We do not support KvsSample. KvsSample allows a model to assign scores only on the selected entities.")
            # return self.forward_k_vs_sample(x=x, target_entity_idx=y_idx)
        else:
            shape_info = x.shape
            if len(shape_info) == 2:
                batch_size, dim = x.shape
                if dim == 3:
                    return self.forward_triples(x)
                elif dim == 2:
                    # h, y = x[0], x[1]
                    # Note that y can be relation or tail entity.
                    return self.forward_k_vs_all(x=x)
            else:
                raise RuntimeError("Unsupported shape: {}".format(shape_info))

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
    def __init__(self, knowledge_graph: KG = None,base_url:str=None, api_key:str=None, llm_model:str=None,
                 temperature:float=0.0, seed:int=42, num_of_hops:int=3, use_val:bool=True) -> None:
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
        self.train_set:List[Tuple[str]] = [(self.idx_to_entity[idx_h],
                                            self.idx_to_relation[idx_r],
                                            self.idx_to_entity[idx_t]) for idx_h,idx_r,idx_t in self.kg.train_set.tolist()]
        # Validation dataset
        self.val_set:List[Tuple[str]] = [(self.idx_to_entity[idx_h],
                                          self.idx_to_relation[idx_r],
                                          self.idx_to_entity[idx_t]) for idx_h,idx_r,idx_t in self.kg.valid_set.tolist()]

        triples = self.train_set + self.val_set if use_val else self.train_set
        self.igraph = self.build_igraph(triples)
        self.str_entity_to_igraph_vertice={i["name"]:i for i in self.igraph.vs}
        self.str_rel_to_igraph_edges={i["label"]:i for i in self.igraph.es}

        # Mapping from an entity to relevant triples.
        # A relevant triple contains a entity that is num_of_hops around of a given entity
        self.node_to_relevant_triples=dict()
        for entity, entity_node_object in self.str_entity_to_igraph_vertice.items():
            neighboring_nodes = self.igraph.neighborhood(entity_node_object, order=num_of_hops)
            subgraph = self.igraph.subgraph(neighboring_nodes)
            self.node_to_relevant_triples[entity] = {(subgraph.vs[edge.source]["name"], edge["label"], subgraph.vs[edge.target]["name"]) for edge in subgraph.es}

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
        
        Please provide a ranked list of at most {min(len(self.target_entities),15)} likely target entities from the following list, along with likelihoods for each: {self.target_entities}
    
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
    def build_igraph(graph:List[Tuple[str,str,str]]):
        ig_graph = igraph.Graph(directed=True)
        # Extract unique vertices from all quadruples
        vertices = set()
        edges = []
        labels = []
        for s,p,o in graph:
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
        extracted_triples = [(ig_graph.vs[edge.source]["name"], edge["label"], ig_graph.vs[edge.target]["name"]) for edge in ig_graph.es]
        # Not only the number but even the order must match
        assert extracted_triples == [triple for triple in graph]
        return ig_graph

    def forward_triples(self, x: torch.LongTensor):
        raise NotImplementedError("GraphContextLearner needs to implement it")

    def forward_k_vs_all(self,x: torch.LongTensor) -> torch.FloatTensor:
        batch_output = []
        # Iterate over batch of subject and relation pairs
        for i in x.tolist():
            # index of an entity and index of a relation.
            idx_h, idx_r =i
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
            scores_for_all_entities = [ -1.0 for _ in range(len(self.idx_to_entity))]
            for pred in prediction_response.predictions:
                try:
                    scores_for_all_entities[self.entity_to_idx[pred.entity]]=pred.score
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
                 temperature:float=1,seed:int=42) -> None:
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

    def forward_k_vs_all(self,x):
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
    def __init__(self, knowledge_graph: KG = None, base_url:str=None, api_key:str=None, llm_model:str=None,
                 temperature:float=0.0, seed:int=42, max_relation_examples:int=50, use_val:bool=True, 
                 exclude_source:bool=True) -> None:
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
        self.train_set:List[Tuple[str]] = [(self.idx_to_entity[idx_h],
                                            self.idx_to_relation[idx_r],
                                            self.idx_to_entity[idx_t]) for idx_h,idx_r,idx_t in self.kg.train_set.tolist()]
        # Validation dataset
        self.val_set:List[Tuple[str]] = [(self.idx_to_entity[idx_h],
                                          self.idx_to_relation[idx_r],
                                          self.idx_to_entity[idx_t]) for idx_h,idx_r,idx_t in self.kg.valid_set.tolist()]

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
                
        relation_context = "Here are examples of how the relation is used in the knowledge base:\n"
        for s, p, o in sorted(relation_triples):
            relation_context += f"- {s} {p} {o}\n"
        relation_context += "\n"

        base_prompt = f"""
        I'm trying to predict the most likely target entities for the following query:
        Source entity: {source}
        Relation: {relation}
        Query: ({source}, {relation}, ?)

        {relation_context}
        
        Please provide a ranked list of at most {min(len(self.target_entities),15)} likely target entities from the following list, along with likelihoods for each: {self.target_entities}
    
        Provide your answer in the following JSON format: {{"predictions": [{{"entity": "entity_name", "score": float_number}}]}}

        Notes:
        1. Use the provided knowledge about how the relation is used to inform your predictions.
        2. Only include entities that are plausible targets for this relation.
        3. For geographic entities, consider geographic location, regional classifications, and political associations.
        4. Rank the entities by likelihood of being the correct target.
        5. ONLY INCLUDE entities from the provided list in your predictions.
        6. If certain entities are not suitable for this relation, don't include them.
        7. Return a valid JSON output.
        8. Make sure scores are floating point numbers between 0 and 1, not strings.
        9. A score can only be between 0 and 1, i.e. score ∈ [0, 1]. They can never be negative or greater than 1!
        """
        return base_prompt

    def forward_triples(self, x: torch.LongTensor):
        raise NotImplementedError("RCL needs to implement it")

    def forward_k_vs_all(self,x: torch.LongTensor) -> torch.FloatTensor:
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
            scores_for_all_entities = [ -1.0 for _ in range(len(self.idx_to_entity))]
            for pred in prediction_response.predictions:
                try:
                    scores_for_all_entities[self.entity_to_idx[pred.entity]]=pred.score
                except KeyError:
                    print(f"For {h},{r}, {pred} not found\tPrediction Size: {len(prediction_response.predictions)}")
                    continue
            batch_output.append(scores_for_all_entities)
        return torch.FloatTensor(batch_output)

def sanity_checking(args,kg):
    if args.eval_size is not None:
        assert len(kg.test_set) >= args.eval_size, (f"Evaluation size cant be greater than the "
                                                    f"total amount of triples in the test set: {len(kg.test_set)}")
    else:
        args.eval_size = len(kg.test_set)
    if args.api_key is None:
        args.api_key = os.environ.get("TENTRIS_TOKEN")

def get_model(args,kg)->AbstractBaseLinkPredictorClass:
    # () Initialize the link prediction model
    if args.model == "RALP":
        model = RALP(knowledge_graph=kg, base_url=args.base_url, api_key=args.api_key,
                     llm_model=args.llm_model_name, temperature=args.temperature, seed=args.seed)
    elif args.model == "GCL":
        model = GCL(knowledge_graph=kg, base_url=args.base_url, api_key=args.api_key,
                    llm_model=args.llm_model_name, temperature=args.temperature, seed=args.seed,num_of_hops=args.num_of_hops)
    elif args.model == "RCL":
        model = RCL(knowledge_graph=kg, base_url=args.base_url, api_key=args.api_key,
                    llm_model=args.llm_model_name, temperature=args.temperature, seed=args.seed, 
                    max_relation_examples=args.max_relation_examples, exclude_source=args.exclude_source)
    else:
        raise KeyError(f"{args.model} is not a valid model")
    assert model is not None, f"Couldn't assign a model named: {args.model}"
    return model

def run(args):
    # Important: add_reciprocal=False in KvsAll implies that inverse relation has been introduced.
    # Therefore, The link prediction results are based on the missing tail rankings only!
    kg = KG(dataset_dir=args.dataset_dir, separator="\s+", eval_model=args.eval_model, add_reciprocal=False)

    sanity_checking(args,kg)

    model = get_model(args,kg)

    results:dict = evaluate_lp_k_vs_all(model=model, triple_idx=kg.test_set[:args.eval_size],
                         er_vocab=kg.er_vocab, info='Eval KvsAll Starts', batch_size=args.batch_size)
    print(results)
    #evaluate_lp(model=model, triple_idx=kg.test_set[:args.eval_size], num_entities=len(kg.entity_to_idx),
    #            er_vocab=kg.er_vocab, re_vocab=kg.re_vocab, info='Eval LP Starts', batch_size=args.batch_size,
    #            chunk_size=args.chunk_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="KGs/Countries-S1", help="Path to dataset.")
    parser.add_argument("--model", type=str, default="GCL", help="Model name to use for link prediction.", choices=["RALP", "GCL", "RCL"])
    parser.add_argument("--base_url", type=str, default="http://harebell.cs.upb.de:8501/v1",
                        choices=["http://harebell.cs.upb.de:8501/v1", "http://tentris-ml.cs.upb.de:8502/v1"],
                        help="Base URL for the OpenAI client.")
    parser.add_argument("--llm_model_name", type=str, default="tentris", help="Model name of the LLM to use.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature hyperparameter for LLM calls.")
    parser.add_argument("--api_key", type=str, default=None, help="API key for the OpenAI client. If left to None, "
                                                                  "it will look at the environment variable named "
                                                                  "TENTRIS_TOKEN from a local .env file.")
    parser.add_argument("--eval_size", type=int, default=None,
                        help="Amount of triples from the test set to evaluate. "
                             "Leave it None to include all triples on the test set.")
    parser.add_argument("--eval_model", type=str, default="train_value_test",
                        help="Type of evaluation model.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_of_hops", type=int, default=1, help="Number of hops to use to extract a subgraph around an entity.")
    parser.add_argument("--max_relation_examples", type=int, default=50, help="Maximum number of relation examples to include in RCL context.")
    parser.add_argument("--exclude_source", default=True, help="Exclude triples with the same source entity in RCL context.")
    run(parser.parse_args())
