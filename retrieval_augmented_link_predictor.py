"""
# pip install openai==1.66.3
# Data Preparation
# Read a knowledge graph (Countries)

Let g:(train,val,test) be a tuple of three knowledge graphs
For g[i] \in E x R x E, where
E denotes a set of entities
R denotes a set of relations

# Get the test dataset
train : List[Tuple[str,str,str]] = g[0]
test  : List[Tuple[str,str,str]] = g[2]

1. Move to train into directed graph of networkx (https://networkx.org/documentation/stable/index.html) or igraph (https://python.igraph.org/en/stable/)
Although this is not necessary, they implement few functions that we would like to use in the next steps.


# Link Prediction

Let (h,r,t) be a test triple

## Predicting missing tail
Given (h,r) rank elements of E in te descending order of their relevance.

#### Getting information about an entity (h)

1. Getting k order neighbors of an entity
Let n_h := {(s,p,o)} denote a set of triples from the train set, where h==s or h==o.
n_h denotes the first order neighborhood (see https://python.igraph.org/en/stable/analysis.html#neighborhood)
we can extend this into k>1 to get a subgraph that is "about h".


2. Getting k order neighbors of a relation
Let m_r := {(s,p,o)} denote a set of triples from the train set, where p==r.
Similarly,
- m_h denotes the first order neighborhood of r
- we can extend this into k>1 to get a subgraph that is "about r".

For the time being, assume that k=3.

3. Assigning scores to entities based on information derived from (1) and (2)
Let G_hr denote a set of triples derived from (1) and (2)
Let E_hr denote a set of filtered entities (a concept from the link prediction evaluation)

3.1.Write a prompt based on G_hr, and E_hr so that LLM generates scores for each item of E_hr

3.2. We are done :)


"""
import argparse

#### NOTE: LF: First implementation approach

import networkx as nx
import rdflib
from rdflib import URIRef
from openai import OpenAI
import os
from typing import List, Dict, Tuple
import json
from dicee.knowledge_graph import KG
from dicee.evaluator import evaluate_lp
from abc import ABC, abstractmethod
import torch
import re

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
                    {"role": "system", "content": "You are a helpful assistant that analyzes knowledge graphs and predicts missing relations. Always return valid JSON."},
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
    def __init__(self, knowledge_graph:KG=None,name="dummy"):
        assert knowledge_graph is not None
        assert name is not None
        self.kg = knowledge_graph
        self.name = name

        # Mappings from str to idx
        # kg.entity_to_idx:pd.DataFrame
        # kg.relation_to_idx:pd.DataFrame
        # indexed KGs
        # kg.train_set : numpy.ndarray
        # kg.valid_set : numpy.ndarray
        # kg.test_set  :  numpy.ndarray

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
    def __call__(self,*args,**kwargs):
        """Predicting missing triples"""


class Dummy(AbstractBaseLinkPredictorClass):
    def __init__(self, knowledge_graph: KG = None, name="dummy") -> None:
        super().__init__(knowledge_graph, name)

    def __call__(self,indexed_triples:torch.LongTensor):
        n,d=indexed_triples.shape
        # For the time being
        assert d == 3
        assert n == 1
        scores = []
        for triple in indexed_triples.tolist():
            idx_h, idx_r, idx_t = triple
            h, r, t = self.idx_to_entity[idx_h], self.idx_to_relation[idx_r], self.idx_to_entity[idx_t]
            # Given this triple, we need to assign a score
            scores.append([0.0])
        return torch.FloatTensor(scores)


class RALP(AbstractBaseLinkPredictorClass):
    def __init__(self, knowledge_graph: KG = None,
                 name="ralp-1.0",
                 base_url="http://tentris-ml.cs.upb.de:8501/v1",
                 api_key=None,
                 llm_model="tentris",
                 temperature=1) -> None:
        super().__init__(knowledge_graph, name)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.llm_model = llm_model
        self.temperature = temperature

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

    def __call__(self, indexed_triples: torch.LongTensor):
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


def run(args):

    # () Read KG
    kg = KG(dataset_dir=args.dataset_dir, separator="\s+", eval_model=args.eval_model)
    if args.eval_size is not None:
        assert len(kg.test_set) >= args.eval_size, (f"Evaluation size cant be greater than the "
                                                    f"total amount of triples in the test set: {len(kg.test_set)}")
    else:
        args.eval_size = len(kg.test_set)
    model = None

    # () Initialize the link prediction model
    if args.model == "RALP":
        model = RALP(knowledge_graph=kg,
                     base_url=args.base_url,
                     api_key=args.api_key,
                     llm_model=args.llm_model_name,
                     temperature=args.temperature)

    assert model is not None, f"Couldn't assign a model named: {args.model}"

    # () Start evaluation
    evaluate_lp(model=model, triple_idx=kg.test_set[:args.eval_size], num_entities=len(kg.entity_to_idx),
                er_vocab=kg.er_vocab, re_vocab=kg.re_vocab, info='Eval LP Starts', batch_size=args.batch_size,
                chunk_size=args.chunk_size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="KGs/Countries-S1", help="Path to dataset.")
    parser.add_argument("--model", type=str, default="RALP", help="Model name to use for link prediction.", choices=["RALP"]) # add new models in 'choices'
    parser.add_argument("--base_url", type=str, default="http://tentris-ml.cs.upb.de:8501/v1",
                        help="Base URL for the OpenAI client.")
    parser.add_argument("--llm_model_name", type=str, default="tentris", help="Model name of the LLM to use.")
    parser.add_argument("--api_key", type=str, default="INSERT_API_KEY", help="API key for the OpenAI client.")
    parser.add_argument("--temperature", type=float, default=1, help="Temperature hyperparameter for LLM calls.")
    parser.add_argument("--eval_size", type=int, default=None,
                        help="Amount of triples from the test set to evaluate. "
                             "Leave it None to include all triples on the test set.")
    parser.add_argument("--eval_model", type=str, default="train_value_test", help="Type of evaluation model.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=1)

    run(parser.parse_args())
