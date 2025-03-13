"""
# pip install openai
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
    def __init__(self, knowledge_graph:KG=None, name="dummy") -> None:
        super().__init__(knowledge_graph,name)

    def __call__(self,indexed_triples:torch.LongTensor):
        n,d=indexed_triples.shape
        # For the time being
        assert d==3
        assert n==1
        scores=[]
        for triple in indexed_triples.tolist():
            idx_h, idx_r, idx_t = triple
            h,r,t=self.idx_to_entity[idx_h], self.idx_to_relation[idx_r], self.idx_to_entity[idx_t]
            # Given this triple, we need to assign a score
            scores.append([0.0])
        return torch.FloatTensor(scores)

if __name__ == "__main__":
    # () Read / Preprocess KG
    kg = KG(dataset_dir="KGs/Countries-S1",separator="\s+",eval_model="train_val_test")

    evaluate_lp(model=Dummy(knowledge_graph=kg), triple_idx=kg.train_set, num_entities=len(kg.entity_to_idx), er_vocab=kg.er_vocab,
                re_vocab=kg.re_vocab,  info='Eval LP Starts', batch_size=1, chunk_size=1)

    # @TODO: Create classes inherits from AbstractBaseLinkPredictorClass and improve the link prediction results
    exit(1)
    # @TODO:CD -> Luke: Please refactor the below code to work with the above code.
    # Create predictor (uses Tentris model by default)
    predictor = KnowledgeGraphPredictor()

    print("\nExample: Countries that border Italy")
    head = "Italy"
    relation = "borders"
    candidates = ["France", "Austria", "Switzerland", "Slovenia", "Vatican", "San Marino"]
    
    # Predict missing tails
    ranked_candidates = predictor.predict_missing_tails(head, relation, candidates, "data/countries.ttl")
    
    # Print results
    print(f"\nPredicting missing tails for ({head}, {relation}, ?)")
    print("\nRanked candidates with scores:")
    for candidate, score in ranked_candidates:
        print(f"{candidate}: {score:.2f}")

    print("\ndone!")


'''
Data used:

@prefix ex: <http://example.org/> .
@prefix dbo: <http://dbpedia.org/ontology/> .
@prefix dbr: <http://dbpedia.org/resource/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# Germany and its borders
ex:Germany a dbo:Country ;
    rdfs:label "Germany" ;
    dbo:capital dbr:Berlin ;
    dbo:continent dbr:Europe ;
    dbo:borders ex:France, ex:Poland, ex:Netherlands, ex:Austria, ex:Czech_Republic, ex:Denmark . 

# France with partial missing borders
ex:France a dbo:Country ;
    rdfs:label "France" ;
    dbo:capital dbr:Paris ;
    dbo:continent dbr:Europe ;
    dbo:borders ex:Germany, ex:Belgium, ex:Spain, ex:Italy .  
    # Missing Luxembourg, Switzerland, Monaco

# Poland with missing some eastern borders
ex:Poland a dbo:Country ;
    rdfs:label "Poland" ;
    dbo:capital dbr:Warsaw ;
    dbo:continent dbr:Europe ;
    dbo:borders ex:Germany, ex:Czech_Republic, ex:Slovakia, ex:Lithuania .  
    # Missing Ukraine, Belarus

# Netherlands with incomplete neighbors
ex:Netherlands a dbo:Country ;
    rdfs:label "Netherlands" ;
    dbo:capital dbr:Amsterdam ;
    dbo:continent dbr:Europe ;
    dbo:borders ex:Germany, ex:Belgium .  
    # Missing North Sea relation

# Belgium with missing some minor relations
ex:Belgium a dbo:Country ;
    rdfs:label "Belgium" ;
    dbo:capital dbr:Brussels ;
    dbo:borders ex:France, ex:Netherlands, ex:Germany .  
    # Missing Luxembourg

# Austria with some missing borders
ex:Austria a dbo:Country ;
    rdfs:label "Austria" ;
    dbo:capital dbr:Vienna ;
    dbo:continent dbr:Europe ;
    dbo:borders ex:Germany, ex:Czech_Republic, ex:Slovakia, ex:Italy .  
    # Missing Switzerland, Slovenia, Hungary

# Czech Republic with missing some eastern borders
ex:Czech_Republic a dbo:Country ;
    rdfs:label "Czech Republic" ;
    dbo:capital dbr:Prague ;
    dbo:continent dbr:Europe ;
    dbo:borders ex:Germany, ex:Poland, ex:Austria .  
    # Missing Slovakia

# Denmark with only one neighbor
ex:Denmark a dbo:Country ;
    rdfs:label "Denmark" ;
    dbo:capital dbr:Copenhagen ;
    dbo:continent dbr:Europe ;
    dbo:borders ex:Germany .  
    # Missing maritime neighbors (Sweden, Norway via sea)

# Italy with missing eastern borders
ex:Italy a dbo:Country ;
    rdfs:label "Italy" ;
    dbo:capital dbr:Rome ;
    dbo:continent dbr:Europe ;
    dbo:borders ex:France, ex:Austria .  
    # Missing Slovenia, Switzerland, Vatican, San Marino

# Slovakia with partial information
ex:Slovakia a dbo:Country ;
    rdfs:label "Slovakia" ;
    dbo:capital dbr:Bratislava ;
    dbo:continent dbr:Europe ;
    dbo:borders ex:Poland, ex:Czech_Republic, ex:Austria .  
    # Missing Hungary, Ukraine
'''