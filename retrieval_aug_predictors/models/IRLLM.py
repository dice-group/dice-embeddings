import argparse
import os
from typing import List, Set

import dspy
from owlapy.class_expression import OWLClass
from owlapy.owl_ontology import Ontology
from owlapy.owl_reasoner import StructuralReasoner
from rdflib import Graph
from dotenv import load_dotenv

load_dotenv()

class Classifier(dspy.Signature):
    graph = dspy.InputField(desc="All triples in the knowledge graph.")
    concept: str = dspy.InputField(desc="The named concept that is used to classify entities of a knowledge graph")
    classified_entities: List[str] = dspy.OutputField(desc="All the entities from the knowledge graph that can be classified by the given concept. Result must be a list of unique entities.")




class IRLLM:
    """Instance Retrieval via LLM"""
    def __init__(self, path, base_url, api_key, temperature, seed, llm_model):
        self.kg_path = path
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.llm_model = llm_model
        self.triples = []

        g = Graph()
        g.parse(path, format="xml")
        self.triples = [f"{str(s)} {str(p)} {str(o)}" for s, p, o in g]

    def evaluate(self):

        onto = Ontology(self.kg_path)
        reasoner = StructuralReasoner(onto)

        lm = dspy.LM(model=f"openai/{self.llm_model}", api_key=self.api_key, api_base=self.base_url,
                     temperature=self.temperature, seed=self.seed, cache=True, cache_in_memory=True)
        dspy.configure(lm=lm)

        cot_model = dspy.ChainOfThought(Classifier)

        concept = "Female"
        concept_in_owl = OWLClass("http://example.com/father#female")

        true_instances = {i.str for i in reasoner.instances(concept_in_owl)}
        print(true_instances)
        predicted_instances = set(cot_model(graph=self.triples, concept=concept).classified_entities)
        print(predicted_instances)

        intersection = true_instances.intersection(predicted_instances)
        union = true_instances.union(predicted_instances)
        jaccard_similarity = len(intersection) / len(union)

        print("Jaccard similarity:", jaccard_similarity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg_path", type=str, default="/home/alkid/PycharmProjects/dice-embeddings/KGs/Family/father.owl")
    parser.add_argument("--base_url", type=str, default="http://harebell.cs.upb.de:8501/v1")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm_model", type=str, default="tentris")
    args = parser.parse_args()
    if args.api_key is None:
        args.api_key = os.environ.get("TENTRIS_TOKEN")

    model = IRLLM(args.kg_path, args.base_url, args.api_key, args.temperature, args.seed, args.llm_model)

    model.evaluate()