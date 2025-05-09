import argparse
import ast
import os
from typing import List, Set

import dspy
import pandas as pd
from owlapy import OntologyManager
from owlapy.class_expression import OWLClass
from owlapy.iri import IRI
from owlapy.owl_ontology import Ontology
from owlapy.owl_reasoner import StructuralReasoner
from rdflib import Graph
from dotenv import load_dotenv

load_dotenv()

class Classifier(dspy.Signature):
    graph: str = dspy.InputField(desc="All triples in the knowledge graph. Entities and relations include their namespace.")
    # concept: str = dspy.InputField(desc="The description logics concept that is used to classify entities of a knowledge graph")
    concept: str = dspy.InputField(desc="A concept or class in natural language.")
    classified_entities: list[str] = dspy.OutputField(desc="All the entities from the knowledge graph that can be classified by the given concept or class. Result must be a list of unique entities.")

class Verbaliser(dspy.Signature):
    expression = dspy.InputField(desc="An expression in description logics. Example: ∃ hasChild.{markus} --> There exist a named individual who has child and this child is Markus")
    verbalisation = dspy.OutputField(desc="A concise verbalisation of the expression in natural language.")

class PredictionModule(dspy.Module):
    def __init__(self):
        self.verbaliser = dspy.Predict(Verbaliser)
        self.classifier = dspy.ChainOfThought(Classifier)

    def forward(self, graph: str, dl_expression: str) -> list[str]:

        verbalized_concept = self.verbaliser(expression=dl_expression).verbalisation
        predicted_instances = self.classifier(graph=graph, concept=verbalized_concept).classified_entities

        return predicted_instances

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
        # mg = OntologyManager()
        # onto = Ontology(manager=mg, ontology_iri=IRI.create("file://" + self.kg_path), load=True)
        # reasoner = StructuralReasoner(onto)

        lm = dspy.LM(model=f"openai/{self.llm_model}", api_key=self.api_key, api_base=self.base_url,
                     temperature=self.temperature, seed=self.seed, cache=True, cache_in_memory=True)
        dspy.configure(lm=lm)

        program = PredictionModule()


        df = pd.read_csv("ALCQHI_Retrieval_Results.csv")
        file_exists = False
        for index, row in df.iterrows():
            concept = row[0]

            # true_instances = {i.str for i in reasoner.instances(concept_in_owl)}
            true_instances = ast.literal_eval(row[2])
            predicted_instances = set(program(graph=self.triples, dl_expression=concept))

            intersection = true_instances.intersection(predicted_instances)
            union = true_instances.union(predicted_instances)
            if len(union) == 0:
                jaccard_similarity = 1.0
            else:
                jaccard_similarity = len(intersection) / len(union)

            df_row = pd.DataFrame(
                [{
                    "Expression": concept,
                    "True_set": true_instances,
                    "Pred_set": predicted_instances,
                    "Jaccard_similarity": jaccard_similarity,
                }])
            # Append the row to the CSV file
            df_row.to_csv("Results.csv", mode='a', header=not file_exists, index=False)
            file_exists = True


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