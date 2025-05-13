import argparse
import ast
import os
import sys
from typing import List, Set, Dict, Any

import dspy
import pandas as pd
from owlapy import OntologyManager # Although commented out, keeping imports for potential future use
from owlapy.class_expression import OWLClass
from owlapy.iri import IRI
from owlapy.owl_ontology import Ontology
from owlapy.owl_reasoner import StructuralReasoner
from rdflib import Graph, exceptions as rdf_exceptions
from dotenv import load_dotenv
from tqdm import tqdm # Import tqdm

# Load environment variables from a .env file
load_dotenv()

# (Keep the Classifier, Verbaliser, and PredictionModule classes as they were)
class Classifier(dspy.Signature):
    """Classifies entities in a knowledge graph based on a natural language concept."""
    graph: str = dspy.InputField(desc="All triples in the knowledge graph. Entities and relations include their namespace.")
    concept: str = dspy.InputField(desc="A concept or class in natural language.")
    classified_entities: List[str] = dspy.OutputField(desc="All the entities from the knowledge graph that can be classified by the given concept or class. Result must be a list of unique entities.")

class Verbaliser(dspy.Signature):
    """Verbalises a description logic expression into natural language."""
    expression: str = dspy.InputField(desc="An expression in description logics. Example: ∃ hasChild.{markus} --> There exist a named individual who has child and this child is Markus")
    verbalisation: str = dspy.OutputField(desc="A concise verbalisation of the expression in natural language.")

class PredictionModule(dspy.Module):
    """A module that verbalises a DL expression and then classifies entities using the verbalisation."""
    def __init__(self):
        super().__init__()
        # Use dspy.Predict for simple signature execution
        self.verbaliser = dspy.Predict(Verbaliser)
        # Use dspy.ChainOfThought for multi-step reasoning in classification
        self.classifier = dspy.ChainOfThought(Classifier)

    def forward(self, graph: str, dl_expression: str) -> List[str]:
        """
        Verbalises a DL expression and uses the verbalisation to classify entities in a graph.

        Args:
            graph: The knowledge graph as a string of triples.
            dl_expression: The description logic expression.

        Returns:
            A list of entities classified by the verbalised concept.
        """
        # Verbalise the description logic expression
        verbalized_concept = self.verbaliser(expression=dl_expression).verbalisation
        # Classify entities in the graph based on the verbalised concept
        predicted_instances = self.classifier(graph=graph, concept=verbalized_concept).classified_entities

        return predicted_instances

# (Keep the IRLLM class __init__ and _load_knowledge_graph methods as they were)
class IRLLM:
    """
    Instance Retrieval via LLM.

    Handles loading the knowledge graph, configuring the LLM, and evaluating
    the LLM's ability to retrieve instances for given concepts.
    """
    def __init__(self, kg_path: str, base_url: str, api_key: str, temperature: float, seed: int, llm_model: str):
        """
        Initializes the IRLLM with KG path and LLM configuration.

        Args:
            kg_path: Path to the knowledge graph file.
            base_url: Base URL for the LLM API.
            api_key: API key for the LLM.
            temperature: Temperature setting for the LLM.
            seed: Seed for the LLM (if applicable).
            llm_model: The name or identifier of the LLM model to use.
        """
        self.kg_path = kg_path
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.llm_model = llm_model
        self.triples: List[str] = [] # Using type hint for clarity

        self._load_knowledge_graph()

    def _load_knowledge_graph(self):
        """Loads the knowledge graph from the specified path."""
        if not os.path.exists(self.kg_path):
            print(f"Error: Knowledge graph file not found at {self.kg_path}", file=sys.stderr)
            sys.exit(1)

        g = Graph()
        try:
            # Attempt to parse the graph, try different formats if necessary
            # You might want to add more robust format detection if needed
            g.parse(self.kg_path, format="xml") # Starting with xml as in original code
        except rdf_exceptions.ParserError as e:
             print(f"Error parsing knowledge graph file {self.kg_path}: {e}", file=sys.stderr)
             # Attempt other formats if parsing fails
             try:
                 g.parse(self.kg_path, format="ttl") # Try Turtle format
             except rdf_exceptions.ParserError as e:
                 print(f"Error parsing knowledge graph file {self.kg_path} in Turtle format: {e}", file=sys.stderr)
                 sys.exit(1)
             except Exception as e:
                 print(f"An unexpected error occurred while parsing {self.kg_path} in Turtle format: {e}", file=sys.stderr)
                 sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while parsing {self.kg_path}: {e}", file=sys.stderr)
            sys.exit(1)


        self.triples = [f"{str(s)} {str(p)} {str(o)}" for s, p, o in g]
        print(f"Successfully loaded {len(self.triples)} triples from {self.kg_path}")


    def evaluate(self, results_csv_path: str = "ALCQHI_Retrieval_Results.csv", output_csv_path: str = "Results.csv"):
        """
        Evaluates the LLM's instance retrieval performance against a ground truth CSV.

        Includes a progress bar with the current average Jaccard similarity and
        reports descriptive statistics for Jaccard similarities at the end.

        Args:
            results_csv_path: Path to the CSV file containing DL expressions and true instances.
            output_csv_path: Path to save the evaluation results.
        """
        # Configure the DSPy language model
        if not self.api_key:
            print("Error: LLM API key is not provided.", file=sys.stderr)
            sys.exit(1)

        try:
            # Corrected to use dspy.OpenAI directly if it's the intended LM
            # Or use dspy.LM if it wraps other providers. Assuming dspy.OpenAI
            # is the typical use case when api_key and api_base are provided.
            # If 'tentris' requires a different LM class, adjust here.
            # Based on the original code `dspy.LM(model=f"openai/{self.llm_model}"...`
            # it seems like dspy.OpenAI might be the more appropriate class if
            # the base_url points to an OpenAI-compatible API.
            # If `self.llm_model` is not meant to be prepended by "openai/",
            # and the base_url points to a different provider supported by dspy.LM,
            # then the original `dspy.LM` call was correct. Let's stick to the original
            # `dspy.LM` as it handles different backends.
            lm = dspy.LM(model=f"openai/{self.llm_model}", api_key=self.api_key, api_base=self.base_url,
                         temperature=self.temperature, seed=self.seed, cache=True, cache_in_memory=True)
            dspy.configure(lm=lm)
            print(f"DSPy configured with model: {self.llm_model}, base_url: {self.base_url}")
        except Exception as e:
            print(f"Error configuring DSPy or connecting to LLM: {e}", file=sys.stderr)
            sys.exit(1)


        program = PredictionModule()

        # Load the ground truth results CSV once
        if not os.path.exists(results_csv_path):
            print(f"Error: Ground truth results file not found at {results_csv_path}", file=sys.stderr)
            sys.exit(1)

        try:
            df_ground_truth = pd.read_csv(results_csv_path)
            if df_ground_truth.empty:
                 print(f"Warning: Ground truth results file {results_csv_path} is empty. Nothing to evaluate.", file=sys.stderr)
                 return # Exit if there's no data to process
        except FileNotFoundError:
             # Already checked for existence, but good practice to handle
             print(f"Error reading ground truth results file {results_csv_path}.", file=sys.stderr)
             sys.exit(1)
        except pd.errors.EmptyDataError:
            print(f"Warning: Ground truth results file {results_csv_path} is empty. Nothing to evaluate.", file=sys.stderr)
            return
        except Exception as e:
            print(f"An unexpected error occurred while reading {results_csv_path}: {e}", file=sys.stderr)
            sys.exit(1)

        evaluation_results: List[Dict[str, Any]] = []
        total_jaccard_similarity = 0.0 # Variable to sum Jaccard similarities

        print(f"\nStarting evaluation of {len(df_ground_truth)} concepts...")

        # Iterate through each row in the ground truth data with a progress bar
        # Using iterrows is acceptable for smaller DataFrames, but for very large ones,
        # converting to a list of dictionaries might be more performant.
        # Added tqdm wrapper for progress bar
        for index, row in (tqdm_bar :=tqdm(df_ground_truth.iterrows(), total=len(df_ground_truth), desc="Evaluating Concepts")):
            try:
                concept = row.iloc[0] # Use iloc for potentially better performance and clarity
                # Assuming the true instances are in the third column (index 2)
                # and are stored as a string representation of a Python list/set
                true_instances_str = row.iloc[2]
                true_instances: Set[str] = set(ast.literal_eval(true_instances_str))

                # print(f"\nProcessing concept: {concept}") # Suppress verbose output in tqdm loop
                # print(f"True instances: {true_instances}")

                # Get predictions from the LLM
                # Pass graph as a single string as required by the signature
                predicted_instances_list = program(graph="\n".join(self.triples), dl_expression=concept)
                # Ensure predicted instances are unique and in a set for easy comparison
                predicted_instances: Set[str] = set(predicted_instances_list)

                # print(f"Predicted instances: {predicted_instances}")

                # Calculate Jaccard Similarity
                intersection = true_instances.intersection(predicted_instances)
                union = true_instances.union(predicted_instances)

                jaccard_similarity = 1.0 if len(union) == 0 else len(intersection) / len(union)

                # print(f"Jaccard Similarity: {jaccard_similarity}") # Suppress verbose output

                # Accumulate Jaccard similarity for average calculation
                total_jaccard_similarity += jaccard_similarity

                # Update the progress bar description with the current average Jaccard similarity
                current_avg_jaccard = total_jaccard_similarity / (index + 1)
                tqdm_bar.set_postfix({"Avg Jaccard": f"{current_avg_jaccard:.4f}", "Concept": concept[:50] + '...'}) # Show avg and current concept


                # Store results
                evaluation_results.append({
                    "Expression": concept,
                    "True_set": list(true_instances), # Convert set back to list for CSV writing
                    "Pred_set": list(predicted_instances), # Convert set back to list for CSV writing
                    "Jaccard_similarity": jaccard_similarity,
                })
            except KeyError as e:
                print(f"Error processing row {index}: Missing expected column. Details: {e}", file=sys.stderr)
                # Decide whether to continue or break
                continue # Continue processing the next row
            except ValueError as e:
                 print(f"Error processing row {index} due to invalid value (e.g., in ast.literal_eval): {e}", file=sys.stderr)
                 continue # Continue processing the next row
            except Exception as e:
                print(f"An unexpected error occurred while processing row {index} for concept '{concept}': {e}", file=sys.stderr)
                # Depending on severity, you might want to sys.exit(1) here
                continue # Continue processing the next row

        # Write all results to the output CSV outside the loop
        if evaluation_results:
            df_results = pd.DataFrame(evaluation_results)
            try:
                df_results.to_csv(output_csv_path, index=False)
                print(f"\nEvaluation results saved to {output_csv_path}")

                # Use pandas.DataFrame.describe() on jaccard similarities
                print("\n--- Jaccard Similarity Descriptive Statistics ---")
                print(df_results['Jaccard_similarity'].describe())
                print("-----------------------------------------------")

            except Exception as e:
                print(f"Error writing evaluation results to {output_csv_path}: {e}", file=sys.stderr)
        else:
            print("\nNo evaluation results to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instance Retrieval via LLM")
    parser.add_argument("--kg_path", type=str, default="/home/cdemir/Desktop/Softwares/Ontolearn/KGs/Family/father.owl",
                        help="Path to the knowledge graph file (e.g., in OWL/XML format).")
    parser.add_argument("--base_url", type=str, default="http://harebell.cs.upb.de:8501/v1",
                        help="Base URL for the LLM API (e.g., OpenAI compatible endpoint).")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for the LLM. Can also be provided via TENTRIS_TOKEN environment variable.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature setting for the LLM (controls randomness).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the LLM (for reproducibility if the LLM supports it).")
    parser.add_argument("--llm_model", type=str, default="tentris",
                        help="The name or identifier of the LLM model to use.")
    parser.add_argument("--results_csv_path", type=str, default="ALCQHI_Retrieval_Results.csv",
                        help="Path to the CSV file containing ground truth DL expressions and instances.")
    parser.add_argument("--output_csv_path", type=str, default="Results.csv",
                        help="Path to save the evaluation results CSV.")

    args = parser.parse_args()

    # Load API key from environment variable if not provided as argument
    if args.api_key is None:
        args.api_key = os.environ.get("TENTRIS_TOKEN")
        if args.api_key:
            print("Using API key from TENTRIS_TOKEN environment variable.")
        else:
            print("Warning: No API key provided via argument or TENTRIS_TOKEN environment variable.", file=sys.stderr)


    # Instantiate and run the evaluation
    model = IRLLM(args.kg_path, args.base_url, args.api_key, args.temperature, args.seed, args.llm_model)

    model.evaluate(results_csv_path=args.results_csv_path, output_csv_path=args.output_csv_path)