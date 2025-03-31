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
import torch
import re
import igraph
import os
import json
from typing import List, Tuple, Dict
import numpy as np
from typing import List, Optional
from dotenv import load_dotenv
from retrieval_aug_predictors import AbstractBaseLinkPredictorClass, RALP, GCL, RCL, Demir

load_dotenv()


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
    elif args.model == "Demir":
        model = Demir(knowledge_graph=kg, base_url=args.base_url, api_key=args.api_key,
                     llm_model=args.llm_model_name, temperature=args.temperature, seed=args.seed)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="KGs/Countries-S1", help="Path to dataset.")
    parser.add_argument("--model", type=str, default="Demir", help="Model name to use for link prediction.", choices=["Demir", "GCL", "RCL","RALP"])
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
    parser.add_argument("--max_relation_examples", type=int, default=2000, help="Maximum number of relation examples to include in RCL context.")
    parser.add_argument("--exclude_source", action="store_true", help="Exclude triples with the same source entity in RCL context.")
    run(parser.parse_args())