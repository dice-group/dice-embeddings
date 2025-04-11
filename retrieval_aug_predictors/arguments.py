import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="/home/alkid/PycharmProjects/dice-embeddings/KGs/Countries-S1", help="Path to dataset.")
parser.add_argument("--model", type=str, default="Demir", help="Model name to use for link prediction.",
                    choices=["Demir", "GCL", "RCL", "RALP"])
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
parser.add_argument("--num_of_hops", type=int, default=1,
                    help="Number of hops to use to extract a subgraph around an entity.")
parser.add_argument("--max_relation_examples", type=int, default=2000,
                    help="Maximum number of relation examples to include in RCL context.")
parser.add_argument("--exclude_source", action="store_true",
                    help="Exclude triples with the same source entity in RCL context.")
parser.add_argument("--out", type=str, default=None,
                    help="A path of a json file reporting the link prediction results.")
parser.add_argument("--print_top_predictions", action="store_true",
                    help="Whether you want to print top-k predictions. Set k by using --k flag.")
parser.add_argument("--k", type=int, default=1,
                    help="Number of top predictions to print for each (h, r) ")
