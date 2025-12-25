from dicee.executer import Execute
from dicee.scripts.run import get_default_arguments
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run_dicee_eval(
    dataset_folder,
    model,
    num_epochs,
    batch_size,
    learning_rate,
    embedding_dim,
    seed,
    path_to_store_single_run,
    scoring_technique,
    optim,
    ):

    print("scoring_technique: ", scoring_technique)

    args = get_default_arguments(description=[
        "--dataset_dir", dataset_folder,
        "--num_epochs", num_epochs,
        "--model", model,
        "--batch_size", batch_size,
        "--lr", learning_rate,
        "--embedding_dim", embedding_dim,
        "--scoring_technique", scoring_technique,
        "--optim", optim,
        "--save_embeddings_as_csv",
        "--path_to_store_single_run", path_to_store_single_run,
        "--random_seed", str(seed),
        "--eval_model", "test",
    ])
    result = Execute(args=args).start()

    return result
