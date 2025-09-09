from dicee.executer import Execute
from dicee.scripts.run import get_default_arguments
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_dicee_eval(
    dataset_folder,
    model,
    num_epochs,
    batch_size,
    learning_rate,
    embedding_dim,
    loss_function,
    seed,
    path_to_store_single_run,
    scoring_technique="KvsAll",
    optim="Adam",
    ):

    args = get_default_arguments(description=[
        "--dataset_dir", dataset_folder,
        "--num_epochs", num_epochs,
        "--model", model,
        "--batch_size", batch_size,
        "--lr", learning_rate,
        "--embedding_dim", embedding_dim,
        "--loss_fn", loss_function,
        "--scoring_technique", "KvsAll",
        "--optim", "Adam",
        "--save_embeddings_as_csv",
        "--path_to_store_single_run", path_to_store_single_run,
        "--random_seed", str(seed),
    ])
    result = Execute(args=args).start()

    return result
