from dicee.executer import Execute
from dicee.scripts.run import get_default_arguments

def run_dicee_eval(
    dataset_folder,
    model,
    num_epochs,
    batch_size,
    learning_rate,
    embedding_dim,
    loss_function,
    ):

    args = get_default_arguments(description=[
        "--dataset_dir", dataset_folder,
        "--num_epochs", num_epochs,
        "--model", model,
        "--batch_size", batch_size,
        "--lr", learning_rate,
        "--embedding_dim", embedding_dim,
        "--loss_fn", loss_function,
        "--save_embeddings_as_csv",
    ])
    result = Execute(args=args).start()

    return result
