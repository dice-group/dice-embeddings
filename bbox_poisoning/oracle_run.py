from executer import run_dicee_eval

DB = "./KGs/UMLS"
MODEL = "DeCaL"

result_random_poisoned = run_dicee_eval(
    dataset_folder=DB,
    model=MODEL,
    num_epochs="100",
    batch_size="1024",
    learning_rate="0.1",
    embedding_dim="32",
    loss_function="BCELoss",
)

print(f"{result_random_poisoned['Test']['MRR']:.4f}")