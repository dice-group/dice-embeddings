from executer import run_dicee_eval


DB = "UMLS"
MODEL = "DeCaL"


res = run_dicee_eval(
        dataset_folder="./UMLS/sub",
        model=MODEL,
        num_epochs="100",
        batch_size="1024",
        learning_rate="0.1",
        embedding_dim="32",
        loss_function="BCELoss",
    )




res_c = run_dicee_eval(
        dataset_folder="./UMLS/clean",
        model=MODEL,
        num_epochs="100",
        batch_size="1024",
        learning_rate="0.1",
        embedding_dim="32",
        loss_function="BCELoss",
    )

print("clean: ", res_c['Test']['MRR'])
print("poisoned: ", res['Test']['MRR'])