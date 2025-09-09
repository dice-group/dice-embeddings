from executer import run_dicee_eval
from pathlib import Path
import shutil
import csv
import random
from utils import  visualize_results


percentages = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
res_random = []

for idx, ratio in enumerate(percentages):

    dataset_folder_path = f"./Datasets_Perturbed/UMLS/{ratio}/"
    print(dataset_folder_path)

    result_random_poisoned = run_dicee_eval(
                dataset_folder=dataset_folder_path,
                model="Keci",
                num_epochs="100",
                batch_size="1024",
                learning_rate="0.1",
                embedding_dim="32",
                loss_function="BCELoss",
                seed=42,
                path_to_store_single_run=f"saved_models/UMLS/Keci",
                scoring_technique="KvsAll",
                optim="Adam",
            )
    res_random.append(f"{result_random_poisoned['Test']['MRR']}")

    rows = [
        ("triple injection ratios", percentages),
        ("random", res_random),
    ]

    out_path = Path(
        f"sanity_check.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(f"sanity_check.csv", "w", newline="") as file:
            writer = csv.writer(file)
            for name, values in rows:
                writer.writerow([name] + values)


visualize_results(
            f"sanity_check.csv",
            f"sanity_check.png",
            f"sanity_check")