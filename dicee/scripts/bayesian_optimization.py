import json
from dicee.executer import Execute
import argparse
import optuna
from functools import partial
from optuna.visualization import plot_parallel_coordinate, plot_contour, plot_edf, plot_optimization_history
import os

def objective(trial, model, dataset, loss):

    parser = argparse.ArgumentParser(add_help=False)

    dataset = dataset
    model = model

    num_epochs = 1

    embedding_dim = 32 #trial.suggest_categorical("embedding_dim", [32, 64])
    optimizer = "Adam" #trial.suggest_categorical("optimizer", ["Adam", "Adopt"])
    batch_size = 1024 #trial.suggest_categorical("batch_size", [512, 1024])
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)

    label_relaxation_alpha = trial.suggest_float("label_relaxation_alpha", 0.01, 0.1,) if loss == "LRLoss" else 0.0
    label_smoothing_rate = trial.suggest_float("label_smoothing_rate", 0.01, 0.1)  if loss == "LS" else 0.0

    parser.add_argument('--loss_fn', type=str, default=loss)
    parser.add_argument("--label_smoothing_rate", type=float, default=label_smoothing_rate)
    parser.add_argument('--label_relaxation_alpha', type=float, default=label_relaxation_alpha)
    parser.add_argument("--lr", type=float, default=learning_rate)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument("--dataset_dir", type=str, default=dataset)
    parser.add_argument("--model", type=str, default=model)
    parser.add_argument('--embedding_dim', type=int, default=embedding_dim)
    parser.add_argument("--num_epochs", type=int, default=num_epochs)
    parser.add_argument('--optim', type=str, default=optimizer)

    parser.add_argument('--num_folds_for_cv', type=int, default=0)
    parser.add_argument("--backend", type=str, default="pandas")
    parser.add_argument("--sample_triples_ratio", type=float, default=None)
    parser.add_argument("--init_param", type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--scoring_technique', default="KvsAll")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--eval_model", type=str, default="train_val_test")
    parser.add_argument("--add_noise_rate", type=float, default=0.0)
    parser.add_argument("--sparql_endpoint", type=str, default=None)
    parser.add_argument("--path_single_kg", type=str, default=None)
    parser.add_argument("--normalization", type=str, default="None")
    parser.add_argument("--path_to_store_single_run", type=str, default=None)
    parser.add_argument("--storage_path", type=str, default='Experiments')
    parser.add_argument("--byte_pair_encoding", action="store_true")
    parser.add_argument("--read_only_few", type=int, default=None)
    parser.add_argument("--separator", type=str, default="\s+")
    parser.add_argument("--num_core", type=int, default=0)
    parser.add_argument("--adaptive_swa", action="store_true")
    parser.add_argument("--swa", action="store_true")
    parser.add_argument('--callbacks', type=json.loads, default={})
    parser.add_argument("--trainer", type=str, default='PL')
    parser.add_argument('--neg_ratio', type=int, default=2)
    parser.add_argument('--r', type=int, default=0)
    parser.add_argument('--block_size', type=int, default=8)
    parser.add_argument("--auto_batch_finding", action="store_true")
    parser.add_argument('--degree', type=int, default=0)
    parser.add_argument("--save_embeddings_as_csv", action="store_true")
    parser.add_argument('--pykeen_model_kwargs', type=json.loads, default={})

    args = parser.parse_args()
    result = Execute(args=args).start()

    return result["Test"]["MRR"]

# set according to your environment TODO: make it as a parameter
main_math = "../../../KGs/Datasets_Perturbed/"
report_folder_name = "./bo_outputs/"
report_file_name = "bayesian_optimization_report.txt"

datasets = ["UMLS", "KINSHIP", "NELL-995-h100", "WN18RR", "FB15k-237"]
models = ["Keci", "Pykeen_MuRE", "QMult", "Pykeen_DistMult", "Pykeen_ComplEx", "Pykeen_RotatE", "Pykeen_BoxE"]
losses = ["LRLoss", "LS"]

number_of_runs = 1

for dataset in datasets:
    for model in models:
        for loss in losses:
            dataset_path = main_math + dataset + "/0.0"

            study = optuna.create_study(direction="maximize")

            objective_with_params = partial(objective, dataset=dataset_path, model=model, loss=loss)
            study.optimize(objective_with_params, n_trials=number_of_runs)

            best_trial = study.best_trial

            fig1 = plot_parallel_coordinate(study)
            fig1.write_image(report_folder_name + f"parallel_coordinate-{dataset}-{model}-{loss}"+ ".png")

            fig3 = plot_edf(study)
            fig3.write_image(report_folder_name + f"plot_edf-{dataset}-{model}-{loss}" + ".png")

            fig4 = plot_optimization_history(study)
            fig4.write_image(report_folder_name + f"plot_optimization_history-{dataset}-{model}-{loss}" + ".png")
            
            if loss == "LRLoss":
                fig2 = plot_contour(study, params=["label_relaxation_alpha", "learning_rate"])
                fig2.write_image(report_folder_name + f"contour-{dataset}-{model}-{loss}" + ".png")

            if loss == "LS":
                fig2 = plot_contour(study, params=["label_smoothing_rate", "learning_rate"])
                fig2.write_image(report_folder_name + f"contour-{dataset}-{model}-{loss}" + ".png")


            os.makedirs(os.path.dirname(report_folder_name), exist_ok=True)
            with open(report_folder_name + report_file_name, "a") as file:
                file.write(f"Value: {best_trial.value}, Params: {best_trial.params}, Dataset: {dataset}, Model: {model}, Loss: {loss} \n")


0
