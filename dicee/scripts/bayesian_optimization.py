import json
from dicee.executer import Execute
import argparse
import optuna
from functools import partial
from optuna.visualization import plot_parallel_coordinate
import os
import numpy as np

def objective(trial, model, dataset, loss):

    parser = argparse.ArgumentParser(add_help=False)

    dataset = dataset
    model = model

    num_epochs = 100

    embedding_dim = 32 #trial.suggest_categorical("embedding_dim", [32, 64])
    optimizer = "Adam" #trial.suggest_categorical("optimizer", ["Adam", "Adopt"])
    batch_size = 1024 #trial.suggest_categorical("batch_size", [512, 1024])
    learning_rate = 0.1 #trial.suggest_float("learning_rate", 0.01, 0.1)


    if loss == 'AGCELoss':
        agce_a = trial.suggest_float('agce_a', 1e-3, 1.0, log = True)
        agce_q = trial.suggest_float('agce_q', 1e-3, 1.0, log = True)
        parser.add_argument('--agce_a', type=float, default=agce_a)
        parser.add_argument('--agce_q', type=float, default=agce_q)
    
    elif loss == 'AULoss':
        aul_a = trial.suggest_float("aul_a", 1.01, 2.0)
        aul_p = trial.suggest_float("aul_p",1e-4, 1.0)
        parser.add_argument("--aul_a", type=float, default=aul_a)
        parser.add_argument("--aul_p", type=float, default=aul_p)
    
    elif loss == "AELoss":
        a_ael = trial.suggest_float("a_ael", 0.01, 1.0)
        parser.add_argument("--a_ael", type=float, default=a_ael)

    elif loss == "RoBoSS":
        a_roboss = trial.suggest_float("a_roboss", 0.1, 1.0)
        lambda_roboss = trial.suggest_float("lambda_roboss", 1.0, 10.0)
        parser.add_argument("--a_roboss", type=float, default=a_roboss)
        parser.add_argument("--lambda_roboss", type=float, default=lambda_roboss)

    elif loss == "WaveLoss":
        wave_a = trial.suggest_float("wave_a", 0.01, 3.0)
        lambda_param = trial.suggest_float("lambda_param", 0.05, 2.0)
        parser.add_argument("--wave_a", type=float, default=wave_a)
        parser.add_argument("--lambda_param", type=float, default=lambda_param)


    parser.add_argument('--loss_fn', type=str, default=loss)
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
    parser.add_argument("--label_smoothing_rate", type=float, default=0.0)

    args = parser.parse_args()
    result = Execute(args=args).start()

    return result["Val"]["MRR"]

def main():
    # set according to your environment TODO: make it as a parameter
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(script_dir, "../..")
    main_path = os.path.join(root, "Datasets_Perturbed/")

    report_folder_name = "./bo_outputs/512_05_100Epochs/"
    report_file_name = "bayesian_optimization_report.txt"

    datasets = ["WN18RR", "FB15k-237"]  #, "KINSHIP" , "NELL-995-h100", "WN18RR", "FB15k-237"]
    models = ["DistMult", "ComplEx", "Pykeen_MuRE", "Pykeen_RotatE", "QMult"]  # , "Pykeen_MuRE", "QMult", "Pykeen_DistMult", "Pykeen_ComplEx", "Pykeen_BoxE", "Pykeen_RotatE"]
    losses = ['AGCELoss', 'AELoss', 'WaveLoss', 'RoBoSS']  # , 'AUL', 'AEL', 'WaveLoss', 'RoBoSS']

    number_of_runs = 30

    os.makedirs(os.path.dirname(report_folder_name), exist_ok=True)

    for dataset in datasets:
            for model in models:
                for loss in losses:

                    dataset_path = main_path + dataset + "/0.0"
                    study = optuna.create_study(direction="maximize")

                    objective_with_params = partial(objective, dataset=dataset_path, model=model, loss=loss)
                    study.optimize(objective_with_params, n_trials=number_of_runs)

                    best_trial = study.best_trial

                    loss_type = loss
                    fig = plot_parallel_coordinate(study)
                    fig.update_layout(title={"text": f"Dataset: {dataset}, Model: {model}, Softening Method: {loss_type}",
                                             "x": 0.5,
                                             "xanchor": "center",
                                             "y": 0.97,
                                             "yanchor": "top"},
                                       title_font=dict(size=24),
                                       font=dict(size=22),
                                       legend=dict(font=dict(size=4)),
                                       )

                    for dim in fig.data[0].dimensions:
                        if dim['label'] == "Objective Value":
                            dim['label'] = "MRR"
                        dim["label"] = f"<br>{dim['label']}"

                    fig.data[0]['labelangle'] = 0
                    fig.data[0]['labelside'] = 'bottom'
                    fig.data[0]['line']['colorbar']['title']['text'] = ''

                    fig.write_image(report_folder_name + f"parallel_coordinate-{dataset}-{model}-{loss}" + ".png")

                    with open(report_folder_name + report_file_name, "a") as file:
                        file.write(f"Value: {best_trial.value}, "
                                   f"Params: {best_trial.params}, "
                                   f"Dataset: {dataset}, "
                                   f"Model: {model}, "
                                   f"Loss: {loss} "
                                   f"\n")


if __name__ == "__main__":
    main()
