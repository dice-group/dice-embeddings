import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dicee.executer import Execute
from dicee.config import Namespace
import optuna
from functools import partial
from optuna.visualization import plot_parallel_coordinate

def objective(trial, model, dataset, loss):

    num_epochs = 100
    embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    learning_rate = trial.suggest_categorical("learning_rate", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    optimizer = "Adam"

    args = Namespace()
    args.dataset_dir = dataset
    args.model = model
    args.loss_fn = loss
    args.lr = learning_rate
    args.batch_size = batch_size
    args.embedding_dim = embedding_dim
    args.num_epochs = num_epochs
    args.optim = optimizer
    args.trainer = "PL"

    if loss == "AELoss":
        args.a_ael = trial.suggest_float("a_ael", 0.1, 10.0, log=True)

    elif loss == "RoBoSS":
        args.a_roboss = trial.suggest_float("a_roboss", 0.5, 10.0, log=True)
        args.lambda_roboss = trial.suggest_float("lambda_roboss", 0.5, 3.0)

    elif loss == 'AGCELoss':
        args.agce_a = trial.suggest_float('agce_a', 1e-3, 1.0, log=True)
        args.agce_q = trial.suggest_float('agce_q', 1e-3, 1.0, log=True)

    elif loss == "WaveLoss":
        args.wave_a = trial.suggest_float("wave_a", 0.01, 3.0, log=True)
        args.lambda_param = trial.suggest_float("lambda_param", 0.05, 2.0)

    elif loss == "BCELoss":
        pass
        
    result = Execute(args=args).start()
    return result["Val"]["MRR"]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(script_dir, "../..")
    main_path = os.path.join(root, "Datasets_Perturbed/")

    report_folder_name = "./bo_outputs/100_Epochs/"
    report_file_name = "bayesian_optimization_report.txt"

    datasets = ["KINSHIP", "UMLS", "NELL-995-h100", "FB15k-237", "WN18RR"]
    models = ["ComplEx", "DistMult", "DualE", "QMult", "Pykeen_RotatE", "Pykeen_MuRE", "Keci", "Pykeen_TransH"]
    losses = ["AGCELoss", 'AELoss', 'RoBoSS', "WaveLoss", "BCELoss"]

    number_of_runs = 100

    os.makedirs(os.path.dirname(report_folder_name), exist_ok=True)

    for dataset in datasets:
            for model in models:
                for loss in losses:

                    dataset_path = main_path + dataset + "/0.0"
                    study = optuna.create_study(direction="maximize")

                    objective_with_params = partial(objective, dataset=dataset_path, model=model, loss=loss)
                    #same as below
                    # def objective_with_params(trial):
                    #     return objective(trial, dataset=dataset_path, model=model, loss=loss)
                    #partial: Creates a new version of a function with some arguments already "frozen" or pre-filled. 

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

                    fig.write_html(report_folder_name + f"parallel_coordinate-{dataset}-{model}-{loss}" + ".html")

                    with open(report_folder_name + report_file_name, "a") as file:
                        file.write(f"Value: {best_trial.value}, "
                                   f"Params: {best_trial.params}, "
                                   f"Dataset: {dataset}, "
                                   f"Model: {model}, "
                                   f"Loss: {loss} "
                                   f"\n")


if __name__ == "__main__":
    main()
