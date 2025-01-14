import json
from dicee.executer import Execute
import argparse
import optuna

def objective(trial):
    parser = argparse.ArgumentParser(add_help=False)

    dataset = "/home/adel/Documents/dice-embeddings/KGs/Datasets_Perturbed/0_UMLS/0.0/"
    model = "Keci"
    embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64])
    num_epochs = 100
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
    label_smoothing_rate = 0.0
    loss = "LRLoss"
    label_relaxation_alpha = trial.suggest_float("label_relaxation_alpha", 0, 1)
    optimizer = trial.suggest_categorical("optimizer", ['Adam', 'AdamW', 'SGD',"NAdam", "Adagrad", "ASGD", "Adopt"])

    parser.add_argument("--dataset_dir", type=str, default=dataset)
    parser.add_argument("--model", type=str, default=model)
    parser.add_argument('--embedding_dim', type=int, default=embedding_dim)
    parser.add_argument("--num_epochs", type=int, default=num_epochs)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=learning_rate)
    parser.add_argument("--label_smoothing_rate", type=float, default=label_smoothing_rate)
    parser.add_argument('--loss_fn', type=str, default=loss)
    parser.add_argument('--label_relaxation_alpha', type=float, default=label_relaxation_alpha)
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

    args = parser.parse_args()
    result = Execute(args=args).start()

    return result["Test"]["MRR"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best Trial:")
best_trial = study.best_trial

print(f"  Value: {best_trial.value}")
print(f"  Params: {best_trial.params}")