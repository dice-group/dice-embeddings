from dicee.executer import ContinuousExecute
import argparse
def argparse_default(description=None):
    parser = argparse.ArgumentParser(add_help=False)
    # Dataset and storage related
    parser.add_argument("--path_experiment_folder", type=str, default="Experiments/2023-01-07 18:44:47.703307",
                        help="The path of a folder containing pretrained model")
    parser.add_argument("--num_epochs", type=int, default=1, help='Number of epochs for training.')
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_core", type=int, default=None, help='Number of cores to be used.')
    parser.add_argument('--scoring_technique', default=None, help="KvsSample, 1vsAll, KvsAll, NegSample")
    parser.add_argument('--neg_ratio', type=int, default=None,
                        help='The number of negative triples generated per positive triple.')
    parser.add_argument('--optim', type=str, default=None, help='[NAdam, Adam, SGD]')
    parser.add_argument('--batch_size', type=int, default=None, help='Mini batch size')
    parser.add_argument("--seed_for_computation", type=int, default=0, help='Seed for all, see pl seed_everything().')
    parser.add_argument("--trainer", type=str, default=None,
                        help='PL (pytorch lightning trainer), torchDDP (custom ddp), torchCPUTrainer (custom cpu only)')
    if description is None:
        return parser.parse_args()
    return parser.parse_args(description)


if __name__ == '__main__':
    args = argparse_default()
    ContinuousExecute(args).continual_start()
