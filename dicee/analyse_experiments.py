import os
import json
import pandas as pd
import argparse


def get_default_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dir", type=str, default=None, help="Path of a directory containing experiments")
    return parser.parse_args()


class Experiment:
    def __init__(self):
        self.model_name = []
        self.callbacks = []
        self.embedding_dim = []
        self.num_params = []
        self.num_epochs = []
        self.batch_size = []
        self.lr = []
        self.byte_pair_encoding = []
        self.aswa = []
        self.path_dataset_folder = []
        self.full_storage_path = []
        self.pq = []
        self.train_mrr = []
        self.train_h1 = []
        self.train_h3 = []
        self.train_h10 = []

        self.val_mrr = []
        self.val_h1 = []
        self.val_h3 = []
        self.val_h10 = []

        self.test_mrr = []
        self.test_h1 = []
        self.test_h3 = []
        self.test_h10 = []

        self.runtime = []
        self.normalization = []
        self.scoring_technique = []

    def save_experiment(self, x):
        self.model_name.append(x['model'])
        self.embedding_dim.append(x['embedding_dim'])
        self.num_epochs.append(x['num_epochs'])
        self.batch_size.append(x['batch_size'])
        self.lr.append(x['lr'])

        self.byte_pair_encoding.append(x["byte_pair_encoding"])
        self.aswa.append(x["adaptive_swa"])
        self.path_dataset_folder.append(x['dataset_dir'])
        self.pq.append((x['p'], x['q']))
        self.runtime.append(x['Runtime'])
        self.num_params.append(x['NumParam'])

        self.normalization.append(x['normalization'])
        self.scoring_technique.append(x['scoring_technique'])
        self.callbacks.append(x['callbacks'])

        self.train_mrr.append(x['Train']['MRR'])
        self.train_h1.append(x['Train']['H@1'])
        self.train_h3.append(x['Train']['H@3'])
        self.train_h10.append(x['Train']['H@10'])

        # Partition by delim and take the last one
        # self.full_storage_path.append(x['full_storage_path'].partition('dice-embeddings')[-1])

        self.val_mrr.append(x['Val']['MRR'])
        self.val_h1.append(x['Val']['H@1'])
        self.val_h3.append(x['Val']['H@3'])
        self.val_h10.append(x['Val']['H@10'])

        self.test_mrr.append(x['Test']['MRR'])
        self.test_h1.append(x['Test']['H@1'])
        self.test_h3.append(x['Test']['H@3'])
        self.test_h10.append(x['Test']['H@10'])

    def to_df(self):
        return pd.DataFrame(
            dict(model=self.model_name,
                 byte_pair_encoding=self.byte_pair_encoding,
                 aswa=self.aswa,
                 Dataset=self.path_dataset_folder,
                 trainMRR=self.train_mrr,
                 trainH1=self.train_h1,
                 trainH3=self.train_h3,
                 trainH10=self.train_h10,
                 num_epochs=self.num_epochs,
                 full_storage_path=self.full_storage_path,
                 valMRR=self.val_mrr,
                 valH1=self.val_h1,
                 valH3=self.val_h3,
                 valH10=self.val_h10,
                 testMRR=self.test_mrr,
                 testH1=self.test_h1,
                 testH3=self.test_h3,
                 testH10=self.test_h10,
                 runtime=self.runtime,
                 params=self.num_params,
                 callbacks=self.callbacks,
                 embeddingdim=self.embedding_dim,
                 scoring_technique=self.scoring_technique
                 )
        )


def analyse(args):
    # (2) Get all subfolders
    sub_folder_str_paths = os.listdir(args.dir)
    experiments = []
    for path in sub_folder_str_paths:
        if path=="summary.csv":
            continue
        full_path = args.dir + "/" + path
        with open(f'{full_path}/configuration.json', 'r') as f:
            config = json.load(f)
            """
            config = {i: config[i] for i in
                      ['model', 'dataset_dir', 'embedding_dim',
                       'normalization', 'num_epochs', 'batch_size', 'lr',
                       'callbacks',
                       'adaptive_swa',
                       'scoring_technique',
                       "byte_pair_encoding",
                       'dataset_dir', 'p', 'q']}
            """
        try:
            with open(f'{full_path}/report.json', 'r') as f:
                report = json.load(f)
                report = {i: report[i] for i in ['Runtime', 'NumParam']}
            with open(f'{full_path}/eval_report.json', 'r') as f:
                eval_report = json.load(f)
        except FileNotFoundError:
            print("NOT found")
            continue
        config.update(eval_report)
        config.update(report)
        if "Train" in config:
            for k, v in config["Train"].items():
                config[f"train{k}"] = v

        if "Val" in config:
            for k, v in config["Val"].items():
                config[f"val{k}"] = v

        if "Test" in config:
            for k, v in config["Test"].items():
                config[f"test{k}"] = v

        del config["Train"]
        del config["Val"]
        del config["Test"]
        experiments.append(config)

    df = pd.DataFrame(experiments)
    df.sort_values(by=['testMRR'], ascending=False, inplace=True)
    pd.set_option("display.precision", 3)

    df=df[["model",
            "adaptive_swa",
               "stochastic_weight_avg",
              "trainMRR",
              "trainH@1",
              "trainH@3",
              "trainH@10",
              #"valMRR",
              #"valH@1",
              #"valH@3",
              #"valH@10",
              "testMRR",
              "testH@1",
              "testH@3",
              "testH@10",
              #'dataset_dir', 'embedding_dim',
              #'num_epochs', 'batch_size', 'lr',

              ]]
    print(df)
    print(df.to_latex(index=False, float_format="%.3f"))
    path_to_save = args.dir + '/summary.csv'
    df.to_csv(path_or_buf=path_to_save)
    print(f"Saved in {path_to_save}")


if __name__ == '__main__':
    analyse(get_default_arguments())
