""" This script should be moved to dicee/scripts"""
import os
import json
import pandas as pd
import argparse


def get_default_arguments():
    """
    Returns the default arguments for the script.

    Returns
    -------
    args : argparse.Namespace
        The namespace containing the default argument values.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dir", type=str, default=None, help="Path of a directory containing experiments")
    parser.add_argument('--features', nargs='+', default=[])
    # TODO: features/columns for pandas dataframe
    return parser.parse_args()


class Experiment:
    """
    A class to store and manage data from experiments.

    Attributes
    ----------
    model_name : list
        A list storing the names of the models used in experiments.
    embedding_dim : list
        Dimensions of embeddings used in experiments.
    num_params : list
        The number of parameters in the models.
    num_epochs : list
        Number of epochs training was run for each experiment.
    batch_size : list
        Batch sizes used in experiments.
    lr : list
        Learning rates used in experiments.
    byte_pair_encoding : list
        Indicates whether byte pair encoding was used.
    aswa : list
        Indicates whether adaptive SWA was used.
    path_dataset_folder : list
        Paths to dataset folders used in experiments.
    pq : list
        P and Q parameters used in experiments, for models that require them.
    train_mrr, train_h1, train_h3, train_h10 : list
        Training metrics: Mean Reciprocal Rank, Hits@1, Hits@3, and Hits@10.
    val_mrr, val_h1, val_h3, val_h10 : list
        Validation metrics.
    test_mrr, test_h1, test_h3, test_h10 : list
        Test metrics.
    runtime : list
        Runtime of each experiment.
    normalization : list
        Indicates whether normalization was applied.
    scoring_technique : list
        Scoring techniques used in experiments.
    callbacks : list
        Callbacks used in experiments.

    Methods
    -------
    save_experiment(x: dict)
        Saves the data from a single experiment into the class's attributes.
    to_df() -> pd.DataFrame
        Converts the accumulated experiment data into a pandas DataFrame.
    """

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

    def save_experiment(self, x: dict):
        """
        Saves the data from a single experiment into the class's attributes.

        Parameters
        ----------
        x : dict
            A dictionary containing the data from a single experiment.
        """
        self.model_name.append(x["model"])
        self.embedding_dim.append(x["embedding_dim"])
        self.num_epochs.append(x["num_epochs"])
        self.batch_size.append(x["batch_size"])
        self.lr.append(x["lr"])

        self.byte_pair_encoding.append(x["byte_pair_encoding"])
        self.aswa.append(x["adaptive_swa"])
        self.path_dataset_folder.append(x["dataset_dir"])
        self.pq.append((x["p"], x["q"]))
        self.runtime.append(x["Runtime"])
        self.num_params.append(x["NumParam"])

        self.normalization.append(x["normalization"])
        self.scoring_technique.append(x["scoring_technique"])
        self.callbacks.append(x["callbacks"])

        self.train_mrr.append(x["Train"]["MRR"])
        self.train_h1.append(x["Train"]["H@1"])
        self.train_h3.append(x["Train"]["H@3"])
        self.train_h10.append(x["Train"]["H@10"])

        # Partition by delim and take the last one
        # self.full_storage_path.append(x['full_storage_path'].partition('dice-embeddings')[-1])

        self.val_mrr.append(x["Val"]["MRR"])
        self.val_h1.append(x["Val"]["H@1"])
        self.val_h3.append(x["Val"]["H@3"])
        self.val_h10.append(x["Val"]["H@10"])

        self.test_mrr.append(x["Test"]["MRR"])
        self.test_h1.append(x["Test"]["H@1"])
        self.test_h3.append(x["Test"]["H@3"])
        self.test_h10.append(x["Test"]["H@10"])

    def to_df(self) -> pd.DataFrame:
        """
        Converts the accumulated experiment data into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the accumulated data from all experiments.
        """
        return pd.DataFrame(
            dict(
                model=self.model_name,
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
                scoring_technique=self.scoring_technique,
            )
        )


def analyse(args):
    """
    Analyzes and summarizes the results of experiments stored in subdirectories.

    This function reads configurations and evaluation reports from each experiment,
    compiles a summary, and saves it to a CSV file. It also prints the summary
    and its LaTeX format to the console.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments passed to the script. Expected to contain:
        - args.dir: The directory containing subdirectories of experiments.

    Returns
    -------
    None
    """
    # (2) Get all subfolders
    sub_folder_str_paths = os.listdir(args.dir)
    experiments = []
    for path in sub_folder_str_paths:
        full_path = args.dir + "/" + path
        if os.path.isdir(full_path) is False:
            continue

        
        with open(f'{full_path}/configuration.json', 'r') as f:
            config = json.load(f)
            
        try:
            with open(f'{full_path}/report.json', 'r') as f:
                report = json.load(f)
                report = {i: report[i] for i in ["Runtime", "NumParam"]}
            with open(f"{full_path}/eval_report.json", "r") as f:
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
    df.sort_values(by=["testMRR"], ascending=False, inplace=True)
    pd.set_option("display.precision", 3)
    
    #features=["model","testMRR"]
    # print(df.columns)
    try:
        df_features = df[args.features]
    except:
        print(f"--features ({args.features}) is not a subset of {df.columns}")
        exit(1)

    print(df_features.to_latex(index=False, float_format="%.3f"))
    path_to_save = args.dir + '/summary.csv'
    df_features.to_csv(path_or_buf=path_to_save)
    print(f"Saved in {path_to_save}")


if __name__ == "__main__":
    analyse(get_default_arguments())
