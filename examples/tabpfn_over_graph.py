"""Train and evaluate the TabPFN-based tabular pipeline with DICE."""

from pprint import pprint

from dicee.config import Namespace
from dicee.executer import TabularExecute


# Configure a minimal TabPFN experiment on the UMLS example dataset.
args = Namespace()
args.tabpfn = True
args.dataset_dir = "KGs/UMLS"
args.path_to_store_single_run = "Experiments/TabPFN_UMLS"
args.random_seed = 1
args.num_core = 1
args.neg_ratio = 1
args.separator = r"\s+"
args.tabpfn_kwargs = {
    "device": "cpu",
    "max_train_samples": 1000,
    "n_estimators": 8,
    "entity_centric": False,
}

# Run the tabular pipeline and print the stored report fields.
result = TabularExecute(args).start()
# pprint(result)
