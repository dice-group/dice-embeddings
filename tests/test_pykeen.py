from dicee.executer import Execute
import sys
import pytest
from dicee.config import Namespace


def template(model_name):
    args = Namespace()
    args.dataset_dir = "KGs/UMLS"
    args.trainer = "PL"
    args.model = model_name
    args.num_epochs = 10
    args.batch_size = 256
    args.lr = 0.1
    args.num_workers = 1
    args.num_core = 1
    args.scoring_technique = "KvsAll"
    args.num_epochs = 10
    args.sample_triples_ratio = None
    args.read_only_few = None
    args.num_folds_for_cv = None
    return args


@pytest.mark.parametrize("model_name", ["Pykeen_DistMult", "Pykeen_ComplEx", "Pykeen_HolE", "Pykeen_CP",
                                        "Pykeen_ProjE", "Pykeen_TuckER", "Pykeen_TransR", "Pykeen_TransH",
                                        "Pykeen_TransD", "Pykeen_TransE", "Pykeen_QuatE", "Pykeen_MuRE",
                                        "Pykeen_BoxE", "Pykeen_RotatE"])
class TestClass:
    def test_defaultParameters_case(self, model_name):
        args = template(model_name)
        result = Execute(args).start()
        if args.model == "Pykeen_DistMult":
            assert 0.84 >= result["Train"]["MRR"] >= 0.79
        elif args.model == "Pykeen_ComplEx":
            assert 0.92 >= result["Train"]["MRR"] >= 0.77
        elif args.model == "Pykeen_QuatE":
            assert 0.999 >= result["Train"]["MRR"] >= 0.84
        elif args.model == "Pykeen_MuRE":
            assert 0.89 >= result["Train"]["MRR"] >= 0.85
        elif args.model == "Pykeen_BoxE":
            assert 0.85 >= result["Train"]["MRR"] >= 0.78
        elif args.model == "Pykeen_RotatE":
            assert 0.67 >= result["Train"]["MRR"] >= 0.60
        elif args.model == "Pykeen_CP":  # 1.5M params
            assert 1.00 >= result["Train"]["MRR"] >= 0.99
        elif args.model == "Pykeen_HolE":  # 14.k params
            assert 0.95 >= result["Train"]["MRR"] >= 0.88
        elif args.model == "Pykeen_ProjE":  # 14.k params
            assert 0.88 >= result["Train"]["MRR"] >= 0.78
        elif args.model == "Pykeen_TuckER":  # 276.k params
            assert 0.36 >= result["Train"]["MRR"] >= 0.31
        elif args.model == "Pykeen_TransR":  # 188.k params
            assert 0.72 >= result["Train"]["MRR"] >= 0.66
        elif args.model == "Pykeen_TransF":  # 14.5 k params
            assert 0.17 >= result["Train"]["MRR"] >= 0.16
        elif args.model == "Pykeen_TransH":  # 20.4 k params
            assert 0.69 >= result["Train"]["MRR"] >= 0.58
        elif args.model == "Pykeen_TransD":  # 29.1 k params
            assert 0.73 >= result["Train"]["MRR"] >= 0.60
        elif args.model == "Pykeen_TransE":  # 29.1 k params
            assert 0.45 >= result["Train"]["MRR"] >= 0.15

    def test_perturb_callback_case(self, model_name):
        args = template(model_name)
        args.callbacks = {"Perturb": {"level": "out", "ratio": 0.2, "method": "Soft", "scaler": 0.3}}
        Execute(args).start()
