"""Quality regression tests for PyKEEN-wrapped models trained with the
NegSampleMargin (margin-based ranking loss) scoring technique.

Unlike tests/test_neg_sample_margin_model_coverage.py (a trainability smoke
test across every model), these assert actual MRR/H@k bounds calibrated
from real runs on KGs/UMLS, following the style of
tests/test_regression_distmult.py.
"""
import pytest

from dicee.config import Namespace
from dicee.executer import Execute


def _pykeen_margin_args(model_name):
    args = Namespace()
    args.dataset_dir = "KGs/UMLS"
    args.trainer = "torchCPUTrainer"
    args.model = model_name
    args.num_epochs = 20
    args.batch_size = 256
    args.lr = 0.1
    args.num_workers = 1
    args.num_core = 1
    args.scoring_technique = "NegSampleMargin"
    args.neg_ratio = 5
    args.margin = 1.0
    args.sample_triples_ratio = None
    args.read_only_few = None
    args.num_folds_for_cv = None
    args.eval_model = "train_val_test"
    return args


class TestPykeenNegSampleMargin:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_distmult(self):
        args = _pykeen_margin_args("Pykeen_DistMult")
        result = Execute(args).start()
        assert 0.80 >= result["Train"]["MRR"] >= 0.55
        assert 0.72 >= result["Val"]["MRR"] >= 0.45
        assert 0.72 >= result["Test"]["MRR"] >= 0.45

        assert result["Train"]["H@10"] >= result["Train"]["H@3"] >= result["Train"]["H@1"]
        assert result["Val"]["H@10"] >= result["Val"]["H@3"] >= result["Val"]["H@1"]
        assert result["Test"]["H@10"] >= result["Test"]["H@3"] >= result["Test"]["H@1"]

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_transh(self):
        args = _pykeen_margin_args("Pykeen_TransH")
        result = Execute(args).start()
        assert 0.60 >= result["Train"]["MRR"] >= 0.30
        assert 0.55 >= result["Val"]["MRR"] >= 0.25
        assert 0.55 >= result["Test"]["MRR"] >= 0.25

        assert result["Train"]["H@10"] >= result["Train"]["H@3"] >= result["Train"]["H@1"]
        assert result["Val"]["H@10"] >= result["Val"]["H@3"] >= result["Val"]["H@1"]
        assert result["Test"]["H@10"] >= result["Test"]["H@3"] >= result["Test"]["H@1"]

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_transe(self):
        args = _pykeen_margin_args("Pykeen_TransE")
        result = Execute(args).start()
        assert 0.80 >= result["Train"]["MRR"] >= 0.50
        assert 0.70 >= result["Val"]["MRR"] >= 0.40
        assert 0.70 >= result["Test"]["MRR"] >= 0.40

        assert result["Train"]["H@10"] >= result["Train"]["H@3"] >= result["Train"]["H@1"]
        assert result["Val"]["H@10"] >= result["Val"]["H@3"] >= result["Val"]["H@1"]
        assert result["Test"]["H@10"] >= result["Test"]["H@3"] >= result["Test"]["H@1"]
