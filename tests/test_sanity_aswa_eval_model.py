import pytest

from dicee.config import Namespace
from dicee.sanity_checkers import sanity_check_callback_args


def make_args(**overrides):
    args = Namespace()
    args.trainer = "torchCPUTrainer"
    args.adaptive_swa = True
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestAdaptiveSWARequiresValidation:
    @pytest.mark.parametrize("eval_model", ["None", None])
    def test_rejected_without_evaluation(self, eval_model):
        with pytest.raises(ValueError, match="validation data"):
            sanity_check_callback_args(make_args(eval_model=eval_model))

    @pytest.mark.parametrize("eval_model", ["train", "val", "train_val", "test", "train_val_test"])
    def test_other_eval_models_allowed(self, eval_model):
        sanity_check_callback_args(make_args(eval_model=eval_model))

    def test_other_callbacks_unaffected(self):
        sanity_check_callback_args(make_args(adaptive_swa=False, swa=True, eval_model="None"))
