import pytest
import torch

from dicee.config import Namespace
from dicee.sanity_checkers import sanity_check_callback_args


def make_args(trainer, **overrides):
    args = Namespace()
    args.trainer = trainer
    args.adaptive_swa = True
    args.path_to_store_single_run = "Experiments/aswa_guard"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def set_gpu_count(monkeypatch, count):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


class TestAdaptiveSWAGuard:
    def test_rejected_with_tp(self, monkeypatch):
        set_gpu_count(monkeypatch, 2)
        with pytest.raises(ValueError, match="adaptive_swa"):
            sanity_check_callback_args(make_args("TP"))

    def test_rejected_with_pl_on_multiple_gpus(self, monkeypatch):
        set_gpu_count(monkeypatch, 2)
        with pytest.raises(ValueError, match="adaptive_swa"):
            sanity_check_callback_args(make_args("PL"))

    @pytest.mark.parametrize("gpus", [0, 1])
    def test_allowed_with_pl_on_at_most_one_gpu(self, monkeypatch, gpus):
        set_gpu_count(monkeypatch, gpus)
        sanity_check_callback_args(make_args("PL"))

    def test_allowed_with_cpu_trainer(self, monkeypatch):
        set_gpu_count(monkeypatch, 2)
        sanity_check_callback_args(make_args("torchCPUTrainer"))

    def test_other_callbacks_unaffected_on_multiple_gpus(self, monkeypatch):
        set_gpu_count(monkeypatch, 2)
        sanity_check_callback_args(make_args("PL", adaptive_swa=False, swa=True))
