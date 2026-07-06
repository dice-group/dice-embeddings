import pytest
import torch

from dicee.config import Namespace
from dicee.executer import Execute
from dicee.trainer.auto_batch_finder import find_good_batch_size


def make_mock_loader(batch_size=32, dataset_size=1000):
    """Create a minimal mock DataLoader for testing."""
    from unittest.mock import MagicMock
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=dataset_size)
    dataset.collate_fn = None
    loader = MagicMock()
    loader.batch_size = batch_size
    loader.dataset = dataset
    loader.num_workers = 0
    return loader


def dummy_training_step(batch):
    """A no-op training step that returns a scalar loss."""
    return 0.5


def make_training_args(*, trainer, auto_batch_finding):
    args = Namespace()
    args.model = "DistMult"
    args.scoring_technique = "KvsAll"
    args.optim = "Adam"
    args.dataset_dir = "KGs/UMLS"
    args.num_epochs = 1
    args.batch_size = 32
    args.lr = 0.1
    args.embedding_dim = 32
    args.input_dropout_rate = 0.0
    args.hidden_dropout_rate = 0.0
    args.feature_map_dropout_rate = 0.0
    args.auto_batch_finding = auto_batch_finding
    args.read_only_few = None
    args.sample_triples_ratio = None
    args.num_folds_for_cv = None
    args.backend = "pandas"
    args.trainer = trainer
    args.normalization = None
    return args


class TestAutoBatchFinderEndToEnd:
    """End-to-end integration tests using the real model pipeline."""

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_auto_batch_finding_true_cpu_trainer(self):
        """End-to-end: auto_batch_finding=True with torchCPUTrainer runs without error."""
        args = make_training_args(trainer="torchCPUTrainer", auto_batch_finding=True)
        result = Execute(args).start()
        assert result is not None
        assert "Train" in result

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize("trainer", ["torchCPUTrainer", "PL"])
    def test_auto_batch_finding_false_supported_trainers(self, trainer):
        """End-to-end: auto_batch_finding=False runs without error."""
        args = make_training_args(trainer=trainer, auto_batch_finding=False)
        result = Execute(args).start()
        assert result is not None
        assert "Train" in result

    def test_auto_batch_finding_false_torch_ddp_requires_torchrun(self, monkeypatch):
        """torchDDP must be launched with torchrun, even when auto batch finding is disabled."""
        for env_var in ("LOCAL_RANK", "RANK", "WORLD_SIZE"):
            monkeypatch.delenv(env_var, raising=False)

        args = make_training_args(trainer="torchDDP", auto_batch_finding=False)
        with pytest.raises(RuntimeError, match="torchDDP trainer must be launched with torchrun"):
            Execute(args)


class TestFindGoodBatchSizeCPU:
    """Unit tests using real DataLoader on CPU — no GPU required."""

    def test_cpu_skips_batch_finding_and_returns_initial(self):
        """On CPU, batch finding must be skipped and initial batch size returned."""
        loader = make_mock_loader(batch_size=32, dataset_size=1000)
        device = torch.device("cpu")
        result_bs, result_rt = find_good_batch_size(loader, dummy_training_step, device)
        assert result_bs == 32
        assert result_rt is None

    def test_cpu_string_device_also_skips(self):
        """Passing device as a string 'cpu' should also skip gracefully."""
        loader = make_mock_loader(batch_size=16, dataset_size=500)
        result_bs, result_rt = find_good_batch_size(loader, dummy_training_step, "cpu")
        assert result_bs == 16
        assert result_rt is None

    def test_batch_size_already_covers_full_dataset(self):
        """If batch_size >= dataset_size, return dataset_size immediately."""
        loader = make_mock_loader(batch_size=2000, dataset_size=1000)
        device = torch.device("cpu")
        result_bs, result_rt = find_good_batch_size(loader, dummy_training_step, device)
        assert result_bs == 1000
        assert result_rt is None

    def test_batch_size_exactly_equals_dataset_size(self):
        """Edge case: batch_size == dataset_size should return dataset_size."""
        loader = make_mock_loader(batch_size=500, dataset_size=500)
        device = torch.device("cpu")
        result_bs, result_rt = find_good_batch_size(loader, dummy_training_step, device)
        assert result_bs == 500
        assert result_rt is None

    def test_returns_tuple_of_two(self):
        """Return value must always be a 2-tuple (batch_size, runtime_or_None)."""
        loader = make_mock_loader(batch_size=32, dataset_size=1000)
        device = torch.device("cpu")
        result = find_good_batch_size(loader, dummy_training_step, device)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_batch_size_returned_is_positive_integer(self):
        """Returned batch size must be a positive integer."""
        loader = make_mock_loader(batch_size=32, dataset_size=1000)
        device = torch.device("cpu")
        result_bs, _ = find_good_batch_size(loader, dummy_training_step, device)
        assert isinstance(result_bs, int)
        assert result_bs > 0

class TestFindGoodBatchSizeInvalidDevice:
    """Tests for graceful handling of unusual or invalid device inputs."""

    def test_invalid_device_string_raises_runtime_error(self):
        """Passing an invalid device string must raise RuntimeError specifically."""
        loader = make_mock_loader(batch_size=32, dataset_size=1000)
        with pytest.raises(RuntimeError):
            find_good_batch_size(loader, dummy_training_step, "not_a_real_device")

    def test_none_device_raises_type_error(self):
        """Passing None as device must raise TypeError specifically."""
        loader = make_mock_loader(batch_size=32, dataset_size=1000)
        with pytest.raises(TypeError):
            find_good_batch_size(loader, dummy_training_step, None)


class TestFindGoodBatchSizeEdgeCases:
    """Edge cases around dataset size and batch size boundaries."""

    def test_small_dataset_one_sample(self):
        """Dataset with 1 sample: batch_size >= dataset_size, return 1 immediately."""
        loader = make_mock_loader(batch_size=32, dataset_size=1)
        device = torch.device("cpu")
        result_bs, result_rt = find_good_batch_size(loader, dummy_training_step, device)
        assert result_bs == 1
        assert result_rt is None

    def test_batch_size_one(self):
        """Minimum batch size of 1 on CPU should be returned unchanged."""
        loader = make_mock_loader(batch_size=1, dataset_size=1000)
        device = torch.device("cpu")
        result_bs, result_rt = find_good_batch_size(loader, dummy_training_step, device)
        assert result_bs == 1
        assert result_rt is None

    def test_large_batch_size_capped_at_dataset_size(self):
        """Batch size should never exceed dataset size in the return value."""
        loader = make_mock_loader(batch_size=99999, dataset_size=100)
        device = torch.device("cpu")
        result_bs, _ = find_good_batch_size(loader, dummy_training_step, device)
        assert result_bs <= 100

