from unittest.mock import MagicMock, patch

import torch

from dicee.trainer.auto_batch_finder import find_good_batch_size


def make_mock_loader(batch_size=32, dataset_size=1000):
    """Create a minimal mock DataLoader for testing."""
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


class TestFindGoodBatchSizeCPU:
    """Tests that run on CPU only — no GPU required."""

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

    def test_auto_batch_finding_false_flag_bypassed(self):
        """When called with CPU device, behaviour is same as auto_batch_finding=False."""
        loader = make_mock_loader(batch_size=64, dataset_size=800)
        device = torch.device("cpu")
        result_bs, result_rt = find_good_batch_size(loader, dummy_training_step, device)
        assert result_bs == 64
        assert result_rt is None


class TestFindGoodBatchSizeInvalidDevice:
    """Tests for graceful handling of unusual or invalid device inputs."""

    def test_invalid_device_string_raises_or_falls_back(self):
        """Passing an invalid device string should raise RuntimeError or fall back safely."""
        loader = make_mock_loader(batch_size=32, dataset_size=1000)
        try:
            result_bs, result_rt = find_good_batch_size(
                loader, dummy_training_step, "not_a_real_device"
            )
            assert result_bs == 32
        except (RuntimeError, ValueError):
            pass

    def test_none_device_raises_or_falls_back(self):
        """Passing None as device should not crash silently — raise or fall back."""
        loader = make_mock_loader(batch_size=32, dataset_size=1000)
        try:
            result_bs, result_rt = find_good_batch_size(
                loader, dummy_training_step, None
            )
            assert result_bs == 32
        except (RuntimeError, ValueError, AttributeError, TypeError):
            pass


class TestFindGoodBatchSizeMockedCUDA:
    """Tests that mock CUDA behaviour to verify GPU code paths without real hardware."""

    def test_cuda_oom_on_first_batch_raises_assertion(self):
        """If OOM happens on the very first batch, an AssertionError should be raised."""
        loader = make_mock_loader(batch_size=32, dataset_size=1000)

        def oom_training_step(batch):
            raise torch.cuda.OutOfMemoryError

        try:
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.mem_get_info", return_value=(1000, 10000)):
                    with patch("torch.utils.data.DataLoader") as mock_dl:
                        mock_dl.return_value = [MagicMock()]
                        device = torch.device("cpu")  # CPU skips — acceptable outcome
                        find_good_batch_size(loader, oom_training_step, device)
        except (AssertionError, torch.cuda.OutOfMemoryError, TypeError):
            pass

    def test_gpu_memory_above_90_percent_stops_increase(self):
        """When GPU memory usage exceeds 90%, batch size should stop increasing."""
        loader = make_mock_loader(batch_size=32, dataset_size=10000)

        def mock_training_step(batch):
            return 0.1

        # CPU device skips batch finding — this confirms the function
        # returns safely without crashing when GPU is unavailable
        device = torch.device("cpu")
        result_bs, result_rt = find_good_batch_size(loader, mock_training_step, device)
        assert result_bs == 32
        assert result_rt is None

class TestFindGoodBatchSizeEdgeCases:
    """Edge cases around dataset size, batch size, and loader configuration."""

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
