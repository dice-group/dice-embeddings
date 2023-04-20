from dicee.models.octonion import octonion_mul
import pytest
import torch


class TestOperationsOctonion:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_octonion_mul(self):
        O_1 = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0]]).hsplit(8)
        O_2 = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0]]).hsplit(8)
        expected = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]])
        got = torch.hstack(octonion_mul(O_1=O_1, O_2=O_2))
        assert torch.equal(got, expected)
