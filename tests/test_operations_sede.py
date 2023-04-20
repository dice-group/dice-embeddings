from dicee.models.sedenion import conjugate, hermitian, o_mul, s_mul
import pytest
import torch


s = [torch.tensor([1 if j == i else 0 for j in range(16)]) for i in range(16)]


def to_tensor(*args):
    return torch.stack(list(torch.tensor(list(zip(*item))).flatten() for item in args))


class TestOperationsSedE:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_conjugate(self):
        a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        b = torch.tensor([1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16])
        input = to_tensor([s[0], a], [a.mul(2), a.mul(3)])
        expected = to_tensor([s[0], b], [b.mul(2), b.mul(3)])
        got = conjugate(C=input, dim=16)
        assert torch.equal(got, expected)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_hermitian(self):
        a = to_tensor([s[1], s[2]], [s[3], s[4]])
        b = to_tensor([s[1], s[3]], [s[4], s[4].mul(2)])
        expected = torch.tensor([[1, 0], [0, 2]])
        got = hermitian(C_1=a, C_2=b, dim=16)
        assert torch.equal(got, expected)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_o_mul(self):
        a = to_tensor([(0, 1, 0, 0, 0, 0, 0, 0),
                       (0, 0, 0, 1, 0, 0, 0, 0)])
        print(a)
        b = to_tensor([(0, 0, 1, 0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 1, 0, 0, 0)])
        print(b)
        expected = to_tensor([(0, 0, 0, 1, 0, 0, 0, 0),
                              (0, 0, 0, 0, 0, 0, 0, 1)])
        got = o_mul(O_1=a, O_2=b)
        assert torch.equal(got, expected)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_s_mul(self):
        S_1 = to_tensor([s[0], s[1]], [s[14], s[15]])
        S_2 = to_tensor([s[1], s[2]], [s[15], s[15]])
        expected = to_tensor([s[1], s[3]], [s[1], s[0].neg()])
        got = s_mul(S_1=S_1, S_2=S_2)
        print(got)
        print(expected)
        assert torch.equal(got, expected)
