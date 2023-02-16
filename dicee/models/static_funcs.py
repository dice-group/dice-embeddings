from typing import Tuple
import torch

def quaternion_mul(*, Q_1, Q_2) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """
    Perform quaternion multiplication
    :param Q_1:
    :param Q_2:
    :return:
    """

    a_h, b_h, c_h, d_h = Q_1
    a_r, b_r, c_r, d_r = Q_2
    r_val = a_h * a_r - b_h * b_r - c_h * c_r - d_h * d_r
    i_val = a_h * b_r + b_h * a_r + c_h * d_r - d_h * c_r
    j_val = a_h * c_r - b_h * d_r + c_h * a_r + d_h * b_r
    k_val = a_h * d_r + b_h * c_r - c_h * b_r + d_h * a_r
    return r_val, i_val, j_val, k_val
