# tests/vpp/user_pgdpo_residual.py
import torch
import torch.nn as nn
# ✨ user_pgdpo_base에서 필요한 변수들을 가져옴
from user_pgdpo_base import d, price_fn, T, device, R_diag, u_min, u_max

class MyopicPolicy(nn.Module):
    """
    ✨ 수정된 Myopic 정책: u_i(t) = P(t) / R_i
    """
    def __init__(self):
        super().__init__()
        # R_diag를 (1,d) 형태로 만들어 브로드캐스팅 준비
        self.register_buffer("inv_R_diag", 1.0 / R_diag.view(1, -1))

    def forward(self, **states_dict) -> torch.Tensor:
        t = T - states_dict['TmT']  # (B,1)
        price = price_fn(t)         # (B,1)
        # (B,1) * (1,d) -> (B,d) (브로드캐스팅)
        u = price * self.inv_R_diag
        return torch.clamp(u, u_min, u_max)