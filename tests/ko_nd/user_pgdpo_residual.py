# user_pgdpo_residual.py
# 역할: 잔차 학습(Residual Learning)의 베이스라인 Myopic 정책 (클램프 없음)
# - Y는 항상 존재한다고 가정
# - u(Y) = (1/γ) · Σ^{-1} · (μ(Y) - r),  (μ - r) = (α @ Y) ⊙ σ

import torch
import torch.nn as nn

# 모델 파라미터는 pgdpo_base(재노출)에서 로드
from user_pgdpo_base import (
    alpha, sigma, Sigma_inv, gamma
)

class MyopicPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_gamma = 1.0 / gamma

    def forward(self, **states_dict) -> torch.Tensor:
        Y = states_dict.get("Y")
        if Y is None:
            raise ValueError("MyopicPolicy requires states['Y'] (Y must exist).")

        # (α @ Y.unsqueeze(-1)) -> (B,d,1) -> squeeze -> (B,d), 이후 σ와 원소곱
        mu_minus_r = (alpha @ Y.unsqueeze(-1)).squeeze(-1) * sigma   # (B,d)
        # u = (1/γ) · Σ^{-1} · (μ - r)
        u = self.inv_gamma * (Sigma_inv @ mu_minus_r.unsqueeze(-1)).squeeze(-1)  # (B,d)
        return u