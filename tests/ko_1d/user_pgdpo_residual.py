# user_pgdpo_residual.py for 1-dimensional Kim-Omberg Problem
# 역할: 잔차 학습의 기반이 될 베이스라인 정책(Myopic)을 정의합니다.

import torch
import torch.nn as nn

# user_pgdpo_base에서 고정된 파라미터를 가져옵니다.
from user_pgdpo_base import (
    gamma, alpha, sigma, Sigma_inv
)

class MyopicPolicy(nn.Module):
    """
    상태 Y에 따라 동적으로 변하는 분석적 마이오픽(Myopic) 정책을 계산합니다.
    u_myopic(Y) = (1/γ) * Σ⁻¹ * (μ(Y) - r)
    """
    def __init__(self):
        super().__init__()
        # 학습 가능한 파라미터가 없습니다.

    def forward(self, **states_dict):
        """
        주어진 상태 Y에 대해 최적의 근시안적 제어(u)를 계산합니다.
        """
        Y = states_dict['Y']
        
        mu_minus_r_vec = (alpha @ Y.unsqueeze(-1)).squeeze(-1) * sigma
        u = (1.0 / gamma) * (Sigma_inv @ mu_minus_r_vec.unsqueeze(-1)).squeeze(-1)
        
        return u