# user_pgdpo_residual.py for the Bensoussan-Park (2024) Model
# 역할: 잔차 학습(Residual Learning)의 베이스라인이 될 Myopic 정책을 정의합니다.
#       이 정책은 노동 소득의 불확실성이 없는 경우(ρ=0)의 최적해를 사용합니다.

import torch
import torch.nn as nn

# user_pgdpo_base.py에서 정의한 시장 파라미터를 가져옵니다.
from user_pgdpo_base import (
    beta, r, mu_income, theta, sigma, Sigma_inv, u_cap
)

class MyopicPolicy(nn.Module):
    """
    노동 소득 불확실성이 없는 경우(ρ=0)의 분석적 최적 정책을 계산합니다.
    이 정책은 잔차 학습의 베이스라인(기준점)으로 사용됩니다.
    
    논문의 Corollary 1 (page 9)을 기반으로 합니다.
    - Optimal Consumption:  Ĉ(z) = β * (1 + z / (r - μ))
    - Optimal Investment:   ŵ(z) = (σ*)⁻¹ * θ * (1 + z / (r - μ))
      where z = y/x
    """
    def __init__(self):
        super().__init__()
        # 이 정책은 학습 가능한 파라미터가 없습니다.
        self.sigma_inv_theta = Sigma_inv @ theta

    def forward(self, **states_dict) -> torch.Tensor:
        """
        주어진 상태 (X, Y)에 대해 베이스라인 제어(투자 비중, 소비율)를 계산합니다.
        """
        X = states_dict.get('X')
        Y = states_dict.get('Y')

        # 소득-자산 비율 z = Y / X 계산
        z = Y / X.clamp_min(1e-8)

        # 공통 항 (1 + z / (r - μ)) 계산
        # 논문에서 r > mu 가정이 있으므로 분모는 양수입니다.
        common_term = 1 + z / (r - mu_income)

        # 1. 베이스라인 소비율 (Baseline Consumption Rate) 계산
        baseline_C = beta * common_term

        # 2. 베이스라인 투자 비중 (Baseline Investment Allocation) 계산
        baseline_w = self.sigma_inv_theta.unsqueeze(0) * common_term

        # 3. 제약 조건 적용 및 결과 결합
        baseline_w = torch.clamp(baseline_w, -u_cap, u_cap)
        baseline_C = torch.clamp(baseline_C, 1e-4, u_cap)

        # DirectPolicy의 출력 형식과 동일하게 (투자 비중, 소비율) 순서로 결합하여 반환
        return torch.cat([baseline_w, baseline_C], dim=1)