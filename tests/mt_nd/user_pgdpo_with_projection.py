# user_pgdpo_with_projection.py for N-dimensional Merton Problem

import torch

# [수정된 부분] pgdpo_base 대신 user_pgdpo_base에서 임포트합니다.
# 기존 코드: from pgdpo_base import (alpha, Sigma_inv, gamma, u_cap)
from user_pgdpo_base import (
    alpha, Sigma_inv, gamma
)

def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    추정된 Co-state를 PMP에 기반한 최적 제어(u)로 매핑합니다.
    (이하 함수 내용은 동일)
    """
    JX = costates.get('JX')
    JXX = costates.get('JXX')
    X = states.get('X')

    # 분모 계산 (수치 안정성을 위해 epsilon 추가)
    denominator = X * JXX
    eps = 1e-8
    denominator = torch.where(
        torch.abs(denominator) < eps, 
        torch.sign(denominator) * eps, 
        denominator
    )
    # 이론적으로 -JX / (X * JXX)는 1/γ 와 같아야 함
    scalar_coeff = -JX / denominator

    # Myopic 수요 계산 (헤징 수요는 없음)
    myopic_term = alpha 
    
    # 최종 제어 u 계산: u = (1/γ) * Σ⁻¹ * α
    u = scalar_coeff * (Sigma_inv @ myopic_term.unsqueeze(-1)).squeeze(-1)

    return u