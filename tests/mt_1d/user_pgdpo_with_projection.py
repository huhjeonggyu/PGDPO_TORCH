# user_pgdpo_with_projection.py for 1-dimensional Merton Problem

import torch

# user_pgdpo_base에서 고정된 파라미터를 가져옵니다.
from user_pgdpo_base import (
    alpha, Sigma_inv, gamma
)

def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    추정된 Co-state를 PMP에 기반한 최적 제어(u)로 매핑합니다.
    이 공식은 차원에 관계없이 동일하게 적용됩니다.
    """
    JX = costates.get('JX')
    JXX = costates.get('JXX')
    X = states.get('X')

    denominator = X * JXX
    eps = 1e-8
    denominator = torch.where(
        torch.abs(denominator) < eps, 
        torch.sign(denominator) * eps, 
        denominator
    )
    scalar_coeff = -JX / denominator
    myopic_term = alpha 
    u = scalar_coeff * (Sigma_inv @ myopic_term.unsqueeze(-1)).squeeze(-1)

    return u