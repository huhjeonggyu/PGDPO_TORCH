# user_pgdpo_with_projection.py for 1-dimensional Kim-Omberg Problem

import torch

# user_pgdpo_base에서 고정된 파라미터를 가져옵니다.
from user_pgdpo_base import (
    alpha, sigma, Sigma_inv, rho_Y, sigma_Y
)

def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    추정된 Co-state를 PMP에 기반한 최적 제어(u)로 매핑합니다.
    이 공식은 차원에 관계없이 동일하게 적용됩니다.
    """
    JX, JXX, JXY = costates['JX'], costates['JXX'], costates['JXY']
    X, Y = states['X'], states['Y']

    denominator = X * JXX
    eps = 1e-8
    denominator = torch.where(torch.abs(denominator) < eps, torch.sign(denominator) * eps, denominator)
    scalar_coeff = -1.0 / denominator

    mu_minus_r_vec = (alpha @ Y.unsqueeze(-1)).squeeze(-1) * sigma
    myopic_term = JX * mu_minus_r_vec

    Sigma_XY = torch.diag(sigma) @ rho_Y @ sigma_Y
    Sigma_YX = Sigma_XY.T
    hedging_term = JXY @ Sigma_YX
    
    bracket_term = myopic_term + hedging_term
    u = scalar_coeff * (Sigma_inv @ bracket_term.unsqueeze(-1)).squeeze(-1)

    return u