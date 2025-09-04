# user_pgdpo_with_projection.py
# 역할: 추정된 Co-state를 PMP 공식에 따라 최적 제어 u로 변환 (Y는 항상 존재한다고 가정)

import torch

# 모델 파라미터는 user_pgdpo_base(정본)에서 로드
from user_pgdpo_base import (
    alpha, sigma, Sigma_inv, rho_Y, sigma_Y, r, gamma, u_cap
)

def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    u*(t, X, Y) = -(1 / (X * J_XX)) · Σ^{-1} · [ J_X · (μ - r) + J_XY · Σ_YX ]
      - J_X   : ∂U/∂X  (shape: B×1)
      - J_XX  : ∂^2U/∂X^2 (shape: B×1)
      - J_XY  : ∂^2U/(∂X∂Y) (shape: B×k)
      - μ - r : (alpha @ Y) * sigma  (shape: B×d)
      - Σ_YX  : (diag(sigma) @ rho_Y @ sigma_Y)^T  (shape: k×d)
      - Σ^{-1}: Sigma_inv (shape: d×d)
    """
    # 필수 항목 확인
    if "Y" not in states:
        raise ValueError("project_pmp: states['Y']가 필요합니다 (Y는 항상 존재해야 함).")
    if "JX" not in costates or "JXX" not in costates or "JXY" not in costates:
        raise ValueError("project_pmp: costates에 'JX', 'JXX', 'JXY'가 모두 필요합니다.")

    # 상태 / 코스테이트
    X   = states["X"]      # (B,1)
    Y   = states["Y"]      # (B,k)
    JX  = costates["JX"]   # (B,1)
    JXX = costates["JXX"]  # (B,1)
    JXY = costates["JXY"]  # (B,k)

    # 스칼라 계수: -(1 / (X * JXX))  (수치 안정화 포함)
    denom = X * JXX
    eps = 1e-8
    denom = torch.where(denom.abs() < eps, denom.sign() * eps, denom)  # (B,1)
    scalar = -1.0 / denom                                             # (B,1)

    # Myopic 항: J_X · (μ - r),  (μ - r) = (alpha @ Y) * sigma
    # (alpha @ Y.unsqueeze(-1)) -> (B,d,1) -> squeeze -> (B,d)
    mu_minus_r = (alpha @ Y.unsqueeze(-1)).squeeze(-1) * sigma        # (B,d)
    myopic = JX * mu_minus_r                                          # (B,1)*(B,d) -> (B,d)

    # Hedging 항: J_XY · Σ_YX,  Σ_XY = diag(sigma) @ rho_Y @ sigma_Y  → Σ_YX = Σ_XY^T
    Sigma_XY = torch.diag(sigma) @ rho_Y @ sigma_Y                    # (d×k)
    Sigma_YX = Sigma_XY.T                                             # (k×d)
    hedging = JXY @ Sigma_YX                                          # (B,k)@(k,d) -> (B,d)

    # 괄호 안: myopic + hedging
    bracket = myopic + hedging                                        # (B,d)

    # 최종 u: scalar*(Σ^{-1}·bracket)
    u = scalar * (Sigma_inv @ bracket.unsqueeze(-1)).squeeze(-1)      # (B,1)*(d×(B,d,1))->(B,d)

    return u