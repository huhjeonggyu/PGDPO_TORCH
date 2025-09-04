# user_pgdpo_with_projection.py for the Bensoussan-Park (2024) Model
# 역할: 추정된 Co-state를 PMP 공식에 따라 최적 제어 u(투자 비중, 소비율)로 변환합니다.

import torch

# user_pgdpo_base.py에서 정의한 시장 파라미터를 가져옵니다.
from user_pgdpo_base import (
    theta, sigma, Sigma_inv, u_cap
)

def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    가치 함수 V의 미분값에 해당하는 Co-state를 사용하여 최적 제어를 계산합니다.
    논문의 식 (12)를 기반으로 합니다.
    
    - Optimal Consumption:  Ĉ = 1 / (x * ∂V/∂x)
    - Optimal Investment:   ŵ = - (1/x) * (σ*)⁻¹ * θ * (∂V/∂x) / (∂²V/∂x²)

    Co-state와 가치 함수의 관계:
    - JX  ≈ ∂V/∂x
    - JXX ≈ ∂²V/∂x²
    """
    # 상태 및 co-state 추출
    X = states.get('X')
    JX = costates.get('JX')
    JXX = costates.get('JXX')

    # 수치 안정성을 위한 작은 값(epsilon)
    eps = 1e-8

    # 1. 최적 소비율 (Optimal Consumption Rate, Ĉ) 계산
    # Ĉ = 1 / (X * JX)
    denominator_C = X * JX
    # 분모가 0에 가까워지는 것을 방지
    denominator_C = torch.where(
        torch.abs(denominator_C) < eps,
        torch.sign(denominator_C) * eps,
        denominator_C
    )
    optimal_C = 1.0 / denominator_C

    # 2. 최적 투자 비중 (Optimal Investment Allocation, ŵ) 계산
    # ŵ = - (σ*)⁻¹ * θ * (1/X) * (JX / JXX)
    # (d,d) @ (d,) -> (d,)
    sigma_inv_theta = Sigma_inv @ theta
    
    denominator_w = X * JXX
    denominator_w = torch.where(
        torch.abs(denominator_w) < eps,
        torch.sign(denominator_w) * eps,
        denominator_w
    )
    # (B,1) / (B,1) -> (B,1)
    ratio_J = JX / denominator_w
    
    # (d,) * (B,1) -> (B,d) (Broadcasting)
    optimal_w = -sigma_inv_theta * ratio_J

    # 3. 제약 조건 적용 및 결과 결합
    optimal_w = torch.clamp(optimal_w, -u_cap, u_cap)
    optimal_C = torch.clamp(optimal_C, 1e-4, u_cap)

    # DirectPolicy의 출력 형식과 동일하게 (투자 비중, 소비율) 순서로 결합하여 반환
    return torch.cat([optimal_w, optimal_C], dim=1)