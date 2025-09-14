# user_pgdpo_with_projection.py for SIR
import torch
from user_pgdpo_base import u_cap, B_cost

# PMP 프로젝션은 JX (1차 미분값)만 필요합니다.
PP_NEEDS = ("JX",)

def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    추정된 Co-state를 PMP 공식에 따라 최적 제어 u(백신 접종률)로 변환합니다.
    """
    JX = costates["JX"]  # 전체 상태 X에 대한 Co-state 벡터
    X  = states["X"]    # 전체 상태 벡터 [S1,I1,R1, S2,I2,R2, ...]

    # ✨ FIX: 올바른 스텝 슬라이싱(step slicing)을 사용하여 각 구획의 데이터를 정확히 추출합니다.
    # X[:, 0::3] -> 0번 인덱스부터 3칸씩 건너뛰며 모든 S를 추출
    S = X[:, 0::3]  # (B, N_regions)
    # X[:, 2::3] -> 2번 인덱스부터 3칸씩 건너뛰며 모든 R을 추출
    R = X[:, 2::3]  # (B, N_regions)

    # Co-state JX에 대해서도 동일한 방식으로 슬라이싱합니다.
    pS = JX[:, 0::3] # S에 대한 Co-state
    pR = JX[:, 2::3] # R에 대한 Co-state
    
    # 논문의 PMP 공식: u* = (S/B) * (pS - pR)
    #u = (S * (pS - pR)) / B_cost.view(1, -1)
    u = (S * (pR - pS)) / B_cost.view(1, -1)
    
    return torch.clamp(u, 0.0, u_cap)