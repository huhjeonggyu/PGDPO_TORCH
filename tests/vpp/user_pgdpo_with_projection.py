# tests/vpp/user_pgdpo_with_projection.py
import torch
# ✨ user_pgdpo_base에서 R_diag와 가격 함수를 가져옴
from user_pgdpo_base import R_diag, price_fn, T

PP_NEEDS = ("JX",)

def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    ✨ 수정된 PMP 프로젝션 공식 (논문 식 15)
    u*_i = (P(t) + p_i) / R_i
    """
    JX = costates['JX']  # Co-state p와 같은 의미로 사용, shape: (B, d)
    TmT = states['TmT']  # shape: (B, 1)
    t = T - TmT          # 현재 시간

    price = price_fn(t)  # 현재 가격 P(t), shape: (B, 1)

    # 각 차원별로 연산
    # (B,1) + (B,d) -> (B,d) (브로드캐스팅)
    # (B,d) / (d,)    -> (B,d) (브로드캐스팅)
    u = (price - JX) / R_diag.view(1, -1)
    
    return u