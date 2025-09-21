# 파일: tests/mt_nd_max_c/user_pgdpo_residual.py
# 모델: 소비 상한 모델 (고정된 절대값 상한) - 수정된 버전

from __future__ import annotations
import os
import torch
import torch.nn as nn

# user_pgdpo_base.py에서 필요한 파라미터 가져오기
from user_pgdpo_base import (
    alpha as ALPHA_CONST,
    Sigma_inv as SIGMAi_CONST,
    gamma as GAMMA_CONST,
    rho as RHO_CONST,
    r as R_CONST,
    # ✨ 수정: alpha_rel 대신 C_abs_cap를 import
    C_abs_cap as C_ABS_CAP_CONST 
)

# 잔차 크기 스케일(코어가 import)
ResCap = float(os.getenv("PGDPO_RES_CAP", 0.10))

class MyopicPolicy(nn.Module):
    """
    MyopicPolicy는 제약 없는 표준 머튼 해를 기준 투자(u)로 사용하고,
    이론적 최적 소비율(m*)에 고정 상한을 적용하여 기준 소비(C)를 계산합니다.
    """
    def __init__(self):
        super().__init__()
        
        # 1. 기준 투자 u_star 계산 (제약 없는 표준 머튼 해)
        u_star = (1.0 / float(GAMMA_CONST)) * (SIGMAi_CONST @ ALPHA_CONST)
        self.register_buffer("u_star", u_star.view(-1))

        # 2. 기준 소비율 m* 계산
        theta_sq = (ALPHA_CONST.T @ SIGMAi_CONST @ ALPHA_CONST).item()
        gamma_val = float(GAMMA_CONST)
        rho_val = float(RHO_CONST)
        r_val = float(R_CONST)
        
        m_star = (rho_val - (1 - gamma_val) * (r_val + theta_sq / (2 * gamma_val))) / gamma_val
        self.m_star = m_star

    @torch.no_grad()
    def forward(self, **states_dict):
        X = states_dict.get("X")
        if X is None:
            raise ValueError("MyopicPolicy requires state 'X'.")
        B = X.shape[0]
        
        # 1. 기준 투자 u_star 생성
        u = self.u_star.unsqueeze(0).expand(B, -1)

        # 2. ✨ 수정: 기준 소비 C를 고정 상한(C_ABS_CAP_CONST)을 사용하여 계산
        C_prop = self.m_star * X
        C_cap  = torch.full_like(X, float(C_ABS_CAP_CONST))
        C = torch.minimum(C_prop, C_cap)

        # 3. u와 C를 결합하여 (B, d+1) 형태로 반환
        return torch.cat([u, C], dim=1)