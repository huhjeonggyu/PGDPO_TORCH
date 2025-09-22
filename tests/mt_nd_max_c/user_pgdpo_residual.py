# 파일: tests/mt_nd_max_c/user_pgdpo_residual.py
# 모델: 소비 상한 모델 (상대 소비 상한; 유한지평 m(t) 기반 마이오픽 베이스라인) - 수정된 버전

from __future__ import annotations
import os
import torch
import torch.nn as nn

# user_pgdpo_base.py에서 필요한 파라미터 가져오기
from user_pgdpo_base import (
    alpha as ALPHA_CONST,          # (d,)
    Sigma_inv as SIGMAi_CONST,     # (d,d)
    gamma as GAMMA_CONST,          # scalar
    rho as RHO_CONST,              # scalar
    r as R_CONST,                  # scalar
    c_frac as C_FRAC_CONST,        # 상대 상한 비율: C_max(X)=c_frac*X
    T as T_CONST,                  # 만기
    kappa as KAPPA_CONST           # 말기효용 가중 (boundary에서 m(T)=kappa^{-1/gamma})
)

# 잔차 크기 스케일(코어가 import)
ResCap = float(os.getenv("PGDPO_RES_CAP", 0.50))

class MyopicPolicy(nn.Module):
    """
    MyopicPolicy는 무제약 머튼 해를 기준 투자(u)로 사용하고,
    유한지평 m(t)와 상대 소비 상한 C_max(X)=c_frac*X를 적용해 기준 소비(C)를 계산한다.
    """
    def __init__(self):
        super().__init__()
        # 1) 기준 투자 u_star = (1/gamma) Σ^{-1} α
        u_star = (1.0 / float(GAMMA_CONST)) * (SIGMAi_CONST @ ALPHA_CONST)
        self.register_buffer("u_star", u_star.view(-1))  # (d,)

        # 2) 유한지평 소비율 m(t) 파라미터 (m*, m_T)
        #    m* = [rho - (1-gamma)(r + theta^2/(2gamma))]/gamma
        theta_sq = (ALPHA_CONST.view(1, -1) @ SIGMAi_CONST @ ALPHA_CONST.view(-1, 1)).item()
        gamma_val = float(GAMMA_CONST)
        rho_val   = float(RHO_CONST)
        r_val     = float(R_CONST)
        self.m_star = (rho_val - (1.0 - gamma_val) * (r_val + theta_sq / (2.0 * gamma_val))) / gamma_val
        # 종결 경계 m(T) = kappa^{-1/gamma}
        kap = float(KAPPA_CONST)
        self.m_T = (kap if kap > 0.0 else 1e-12) ** (-1.0 / gamma_val)

        # 3) 상대 소비 상한 비율
        self.c_frac = float(C_FRAC_CONST)
        self.T = float(T_CONST)

    @torch.no_grad()
    def forward(self, **states_dict):
        X   = states_dict.get("X")
        TmT = states_dict.get("TmT", None)  # 남은 시간 τ = T - t
        if X is None:
            raise ValueError("MyopicPolicy requires state 'X' (shape (B,1)).")
        if TmT is None:
            # τ 정보가 없으면 보수적으로 τ=0으로 두면 m(t)=m(T)에 가까워짐
            TmT = torch.zeros_like(X)

        B = X.shape[0]
        dev, dt = X.device, X.dtype

        # 1) 기준 투자 u_star (B,d)
        u = self.u_star.to(device=dev, dtype=dt).unsqueeze(0).expand(B, -1)

        # 2) 유한지평 소비율 m(t) 계산
        #    m(t) = m* / [1 + (m*/m_T - 1) * exp(-m* * τ)],  τ = T - t
        m_star = torch.as_tensor(self.m_star, device=dev, dtype=dt)
        m_T    = torch.as_tensor(self.m_T,    device=dev, dtype=dt)
        denom  = 1.0 + (m_star / m_T - 1.0) * torch.exp(-m_star * TmT)
        m_t    = (m_star / denom).clamp_min(torch.finfo(dt).eps)  # (B,1)

        # 3) 상대 상한으로 소비 결정: C = min{ m(t)*X, c_frac*X }
        C_cap = self.c_frac * X
        C_un  = m_t * X
        C     = torch.minimum(C_un, C_cap)

        # 4) (B, d+1) 형태로 반환
        return torch.cat([u, C], dim=1)
