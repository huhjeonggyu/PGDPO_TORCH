# tests/<model>/user_pgdpo_residual.py
# ------------------------------------------------------------
# Residual 모드 베이스 정책(Myopic) + 스케일 설정
# - 코어는 여기서 MyopicPolicy, ResCap 을 import 합니다.
# - MyopicPolicy.forward: u_myopic (고정), 잔차는 코어 ResidualPolicy가 학습
# - MyopicPolicy.consumption: 상대상한이 없다고 가정한 평시(consumption-unconstrained) 비례소비를 사용,
#   단, 항상 C <= alpha_rel * X 를 보장 (min).
# ------------------------------------------------------------

from __future__ import annotations
import os
import torch
import torch.nn as nn

from user_pgdpo_base import (
    alpha as ALPHA_CONST,     # (d,)
    Sigma_inv as SIGMAi_CONST,  # (d,d)
    gamma as GAMMA_CONST,     # float
)

# 옵션 심볼 (없으면 기본값 사용)
try:
    from user_pgdpo_base import L_cap as L_CAP_CONST
except Exception:
    L_CAP_CONST = 1.0

try:
    from user_pgdpo_base import alpha_rel as ALPHA_REL_CONST
except Exception:
    ALPHA_REL_CONST = 0.30

# 잔차 크기 스케일(코어가 import)
ResCap = float(os.getenv("PGDPO_RES_CAP", 0.10))

# 소비 기본 비율(제약 안 걸렸을 때의 소비 성향), 0~1 사이 권장
C_FRAC = float(os.getenv("PGDPO_C_FRAC", 0.50))


class MyopicPolicy(nn.Module):
    """
    MyopicPolicy는 이제 투자(u)와 소비(C)를 결합하여 반환합니다.
    """
    def __init__(self):
        super().__init__()
        # u_star 계산 로직은 기존과 동일
        u_unc = (1.0 / float(GAMMA_CONST)) * (SIGMAi_CONST @ ALPHA_CONST)
        u0 = u_unc.clamp_min(0.0)
        ssum = float(u0.sum().item())
        L_cap = float(L_CAP_CONST)
        d = u0.numel()

        if ssum > L_cap:
            u_sorted = torch.sort(u0, descending=True).values
            cssv = torch.cumsum(u_sorted, dim=0) - L_cap
            j = torch.arange(1, d + 1, device=u0.device, dtype=u0.dtype)
            cond = u_sorted > (cssv / j)
            if cond.any():
                rho_idx = int(torch.nonzero(cond, as_tuple=False)[-1].item())
                theta = cssv[rho_idx] / float(rho_idx + 1)
            else:
                theta = cssv[-1] / float(d)
            u0 = (u0 - theta).clamp_min(0.0)

        self.register_buffer("u_star", u0.view(-1))

    @torch.no_grad()
    def forward(self, **states_dict):
        X = states_dict.get("X")
        B = X.shape[0] if X is not None else 1
        
        # 1. 기준 투자 u_star 생성
        u = self.u_star.unsqueeze(0).expand(B, -1)

        # 2. 기준 소비 C 계산
        alpha_rel = float(ALPHA_REL_CONST)
        c_frac = float(C_FRAC)
        C_prop = c_frac * X
        C_cap  = alpha_rel * X
        C = torch.minimum(C_prop, C_cap)

        # 3. u와 C를 결합하여 (B, d+1) 형태로 반환
        return torch.cat([u, C], dim=1)