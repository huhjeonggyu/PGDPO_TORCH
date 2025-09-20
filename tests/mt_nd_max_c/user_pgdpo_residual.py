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
    u_myopic = (1/gamma) * Sigma_inv @ alpha  (머튼 공식)
      - u>=0, sum(u)<=L_cap 로 투영
    consumption(state):
      - 제약 비활성 가정하의 비례소비 C = min{ C_FRAC * X, alpha_rel * X }
    """
    def __init__(self):
        super().__init__()
        # 미리 상수 u_star 계산
        u_unc = (1.0 / float(GAMMA_CONST)) * (SIGMAi_CONST @ ALPHA_CONST)  # (d,)
        u0 = u_unc.clamp_min(0.0)
        ssum = float(u0.sum().item())
        L_cap = float(L_CAP_CONST)
        d = u0.numel()

        if ssum > L_cap:
            # equality simplex로 투영
            u_sorted = torch.sort(u0, descending=True).values
            cssv = torch.cumsum(u_sorted, dim=0) - L_cap
            j = torch.arange(1, d+1, device=u0.device, dtype=u0.dtype)
            cond = u_sorted > (cssv / j)
            if cond.any():
                rho_idx = int(torch.nonzero(cond, as_tuple=False)[-1].item())
                theta = cssv[rho_idx] / float(rho_idx + 1)
            else:
                theta = cssv[-1] / float(d)
            u0 = (u0 - theta).clamp_min(0.0)

        self.register_buffer("u_star", u0.view(-1))  # (d,)

    @torch.no_grad()
    def forward(self, **states_dict):
        # 배치 크기에 맞춰 복제
        X = states_dict.get("X", None)
        if X is None:
            return self.u_star.unsqueeze(0)  # (1,d)
        B = X.shape[0]
        return self.u_star.unsqueeze(0).expand(B, -1)

    @torch.no_grad()
    def consumption(self, **states_dict):
        # 상대상한 미활성 가정의 비례소비, 단 C <= alpha_rel * X 보장
        X = states_dict["X"]
        alpha_rel = float(ALPHA_REL_CONST)
        c_frac = float(C_FRAC)
        C_prop = c_frac * X
        C_cap  = alpha_rel * X
        return torch.minimum(C_prop, C_cap)  # (B,1)
