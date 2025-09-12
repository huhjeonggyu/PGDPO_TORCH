# tests/vpp/user_pgdpo_with_projection.py
import torch
from user_pgdpo_base import R_diag, price_fn, T

# ✨ Q=0이므로 최적 정책은 상태(X)와 무관해져 JX가 필요 없어집니다.
PP_NEEDS = ()

@torch.no_grad()
def project_pmp(costates, states):
    # 순수 차익거래(Q=0)에서 최적 정책은 근시안적(myopic)이며 상태와 무관합니다:
    #   u*(t) = P(t) / R
    # 따라서 가치함수의 미분인 co-state(JX)는 0이 되어야 하므로 무시합니다.
    
    device, dtype = R_diag.device, R_diag.dtype

    # 현재 시간 't'를 계산합니다.
    if isinstance(states, dict) and "t" in states:
        t = states["t"].to(device=device, dtype=dtype)
    else:
        TmT = states["TmT"].to(device=device, dtype=dtype)
        t   = (T - TmT).clamp_min(0.0)

    # 가격(P)과 제어 비용의 역수(R_inv)를 가져옵니다.
    P  = price_fn(t).to(device=device, dtype=dtype)      # (B,1)
    Rinv = (1.0 / R_diag).to(device=device, dtype=dtype).view(1, -1)      # (1,d)
    
    # u = P(t) / R 을 브로드캐스팅을 이용해 계산합니다.
    return P * Rinv