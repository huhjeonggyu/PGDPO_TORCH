# tests/vpp/user_pgdpo_with_projection.py
import torch
from user_pgdpo_base import R_diag, price_fn, T

PP_NEEDS = ("JX",)

@torch.no_grad()
def project_pmp(costates, states):
    # p_hat ≡ JX = ∂(-profit)/∂x = -p  →  u = R^{-1}(P + JX)
    p_hat  = costates if torch.is_tensor(costates) else costates["JX"]   # (B,d)
    device, dtype = p_hat.device, p_hat.dtype

    # t 처리: states에 't'가 있으면 그대로, 없으면 T - TmT
    if isinstance(states, dict) and "t" in states:
        t = states["t"].to(device=device, dtype=dtype)
    else:
        TmT = states["TmT"].to(device=device, dtype=dtype)               # (B,1) = T - t
        t   = (T - TmT).clamp_min(0.0)                                    # (B,1)

    P  = price_fn(t).to(device=device, dtype=dtype).expand_as(p_hat)      # (B,d)
    Rinv = (1.0 / R_diag).to(device=device, dtype=dtype).view(1, -1)      # (1,d)
    return (P + p_hat) * Rinv                                             # (B,d)
