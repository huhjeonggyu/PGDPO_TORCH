# user_pgdpo_with_projection.py — P-PGDPO projector (pre-commitment MV + λ)
# π = ρ G,   ρ = Jx / (λ Φ(τ)^2),   Φ(τ)=exp(r τ),  τ = T - t.
from __future__ import annotations
from typing import Dict, Optional
import os, torch
from torch import Tensor
from user_pgdpo_base import T, r, lambda_curv, Gd, Phi_of_tau, pi_cap, pi_l2_cap

PP_NEEDS = ("JX",)  # J_x만 필요

@torch.no_grad()
def project_pmp(costates: Dict[str, Optional[Tensor]], states: Dict[str, Tensor]) -> Tensor:
    JX = costates.get("JX", None)
    if JX is None:
        raise RuntimeError("[PMP projector] need costate 'JX'.")
    if JX.dim() == 1:
        JX = JX.view(-1, 1)

    X = states["X"]
    if "TmT" in states:
        tau = states["TmT"]
    else:
        t = states.get("t", torch.zeros_like(X))
        tau = (T - t * T)

    Phi   = Phi_of_tau(tau.to(JX.device, JX.dtype))          # (B,1)
    denom = (lambda_curv * (Phi * Phi)).clamp_min(1e-12)     # (B,1)

    # 배치-스칼라 ρ (노이즈 억제)
    rho_scalar = (JX / denom).mean(dim=0, keepdim=True)      # (1,1)
    # 선택적 clip
    rho_cap = float(os.getenv("PGDPO_RHO_CAP", "0"))         # 0이면 off
    if rho_cap > 0:
        rho_scalar = torch.clamp(rho_scalar, -rho_cap, rho_cap)

    # π = ρ G (배치로 확장)
    pi = rho_scalar @ Gd.to(JX.device, JX.dtype)             # (1,d)
    pi = pi.expand(JX.shape[0], -1)                          # (B,d)

    # L2-cap → elementwise-cap 순서 권장
    if pi_l2_cap and pi_l2_cap > 0.0:
        norm  = pi.norm(dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(pi_l2_cap / norm, max=1.0)
        pi    = pi * scale
    if pi_cap and pi_cap > 0.0:
        pi = torch.clamp(pi, -pi_cap, pi_cap)
    return pi