# user_pgdpo_with_projection.py — Theory-free PG-DPO projector (CIS, k=0)
# - No closed-form anchors, no JX_cf / rho_cf.
# - Uses BPTT costates only: JX (=Y) and ZW (=Z^{(W)} scalar).
# - Groupwise ridge regression to estimate ∂_x y from ZW = (∂_x y) * σ_X.
# - Then ρ_g = - median(Y_g) / ∂_x y_g, winsorize per-batch, ρ>=0, π caps.

import os
from typing import Dict, Optional

import torch
from torch import Tensor

from user_pgdpo_base import (
    Sigma, alpha, device, DTYPE,
    pi_cap, pi_l2_cap,
    CURRENT_POLICY,  # baseline policy used when costates were generated
    T,
)

# Fixed market direction G = Σ^{-1} α
G_vec: Tensor = torch.linalg.solve(Sigma, alpha).to(device=device, dtype=DTYPE)

# We require both Y (JX) and Z^{(W)} (ZW)
PP_NEEDS = ("JX", "ZW")

# ---------- helpers ----------

def _apply_caps(pi: Tensor, l2_cap: float, lin_cap: float) -> Tensor:
    if l2_cap > 0.0:
        norm = pi.norm(dim=1, keepdim=True) + 1e-12
        scale = torch.clamp(l2_cap / norm, max=1.0)
        pi = pi * scale
    if lin_cap > 0.0:
        pi = torch.clamp(pi, -lin_cap, lin_cap)
    return pi

def _ensure_col(x: Tensor) -> Tensor:
    return x.view(-1, 1) if (x.dim() == 1) else x

@torch.no_grad()
def project_pmp(costates: Dict[str, Optional[Tensor]], states: Dict[str, Tensor]) -> Tensor:
    """
    Inputs:
      costates: dict with 'JX' (Y) [B,1] or [B,], 'ZW' (scalar Z^{(W)}) [B,1] or [B,]
      states:   dict with 'X' [B,1], 'TmT' [B,1], 'gid' [B] (optional), and possibly 't'/'t_abs'
    Output:
      pi [B, d]
    """
    # ---- load state batch ----
    X   = _ensure_col(states.get("X"))
    TmT = _ensure_col(states.get("TmT", torch.zeros_like(X)))
    gid = states.get("gid", None)  # if None, treat as single group
    B   = X.size(0)

    # ---- costates ----
    JX = costates.get("JX", None)   # this is Y(t,t)
    ZW = costates.get("ZW", None)   # scalar Z^{(W)} for dW̃ used in X update

    if JX is None or ZW is None:
        # failing safe: theory-free requires both; if missing, output zeros (or baseline)
        # (Better to let the core know PP_NEEDS must include both)
        pi_zero = torch.zeros(B, G_vec.numel(), device=device, dtype=DTYPE)
        return _apply_caps(pi_zero, float(pi_l2_cap), float(pi_cap))

    JX = _ensure_col(JX).to(device=device, dtype=DTYPE)
    ZW = _ensure_col(ZW).to(device=device, dtype=DTYPE)

    # ---- baseline policy used when costates were generated ----
    # We need σ_X = sqrt(pi_base^T Σ pi_base).
    if CURRENT_POLICY is None:
        # Should not happen in normal pipeline; fall back to zero policy
        pi_base = torch.zeros(B, G_vec.numel(), device=device, dtype=DTYPE)
    else:
        # Accept various key names for time:
        t_abs = states.get("t_abs", states.get("t", None))
        if t_abs is None:
            # if only TmT is given:
            t_abs = T - TmT
        t_abs = _ensure_col(t_abs)
        # forward baseline:
        pi_base = CURRENT_POLICY(t_abs, X).to(device=device, dtype=DTYPE)

    # σ_X (B,1)
    var = (pi_base @ Sigma @ pi_base.T).diagonal().clamp_min(1e-12).view(-1, 1)
    sigX = torch.sqrt(var)  # (B,1)

    # ---- estimate ∂_x y by groupwise ridge:  ZW_i ≈ (∂_x y)_g * sigX_i ----
    # a_g = argmin_a sum_i (ZW_i - a * sigX_i)^2  ⇒  a_g = (sum ZW_i*sigX_i)/(sum sigX_i^2 + λ)
    ridge = float(os.getenv("PGDPO_RIDGE_LAMBDA", "1e-6"))

    if gid is None:
        # single group
        num = (ZW * sigX).sum(dim=0)                       # [1,1]
        den = (sigX.pow(2)).sum(dim=0) + ridge            # [1,1]
        dyy_g = (num / den).view(1, 1)                    # ∂_x y (group)
        # robust Y_g with median
        Y_g = JX.median(dim=0).values.view(1, 1)
        rho_g = - Y_g / dyy_g                              # [1,1]
        # per-sample ρ is constant in group
        rho = rho_g.expand(B, 1)
    else:
        gid = gid.to(device)
        Gmax = int(gid.max().item()) + 1

        # group sums for ridge
        # sum_i ZW_i*sigX_i
        num_by_g = torch.zeros(Gmax, 1, device=device, dtype=DTYPE).index_add_(0, gid, (ZW * sigX))
        # sum_i sigX_i^2
        den_by_g = torch.zeros(Gmax, 1, device=device, dtype=DTYPE).index_add_(0, gid, (sigX.pow(2)))
        dyy_by_g = num_by_g / (den_by_g + ridge)           # [G,1]

        # robust Y per group (median)
        # (median은 효율 위해 루프 사용; G는 보통 작음)
        Y_by_g = torch.empty(Gmax, 1, device=device, dtype=DTYPE)
        for g in range(Gmax):
            mask = (gid == g)
            if mask.any():
                Y_by_g[g, 0] = JX[mask].median()
            else:
                Y_by_g[g, 0] = 0.0

        rho_by_g = torch.abs(Y_by_g) / torch.clamp(torch.abs(dyy_by_g), min=1e-9)  # [G,1]
        rho = rho_by_g[gid]                                    # [B,1]

        # winsorize ρ across groups (theory-free)
        q = float(os.getenv("PGDPO_RHO_WINSOR_Q", "0.10"))     # 10% by default
        if 0.0 < q < 0.5 and Gmax > 2:
            rhos_g = rho_by_g.flatten()
            lo = torch.quantile(rhos_g, q)
            hi = torch.quantile(rhos_g, 1.0 - q)
            rho_by_g = torch.clamp(rho_by_g, lo, hi)
            rho = rho_by_g[gid]

    # optional positivity (MV equilibrium ⇒ ρ ≥ 0)
    if os.getenv("PGDPO_RHO_POS", "1") == "1":
        rho = torch.clamp_min(rho, 0.0)

    # final π
    pi = rho * G_vec.view(1, -1)
    pi = _apply_caps(pi, float(pi_l2_cap), float(pi_cap))
    pi = torch.nan_to_num(pi, nan=0.0, posinf=1e6, neginf=-1e6)

    if os.getenv("PGDPO_PP_DEBUG", "0") == "1":
        # quick stats (theory-free)
        with torch.no_grad():
            # print group stats if gid exists
            if gid is None:
                print(f"[pp TF] dyy={dyy_g.item():.4e}, Y={Y_g.item():.4e}, rho={rho_g.item():.4e}")
            else:
                print(f"[pp TF] rho median={rho.median().item():.4e}, "
                      f"sigX med={sigX.median().item():.4e}, "
                      f"|pi| med={pi.norm(dim=1).median().item():.4e}")
    return pi
