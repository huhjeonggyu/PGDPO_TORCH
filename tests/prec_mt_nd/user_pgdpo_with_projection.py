# user_pgdpo_with_projection.py — Dollar PMP projection (ND), with student φ estimator
# -----------------------------------------------------------------------------
# ρ modes:
#   PGDPO_RHO_MODE = student | regress | ratio
#     student (default): estimate φ schedule once per policy version by rolling out
#                        CURRENT_POLICY and computing λ_M, λ_V and diagonal costates.
#     regress:           JX ≈ a·X + b (per-batch) ⇒ rho = φ - X,   φ = -b/a
#     ratio:             rho = -JX / JXX with safe denominator
#
# Other toggles:
#   PGDPO_COSTATES_WRT = X | logX  (default X; if logX then JX=Jz/X, JXX=(Jzz-Jz)/X^2)
#   PGDPO_DENOM_MODE   = neg-softplus | tikhonov  (ratio 모드에만 영향)
#   PGDPO_B_MC_COSTATE = 1024 (default) — student rollouts
#   PGDPO_STRIDE_H2    = 1    (default) — student Hessian stride
#   PGDPO_EMA_BETA     = 0.75 (default) — φ EMA smoothing
#   PGDPO_TAU_ABS      = 1e-3 (default) — absolute ridge in denominator
#   PGDPO_PHI_REFRESH  = 0/1  — force re-estimation of φ next call
#   PGDPO_DEBUG        = 0/1  — print quick stats once
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, math
import torch
import torch.nn.functional as F

from user_pgdpo_base import (
    Sigma, Sigma_chol, alpha, pi_cap, device,
    T, m, r, x0, gamma,
    CURRENT_POLICY, POLICY_VERSION
)

PP_NEEDS = {"costates": ["JX","JXX"], "states": ["X","TmT"]}

_Gd  = torch.linalg.solve(Sigma, alpha.view(-1,1)).view(1,-1).to(device=device, dtype=Sigma.dtype)
_CAP = float(pi_cap) if pi_cap is not None else 0.0
_EPS = 1e-12

_WRT      = os.getenv("PGDPO_COSTATES_WRT", "X").lower()          # 'x' | 'logx'
_DEN      = os.getenv("PGDPO_DENOM_MODE", "neg-softplus").lower()
_RHO_MODE = os.getenv("PGDPO_RHO_MODE", "student").lower()        # 'student' | 'regress' | 'ratio'
_DEBUG    = int(os.getenv("PGDPO_DEBUG", "0"))

# student estimator knobs
_BMC   = int(os.getenv("PGDPO_B_MC_COSTATE", "1024"))
_STRIDE= int(os.getenv("PGDPO_STRIDE_H2", "1"))
_EMA_B = float(os.getenv("PGDPO_EMA_BETA", "0.75"))
_TABS  = float(os.getenv("PGDPO_TAU_ABS", "1e-3"))

# cache for φ schedule
_PHI_CACHE = None
_PHI_CACHE_VER = -1

def _extract(costates):
    if "JX" in costates and "JXX" in costates: return costates["JX"], costates["JXX"]
    if "J1" in costates and "J2" in costates:  return costates["J1"], costates["J2"]
    raise KeyError("costates must contain JX/JXX or J1/J2")

def _to_x(J1, J2, X):
    Xc = X.clamp_min(_EPS)
    JX  = J1 / Xc
    JXX = (J2 - J1) / (Xc * Xc)
    return JX, JXX

def _stabilize(JXX):
    if _DEN.startswith("neg"):   # strictly negative
        return -F.softplus(-JXX) - 1e-8
    with torch.no_grad():        # tikhonov (sign-preserving)
        scale = torch.quantile(JXX.abs().flatten(), 0.50).clamp_min(1e-12)
    lam = 0.05 * scale
    denom = JXX + lam * torch.sign(JXX)
    return torch.where(denom.abs() < 1e-10, denom.sign()*1e-10, denom)

def _rho_ratio(JX, JXX, X):
    denom = _stabilize(JXX)
    return (-JX) / denom

def _rho_regress(JX, X):
    # Fit JX ≈ a·X + b via least-squares on the current batch
    x = X.flatten()
    y = JX.flatten()
    x_mean = x.mean()
    y_mean = y.mean()
    var_x  = (x - x_mean).pow(2).mean().clamp_min(1e-12)
    cov_xy = ((x - x_mean) * (y - y_mean)).mean()
    a = cov_xy / var_x
    a = -torch.abs(a)   # concavity
    b = y_mean - a * x_mean
    phi = (-b / a).view(1,1)
    rho = (phi - X)
    return rho.detach()

def _dbg(X, rho):
    if not _DEBUG: return
    phi_hat = rho + X
    a = phi_hat.flatten(); b = X.flatten()
    am = a - a.mean(); bm = b - b.mean()
    corr = (am @ bm) / ((am.norm()*bm.norm()).clamp_min(1e-12))
    print(f"[pp-debug] RHO_MODE={_RHO_MODE}  corr(phi_hat,X)={float(corr):.4g}")
    os.environ["PGDPO_DEBUG"] = "0"  # print once

@torch.enable_grad()
def _estimate_phi_schedule_student(policy) -> torch.Tensor:
    """Return φ schedule of shape (m,) using the student recipe (on-device)."""
    assert policy is not None, "CURRENT_POLICY is None; cannot estimate φ."
    dt = T / m
    B = _BMC
    x = torch.full((B,1), x0, device=device, dtype=torch.float32, requires_grad=True)
    xs = []
    for k in range(m):
        t = (k / m) * torch.ones(B,1, device=device)
        tau = T - t
        pi = policy(X=x, TmT=tau, t=t)                       # (B,d)
        drift = r * x + (pi * alpha).sum(dim=1, keepdim=True)
        noise_vec = (pi @ Sigma_chol)                        # (B,d)
        dW = torch.randn(B, Sigma_chol.shape[0], device=device)
        x = x + drift * dt + (noise_vec * dW).sum(dim=1, keepdim=True) * math.sqrt(dt)
        xs.append(x)

    xT = xs[-1]
    M = xT.mean()
    Var = (xT*xT).mean() - M*M
    # J~ = M - 0.5γ Var + 0.5γ M^2
    Jt = M - 0.5*gamma*Var + 0.5*gamma*(M*M)

    lam_all = torch.autograd.grad(Jt, xs, retain_graph=True, allow_unused=True)
    lam_list = [(torch.zeros_like(xs[0]) if g is None else g) for g in lam_all]
    lam_mean = torch.stack([g.mean() for g in lam_list], dim=0)      # (m,)

    dlamdx_mean = torch.zeros(m, device=device, dtype=torch.float32)
    last = None
    for k in range(0, m, max(1,_STRIDE)):
        gk = torch.autograd.grad(Jt, xs[k], retain_graph=True, allow_unused=True, create_graph=True)[0]
        if gk is None:
            d2 = torch.tensor(0.0, device=device)
        else:
            hk = torch.autograd.grad(gk.sum(), xs[k], retain_graph=True, allow_unused=True)[0]
            d2 = torch.tensor(0.0, device=device) if hk is None else hk.mean()
        dlamdx_mean[k] = d2
        if last is None: last = d2
        else:
            for j in range(max(0, k-max(1,_STRIDE)+1), k):
                if dlamdx_mean[j].abs() < 1e-12: dlamdx_mean[j] = last
        last = d2
    for j in range(m):
        if dlamdx_mean[j].abs() < 1e-12: dlamdx_mean[j] = last

    lambda_M = 1.0 + gamma * M.detach()
    lambda_V = -0.5 * gamma

    den = dlamdx_mean + 2.0 * lambda_V
    den = den + _TABS * torch.where(den >= 0, torch.ones_like(den), -torch.ones_like(den))

    phi_sched = - (lam_mean + lambda_M) / den                      # (m,)

    out = phi_sched.clone()
    for k in range(1, m):
        out[k] = _EMA_B * out[k] + (1.0 - _EMA_B) * out[k-1]
    return out.detach()  # (m,)

def _get_phi_for_tau(policy_ver: int, tau: torch.Tensor):
    """Return φ(tau) for each sample using cached schedule (re-estimate on policy update)."""
    global _PHI_CACHE, _PHI_CACHE_VER
    need_refresh = (_PHI_CACHE is None) or (_PHI_CACHE_VER != policy_ver) or (os.getenv("PGDPO_PHI_REFRESH","0") == "1")
    if need_refresh:
        from user_pgdpo_base import CURRENT_POLICY  # latest ref
        _PHI_CACHE = _estimate_phi_schedule_student(CURRENT_POLICY).to(device=device)  # (m,)
        _PHI_CACHE_VER = policy_ver
        os.environ["PGDPO_PHI_REFRESH"] = "0"

    dt = T / m
    k_idx = torch.clamp(((T - tau) / dt).floor().long(), 0, m-1).view(-1)   # (B,)
    phi_k = _PHI_CACHE[k_idx].view(-1,1)                                    # (B,1)
    return phi_k

@torch.no_grad()
def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    J1, J2 = _extract(costates)
    X = states["X"].to(J1.dtype).to(J1.device)
    tau = states["TmT"].to(J1.dtype).to(J1.device)

    if _WRT.startswith("log"):
        JX, JXX = _to_x(J1, J2, X)
    else:
        JX, JXX = J1, J2

    if _RHO_MODE.startswith("student"):
        from user_pgdpo_base import POLICY_VERSION as _PV
        phi = _get_phi_for_tau(_PV, tau)             # (B,1)
        rho = (phi - X)                              # (B,1)
    elif _RHO_MODE.startswith("regress"):
        rho = _rho_regress(JX, X)
    else:
        rho = _rho_ratio(JX, JXX, X)

    if _DEBUG:
        _dbg(X, rho)

    pi = rho @ _Gd.to(J1.device, J1.dtype)           # (B,1)@(1,d)->(B,d)
    if _CAP and _CAP > 0:
        pi = torch.clamp(pi, -_CAP, _CAP)
    return pi
