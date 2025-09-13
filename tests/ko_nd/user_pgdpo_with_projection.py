# user_pgdpo_with_projection.py
# 역할: 추정된 Co-state를 PMP 공식에 따라 최적 제어 u로 변환 (Y는 항상 존재한다고 가정)

import warnings
import torch
import torch.nn.functional as F

# 모델 파라미터는 user_pgdpo_base(정본)에서 로드
from user_pgdpo_base import (
    alpha, sigma, Sigma, rho_Y, sigma_Y, r, gamma, u_cap, epsilon
)

PP_NEEDS = ("JX", "JXX", "JXY")

# ──[Σ 관련 캐시]────────────────────────────────────────────────────────────────
_SIGMA_FACT = {}
_SIGMA_YX_CACHE = {}

def _sigma_cache_key(device: torch.device, dtype: torch.dtype, lam_eff: float, eps_eff: float):
    devkey = (device.type, (device.index if device.type == "cuda" else -1))
    return (devkey, dtype, id(Sigma), round(float(lam_eff), 12), round(float(eps_eff), 12))

def _try_set_cuda_backend(next_backend: str) -> None:
    try:
        torch.backends.cuda.preferred_linalg_library(next_backend)
    except Exception:
        pass

def _ensure_sigma_factor(device: torch.device, dtype: torch.dtype,
                         lam: float = 1e-6, tries: int = 3):
    """
    Σ의 안정적 분해/해결 경로를 준비하고 캐시로 반환.
    여기서는 utility-scaled quadratic penalty 반영: (γ Σ + ε I + λ_eff I).
    """
    S = Sigma.to(device=device, dtype=dtype)  # (d,d)
    mean_diag = (S.diagonal().mean().detach().abs().cpu().item() + 1e-16)
    lam_eff = lam * mean_diag
    I = torch.eye(S.shape[0], device=device, dtype=dtype)
    # Regularized Sigma (γ Σ + ε I + λ_eff I)
    S_reg = gamma * S + epsilon * I + lam_eff * I

    key = _sigma_cache_key(device, dtype, lam_eff, epsilon)
    if key in _SIGMA_FACT:
        return _SIGMA_FACT[key]

    last_err = None
    if device.type == "cuda":
        _ = torch.backends.cuda.preferred_linalg_library()

    for _ in range(tries):
        try:
            L = torch.linalg.cholesky(S_reg)
            ent = {
                "mode": ("gpu_chol" if device.type == "cuda" else "cpu_chol"),
                "L": L,
                "lam_eff": lam_eff,
                "backend": (torch.backends.cuda.preferred_linalg_library()
                            if device.type == "cuda" else "cpu"),
            }
            _SIGMA_FACT[key] = ent
            return ent
        except RuntimeError as e:
            last_err = e
            msg = str(e).lower()
            if device.type == "cuda" and ("cusolver" in msg or "magma" in msg):
                other = ("magma"
                         if torch.backends.cuda.preferred_linalg_library() == "cusolver"
                         else "cusolver")
                _try_set_cuda_backend(other)
            else:
                lam_eff *= 10.0
                S_reg = gamma * S + epsilon * I + lam_eff * I
                key = _sigma_cache_key(device, dtype, lam_eff, epsilon)

    try:
        L_cpu = torch.linalg.cholesky(S_reg.to("cpu"))
        ent = {"mode": "cpu_chol", "L": L_cpu, "lam_eff": lam_eff, "backend": "cpu"}
        _SIGMA_FACT[key] = ent
        warnings.warn("[PGDPO] Σ Cholesky를 CPU로 폴백했습니다.")
        return ent
    except RuntimeError:
        pass

    A_cpu = S_reg.to("cpu")
    ent = {"mode": "cpu_solve", "A": A_cpu, "lam_eff": lam_eff, "backend": "cpu"}
    _SIGMA_FACT[key] = ent
    warnings.warn("[PGDPO] Σ가 Cholesky에 부적합하여 CPU LU solve로 폴백합니다.")
    return ent

def clear_sigma_cache():
    _SIGMA_FACT.clear()

def _sigma_solve(rhs: torch.Tensor, lam: float = 1e-6) -> torch.Tensor:
    """
    rhs: (B,d). (γ Σ + ε I + λ_eff I) u = rhs^T 를 푼다.
    """
    device, dtype = rhs.device, rhs.dtype
    ent = _ensure_sigma_factor(device, dtype, lam=lam)
    rhsT = rhs.T
    if ent["mode"] == "gpu_chol":
        xT = torch.cholesky_solve(rhsT, ent["L"])
        return xT.T
    elif ent["mode"] == "cpu_chol":
        xT_cpu = torch.cholesky_solve(rhsT.to("cpu"), ent["L"])
        return xT_cpu.T.to(device=device, dtype=dtype)
    else:
        xT_cpu = torch.linalg.solve(ent["A"], rhsT.to("cpu"))
        return xT_cpu.T.to(device=device, dtype=dtype)

def _Sigma_YX_for(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    if key in _SIGMA_YX_CACHE:
        return _SIGMA_YX_CACHE[key]
    # Σ_YX = (diag(σ) ρ_Y σ_Y)ᵀ ∈ ℝ^{k×d}
    Sigma_YX = (torch.diag(sigma.to(device=device, dtype=dtype))
                @ rho_Y.to(device=device, dtype=dtype)
                @ sigma_Y.to(device=device, dtype=dtype)).T  # (k,d)
    _SIGMA_YX_CACHE[key] = Sigma_YX
    return Sigma_YX

_SOFT_INV_C = 1e-4
_SOFT_INV_FLOOR = 1e-6

def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    Utility-scaled penalty 버전의 PMP 투영.
    목표식: u ≈ (γΣ+εI)^{-1} [ μ_ex(Y) + (J_{XY}/J_X) Σ_{YX} ].
    구현식: u = soft_inv(J_X) · (γΣ+εI)^{-1} [ J_X μ_ex + J_{XY} Σ_{YX} ],
    여기서 soft_inv(J_X)는 CRRA 동차성 X J_{XX} = -γ J_X를 이용해
    soft_inv(J_X) ≈ (-γ) * soft_inv(X J_{XX}) 로 안정화.  # [FIX]
    """
    if "Y" not in states:
        raise ValueError("project_pmp: states['Y'] 필요.")
    if not all(k in costates for k in PP_NEEDS):
        raise ValueError("project_pmp: costates에 'JX','JXX','JXY' 필요.")

    X   = states["X"]   # (B,1)
    Y   = states["Y"]   # (B,k)
    JX  = costates["JX"]    # (B,1)
    JXX = costates["JXX"]   # (B,1)
    JXY = costates["JXY"]   # (B,k)  == ∂^2 J / ∂x ∂y

    device, dtype = Y.device, Y.dtype

    # ---- soft inverse of J_X using X*J_XX identity: X J_XX = -γ J_X
    denom = X * JXX   # (B,1)
    with torch.no_grad():
        scale = denom.abs().median(dim=0).values
        delta = torch.maximum(_SOFT_INV_C * scale,
                              torch.full_like(scale, _SOFT_INV_FLOOR))
    delta = delta.to(device=device, dtype=dtype)
    # target: 1/JX = -γ / (X JXX). Soft inverse: (-γ) * denom / (denom^2 + δ^2).  # [FIX]
    scalar = (-gamma) * denom / (denom.square() + delta.square())  # (B,1)

    # ---- myopic term: μ_ex = diag(σ) (A Y)  (shape: B×d)
    mu_minus_r = F.linear(Y, alpha.to(device=device, dtype=dtype)) \
                 * sigma.to(device=device, dtype=dtype)

    # ---- hedging term: (J_{XY}/J_X) Σ_{YX}  ≈ soft_inv(J_X)*(J_{XY} Σ_{YX})
    Sigma_YX = _Sigma_YX_for(device, dtype)   # (k,d)
    hedging = JXY @ Sigma_YX                  # (B,d)

    # ---- combine, apply (γΣ+εI)^{-1}
    bracket = JX * mu_minus_r + hedging       # (B,d)
    u_bar = _sigma_solve(bracket)             # (B,d)

    # ---- scale by soft_inv(J_X) and clamp
    u = scalar * u_bar                        # (B,d)
    return torch.clamp(u, -u_cap, u_cap)      # [FIX] 안전 클램프