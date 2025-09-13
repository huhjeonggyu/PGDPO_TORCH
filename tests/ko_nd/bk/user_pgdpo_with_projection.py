# user_pgdpo_with_projection.py
# 역할: 추정된 Co-state를 PMP 공식에 따라 최적 제어 u로 변환 (Y는 항상 존재한다고 가정)

import warnings
import torch
import torch.nn.functional as F

# 모델 파라미터는 user_pgdpo_base(정본)에서 로드
from user_pgdpo_base import (
    alpha, sigma, Sigma, rho_Y, sigma_Y, r, gamma, u_cap
)

PP_NEEDS = ("JX", "JXX", "JXY")

# ──[Σ 관련 캐시]────────────────────────────────────────────────────────────────
# 키: (device, dtype, id(Sigma), round(lam_eff,12))  → { 'mode', 'L' or 'A', 'lam_eff', 'backend' }
_SIGMA_FACT = {}
# 키: (device, dtype) → Σ_YX (k, d)
_SIGMA_YX_CACHE = {}

def _sigma_cache_key(device: torch.device, dtype: torch.dtype, lam_eff: float):
    devkey = (device.type, (device.index if device.type == "cuda" else -1))
    return (devkey, dtype, id(Sigma), round(float(lam_eff), 12))

def _try_set_cuda_backend(next_backend: str) -> None:
    try:
        torch.backends.cuda.preferred_linalg_library(next_backend)
    except Exception:
        pass

def _ensure_sigma_factor(device: torch.device, dtype: torch.dtype,
                         lam: float = 1e-6, tries: int = 3):
    """Σ의 안정적 분해/해결 경로를 준비하고 캐시로 반환."""
    S = Sigma.to(device=device, dtype=dtype)  # (d, d)
    mean_diag = (S.diagonal().mean().detach().abs().cpu().item() + 1e-16)
    lam_eff = lam * mean_diag
    I = torch.eye(S.shape[0], device=device, dtype=dtype)
    S_reg = S + lam_eff * I

    key = _sigma_cache_key(device, dtype, lam_eff)
    if key in _SIGMA_FACT:
        return _SIGMA_FACT[key]

    last_err = None
    if device.type == "cuda":
        _ = torch.backends.cuda.preferred_linalg_library()

    # 1) GPU/현재 백엔드 Cholesky 시도 → 실패 시 백엔드 교체 또는 릿지 증강
    for _ in range(tries):
        try:
            L = torch.linalg.cholesky(S_reg)  # 하삼각
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
                _try_set_cuda_backend(other)  # 백엔드 전환 후 재시도
            else:
                lam_eff *= 10.0               # SPD 불량 등 → 릿지 증강 후 재시도
                S_reg = S + lam_eff * I
                key = _sigma_cache_key(device, dtype, lam_eff)

    # 2) CPU Cholesky 폴백
    try:
        L_cpu = torch.linalg.cholesky(S_reg.to("cpu"))
        ent = {"mode": "cpu_chol", "L": L_cpu, "lam_eff": lam_eff, "backend": "cpu"}
        _SIGMA_FACT[key] = ent
        warnings.warn("[PGDPO] Σ Cholesky를 CPU로 폴백했습니다. (속도 저하 가능)")
        return ent
    except RuntimeError:
        pass

    # 3) CPU LU/solve 최후 폴백
    A_cpu = S_reg.to("cpu")
    ent = {"mode": "cpu_solve", "A": A_cpu, "lam_eff": lam_eff, "backend": "cpu"}
    _SIGMA_FACT[key] = ent
    warnings.warn("[PGDPO] Σ가 Cholesky에 부적합하여 CPU LU solve로 폴백합니다. (가장 느림)")
    return ent

def clear_sigma_cache():
    """필요 시 외부에서 캐시를 비워 런 간 간섭을 방지."""
    _SIGMA_FACT.clear()

def _sigma_solve(rhs: torch.Tensor, lam: float = 1e-6) -> torch.Tensor:
    """
    rhs: (B, d).  Σ x = rhs^T 를 푼 뒤 (B, d)로 반환.
    lam: 릿지 수축 강도(조건수 완화용). 단위 일관성 위해 mean(diag(Σ))에 비례.
    """
    device, dtype = rhs.device, rhs.dtype
    ent = _ensure_sigma_factor(device, dtype, lam=lam)
    rhsT = rhs.T  # (d, B)

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
    """Σ_YX를 dtype/device 맞춰 1회 생성하여 캐시."""
    key = (device, dtype)
    if key in _SIGMA_YX_CACHE:
        return _SIGMA_YX_CACHE[key]
    Sigma_YX = (torch.diag(sigma.to(device=device, dtype=dtype))
                @ rho_Y.to(device=device, dtype=dtype)
                @ sigma_Y.to(device=device, dtype=dtype)).T  # (k, d)
    _SIGMA_YX_CACHE[key] = Sigma_YX
    return Sigma_YX

# ──[소프트 역수 파라미터]──────────────────────────────────────────────────────
_SOFT_INV_C = 1e-4     # δ = C * median(|X*JXX|)
_SOFT_INV_FLOOR = 1e-6 # δ 하한 (절대 스케일)

def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    u*(t, X, Y) = soft_inv(X*J_XX) · solve(Σ, [ J_X · (μ - r) + J_XY · Σ_YX ])
      - J_X   : ∂U/∂X  (shape: B×1)
      - J_XX  : ∂^2U/∂X^2 (shape: B×1)
      - J_XY  : ∂^2U/(∂X∂Y) (shape: B×k)
      - μ - r : F.linear(Y, alpha) * sigma  (shape: B×d)  # alpha: (d×k)
      - Σ_YX  : (diag(sigma) @ rho_Y @ sigma_Y)^T  (shape: k×d)
      - solve : Cholesky 기반 linsolve (백엔드/CPU 폴백 포함)
      - soft_inv: -(x)/(x^2 + δ^2), δ = C·median(|x|) with floor
    """
    if "Y" not in states:
        raise ValueError("project_pmp: states['Y']가 필요합니다 (Y는 항상 존재해야 함).")
    if not all(k in costates for k in PP_NEEDS):
        raise ValueError("project_pmp: costates에 'JX', 'JXX', 'JXY'가 모두 필요합니다.")

    X   = states["X"]      # (B,1)
    Y   = states["Y"]      # (B,k)
    JX  = costates["JX"]   # (B,1)
    JXX = costates["JXX"]  # (B,1)
    JXY = costates["JXY"]  # (B,k)

    device, dtype = Y.device, Y.dtype

    # ──[분모 안정화: 소프트 역수 + 바닥]────────────────────────────────────────
    denom = X * JXX  # (B,1)
    with torch.no_grad():
        scale = denom.abs().median(dim=0).values  # (1,1)
        # 너무 작은 배치에서 scale≈0이면 floor로 방어
        delta = torch.maximum(_SOFT_INV_C * scale,
                              torch.full_like(scale, _SOFT_INV_FLOOR))
    delta = delta.to(device=device, dtype=dtype)
    scalar = - denom / (denom.square() + delta.square())  # (B,1)

    # ──[Myopic + Hedging]──────────────────────────────────────────────────────
    mu_minus_r = F.linear(Y, alpha.to(device=device, dtype=dtype)) \
                 * sigma.to(device=device, dtype=dtype)               # (B,d)
    myopic = JX * mu_minus_r                                          # (B,d)

    Sigma_YX = _Sigma_YX_for(device, dtype)                           # (k,d)
    hedging = JXY @ Sigma_YX                                          # (B,k)@(k,d) → (B,d)

    # ──[선형계 풀이 + 스칼라 계수]────────────────────────────────────────────
    bracket = myopic + hedging                                        # (B,d)
    u_bar = _sigma_solve(bracket)                                     # (B,d)
    u = scalar * u_bar                                                # (B,d)

    return u

