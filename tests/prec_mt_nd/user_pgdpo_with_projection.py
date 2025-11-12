# user_pgdpo_with_projection.py — Dollar PMP projection (ND), with median-based student φ estimator (no EMA/stride/backfill)
# -----------------------------------------------------------------------------
# ρ modes:
#   PGDPO_RHO_MODE = student | regress | ratio
#     student (default): estimate φ schedule once per policy version by rolling out
#                        CURRENT_POLICY, collecting per-time {Jx, Jxx} over the whole batch (RAM),
#                        and using robust (median/MoM) reductions to form φ(t_k).
#     regress:           JX ≈ a·X + b (per-batch) ⇒ rho = φ - X,   φ = -b/a
#     ratio:             rho = -JX / JXX with safe denominator
#
# Other toggles:
#   PGDPO_DENOM_MODE   = neg-softplus | tikhonov  (ratio 모드에만 영향)
#   PGDPO_B_MC_COSTATE = 1024 (default) — student rollouts batch size
#   PGDPO_TAU_ABS      = 1e-3 (default) — absolute ridge (sign-preserving) to stabilize denominators
#   PGDPO_ROBUST       = median | mean | mom (default median) — batch reducer
#   PGDPO_MOM_GROUPS   = 8     — groups for median-of-means when PGDPO_ROBUST=mom
#   PGDPO_PHI_REFRESH  = 0/1  — force re-estimation of φ next call
#   PGDPO_DEBUG        = 0/1  — print quick stats once
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, math
from typing import Optional, List

import torch
import torch.nn.functional as F

from user_pgdpo_base import (
    Sigma, Sigma_chol, alpha, pi_cap, device,
    T, m, r, x0, gamma,
    CURRENT_POLICY, POLICY_VERSION
)

# 코어가 요구하는 코스테이트/상태 키
PP_NEEDS = {"costates": ["JX","JXX"], "states": ["X","TmT"]}

# G^T 방향 벡터(1×d), 달러 컨트롤
_Gd  = torch.linalg.solve(Sigma, alpha.view(-1,1)).view(1,-1).to(device=device, dtype=Sigma.dtype)
_CAP = float(pi_cap) if pi_cap is not None else 0.0

_DEN      = os.getenv("PGDPO_DENOM_MODE", "neg-softplus").lower()
_RHO_MODE = os.getenv("PGDPO_RHO_MODE", "student").lower()        # 'student' | 'regress' | 'ratio'
_DEBUG    = int(os.getenv("PGDPO_DEBUG", "0"))

# student estimator knobs
_BMC     = int(os.getenv("PGDPO_B_MC_COSTATE", "1024"))
_TABS    = float(os.getenv("PGDPO_TAU_ABS", "1e-3"))
_ROBUST  = os.getenv("PGDPO_ROBUST", "median").lower()  # median | mean | mom
_MOM_G   = int(os.getenv("PGDPO_MOM_GROUPS", "8"))

# φ 스케줄 캐시
_PHI_CACHE: Optional[torch.Tensor] = None   # (m,)
_PHI_CACHE_VER = -1

def _extract(costates: dict):
    """코스테이트 dict에서 (JX, JXX)를 추출. 대체 키(J1,J2)도 허용."""
    if "JX" in costates and "JXX" in costates:
        return costates["JX"], costates["JXX"]
    if "J1" in costates and "J2" in costates:
        return costates["J1"], costates["J2"]
    raise KeyError("costates must contain JX/JXX or J1/J2")

def _stabilize(JXX: torch.Tensor) -> torch.Tensor:
    """ratio 모드 분모 안정화: neg-softplus 또는 부호보존형 tikhonov."""
    if _DEN.startswith("neg"):   # strictly negative
        return -F.softplus(-JXX) - 1e-8
    with torch.no_grad():        # tikhonov (sign-preserving)
        scale = torch.quantile(JXX.abs().flatten(), 0.50).clamp_min(1e-12)
    lam = 0.05 * scale
    denom = JXX + lam * torch.sign(JXX)
    return torch.where(denom.abs() < 1e-10, denom.sign()*1e-10, denom)

def _rho_ratio(JX: torch.Tensor, JXX: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    denom = _stabilize(JXX)
    # 절대 릿지 추가(부호 보존)
    denom = denom + _TABS * torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
    return (-JX) / denom

def _rho_regress(JX: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """JX ≈ a·X + b 회귀로 φ 추정 → ρ = φ − X. a<0로 오목성 보정."""
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

def _dbg(X: torch.Tensor, rho: torch.Tensor) -> None:
    """디버그: φ̂와 X의 상관 출력(한 번만)."""
    if not _DEBUG: return
    phi_hat = rho + X
    a = phi_hat.flatten(); b = X.flatten()
    am = a - a.mean(); bm = b - b.mean()
    corr = (am @ bm) / ((am.norm()*bm.norm()).clamp_min(1e-12))
    print(f"[pp-debug] RHO_MODE={_RHO_MODE}  corr(phi_hat,X)={float(corr):.4g}")
    os.environ["PGDPO_DEBUG"] = "0"  # print once

def _reduce_robust_1d(x: torch.Tensor) -> torch.Tensor:
    """배치 1D 텐서에 강건 리듀서 적용."""
    x = x.view(-1)
    if _ROBUST == "mean":
        return x.mean()
    if _ROBUST == "mom":
        g = max(1, min(_MOM_G, x.numel()))
        xs = x[: (x.numel() // g) * g].view(g, -1)
        means = xs.mean(dim=1)
        return means.median()
    # default: median
    return x.median()

@torch.enable_grad()
def _estimate_phi_schedule_student(policy) -> torch.Tensor:
    """
    student 레시피로 φ 스케줄 (m,) 추정 (원샷, EMA/스트라이드/백필 없음):
    - 큰 배치 B=_BMC로 한 번 롤아웃
    - 각 시점 k에서 per-sample Jx, Jxx를 CPU RAM에 저장(detach/cpu)
    - 시점별 중앙값(또는 MoM/평균)으로 λ_M, λ_V와 결합해 φ_k 계산
    """
    assert policy is not None, "CURRENT_POLICY is None; cannot estimate φ."
    dt = T / m
    B = _BMC

    # 상태 롤아웃 (X_t graph 유지)
    x = torch.full((B,1), x0, device=device, dtype=torch.float32, requires_grad=True)
    xs: List[torch.Tensor] = []
    for k in range(m):
        t = (k / m) * torch.ones(B,1, device=device)
        tau = T - t
        pi = policy(X=x, TmT=tau, t=t)                       # (B,d)
        drift = r * x + (pi * alpha).sum(dim=1, keepdim=True)
        noise_vec = (pi @ Sigma_chol)                        # (B,d)
        dW = torch.randn(B, Sigma_chol.shape[0], device=device)
        x = x + drift * dt + (noise_vec * dW).sum(dim=1, keepdim=True) * math.sqrt(dt)
        xs.append(x)

    # 목적함수 J~ (pre-commitment MV)
    xT = xs[-1]
    M = xT.mean()                     # (on device)
    Var = (xT*xT).mean() - M*M
    Jt = M - 0.5*gamma*Var + 0.5*gamma*(M*M)

    # 버퍼(시점별 리스트) — CPU RAM에 저장
    buf_jx:  List[List[torch.Tensor]] = [ [] for _ in range(m) ]
    buf_jxx: List[List[torch.Tensor]] = [ [] for _ in range(m) ]

    # 1차: Jx per-time, per-sample — 2차를 위해 create_graph=True로 gk 생성
    jx_all = torch.autograd.grad(Jt, xs, retain_graph=True, allow_unused=True, create_graph=True)
    for k, gk in enumerate(jx_all):
        assert gk is not None, f"[student] Jx at k={k} is None (graph broken)."
        buf_jx[k].append(gk.detach().cpu().view(-1))

    # 2차: Jxx — 전 시점 계산 (백필 제거)
    for k in range(m):
        gk = jx_all[k]
        hk = torch.autograd.grad(gk.sum(), xs[k], retain_graph=True, allow_unused=False)[0]
        assert hk is not None, f"[student] Jxx at k={k} is None (graph broken)."
        buf_jxx[k].append(hk.detach().cpu().view(-1))

    # 시점별 강건 요약 (CPU 텐서)
    lam_med = torch.zeros(m, dtype=torch.float32, device="cpu")
    hxx_med = torch.zeros(m, dtype=torch.float32, device="cpu")

    for k in range(m):
        lam_med[k] = _reduce_robust_1d(torch.cat(buf_jx[k],  dim=0)).to(torch.float32)
        hxx_med[k] = _reduce_robust_1d(torch.cat(buf_jxx[k], dim=0)).to(torch.float32)

    # <<< 디바이스 정렬: 통계를 M의 디바이스로 옮긴 뒤 연산 >>>
    dev = M.device
    lam_med = lam_med.to(dev)
    hxx_med = hxx_med.to(dev)

    # λ_M, λ_V
    lambda_M = 1.0 + gamma * M.detach()
    lambda_V = -0.5 * gamma

    den = hxx_med + 2.0 * lambda_V
    # 절대 릿지(부호 보존) 추가
    den = den + _TABS * torch.where(den >= 0, torch.ones_like(den), -torch.ones_like(den))

    phi_sched = - (lam_med + lambda_M) / den                      # (m,) on dev

    # 원샷 반환 (EMA 없음)
    return phi_sched.detach().to("cpu")

def _get_phi_for_tau(policy_ver: int, tau: torch.Tensor) -> torch.Tensor:
    """캐시된 φ에서 τ에 해당하는 값을 반환(정책 버전 바뀌면 재추정)."""
    global _PHI_CACHE, _PHI_CACHE_VER
    need_refresh = (_PHI_CACHE is None) or (_PHI_CACHE_VER != policy_ver) or (os.getenv("PGDPO_PHI_REFRESH","0") == "1")
    if need_refresh:
        from user_pgdpo_base import CURRENT_POLICY  # latest ref
        # project_pmp는 no_grad로 불리므로 내부에서 enable_grad로 실행
        with torch.enable_grad():
            _PHI_CACHE = _estimate_phi_schedule_student(CURRENT_POLICY)    # (m,) CPU
        _PHI_CACHE_VER = policy_ver
        os.environ["PGDPO_PHI_REFRESH"] = "0"

    # 인덱싱 디바이스 정렬
    if _PHI_CACHE.device != tau.device:
        _PHI_CACHE = _PHI_CACHE.to(tau.device)

    dt = T / m
    k_idx = torch.clamp(((T - tau) / dt).floor().long(), 0, m-1).view(-1)   # (B,)
    phi_k = _PHI_CACHE[k_idx].view(-1,1)                                    # (B,1) on tau.device
    return phi_k

@torch.no_grad()
def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """PMP 사영: ρ를 고르고 π=ρ·G 방향으로 정사영. 항상 X-스케일만 사용."""
    J1, J2 = _extract(costates)                             # (B,1), (B,1)
    X   = states["X"].to(J1.dtype).to(J1.device)            # (B,1)
    tau = states["TmT"].to(J1.dtype).to(J1.device)          # (B,1)
    JX, JXX = J1, J2

    if _RHO_MODE.startswith("student"):
        from user_pgdpo_base import POLICY_VERSION as _PV
        phi = _get_phi_for_tau(_PV, tau)                    # (B,1) on tau.device
        rho = (phi - X)                                     # (B,1)
    elif _RHO_MODE.startswith("regress"):
        rho = _rho_regress(JX, X)
    else:
        rho = _rho_ratio(JX, JXX, X)

    if _DEBUG:
        _dbg(X, rho)

    pi = rho @ _Gd.to(J1.device, J1.dtype)                  # (B,1)@(1,d)->(B,d)
    if _CAP and _CAP > 0:
        pi = torch.clamp(pi, -_CAP, _CAP)
    return pi
