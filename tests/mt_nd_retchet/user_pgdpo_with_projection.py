# -*- coding: utf-8 -*-
# tests/mt_nd_retchet/user_pgdpo_with_projection.py
# 목적: 코스테이트와 상태를 받아 (u_pp, C_pp)를 반환하는 미시 최적화기(project_pmp)
#  - 코어/pgdpo_with_projection.py 가 estimate_costates(...)로 만든 costates를 넘겨줌
#  - 본 파일은 *자기 자신*을 import 하면 안 됨 (순환임포트 금지)

from __future__ import annotations
import os, math, torch

# 코어가 요구하는 심볼(사영에 필요한 코스테이트 키)
PP_NEEDS = ("JX",)
__all__  = ["PP_NEEDS", "project_pmp"]

# 베이스 파라미터 (BASE -> PROJECTION 방향 import: 순환 없음)
from user_pgdpo_base import (
    alpha as ALPHA_CONST, Sigma as SIGMA_CONST,
    gamma as GAMMA_CONST, L_cap as L_CAP_CONST, d as D_CONST,
    r as R_CONST, rho as RHO_CONST, device as DEV_CONST, T as T_CONST, m as M_STEPS_CONST,
)

# --------------------------
# 하이퍼/장벽 및 토글
# --------------------------
BARRIER_EPS   = float(os.getenv("PGDPO_BARRIER_EPS", "1e-6"))   # (C-Y) 배리어 세기
MAX_ITERS     = int(os.getenv("PGDPO_PROJ_ITERS",   "30"))
STEP_DECAY    = float(os.getenv("PGDPO_STEP_DECAY", "0.7"))
RIDGE_C       = float(os.getenv("PGDPO_RIDGE_C",    "1e-8"))

# trust region 강도(소비 한 step 변화량 ≤ TR_FRAC * X/τ)
TR_FRAC       = float(os.getenv("PGDPO_TR_FRAC",    "0.25"))

# 소비 상한: (i) 소프트캡 cap*X, (ii) 예산 X/τ — cap이 비어있어도 예산 캡은 항상 켜짐
_cap_env = os.getenv("PGDPO_C_SOFTCAP", "None")
CAP_MULT = None if _cap_env == "None" else float(_cap_env) if _cap_env else None

# 랫쳇 민감도(FOC에 λ_Y 포함 여부 & 소프트 랫쳇 근사에서의 기울기 β)
USE_LAMY    = int(os.getenv("PGDPO_USE_LAMY", "1"))          # 1=λ_Y 사용(기본), 0=미사용
RAT_BETA_PP = float(os.getenv("PGDPO_RAT_BETA_PP", "128.0")) # β↑ ⇒ hard 랫쳇 근사

# u-사영 모드 토글
NO_U_CAP        = int(os.getenv("PGDPO_NO_U_CAP", "0"))          # 1이면 L1축약/음수클램프 비활성(무제약 u)
APPLY_DW_IN_PP  = int(os.getenv("PGDPO_APPLY_DW_IN_PP", "1"))    # 1이면 DW 스케일 적용, 0이면 미적용

# --------------------------
# 내부 유틸
# --------------------------
def _safe_newton_step(g: float, H: float) -> float:
    """뉴턴 스텝: dc = -g/H, H가 비정상/약하면 보수적으로 음수 곡률 가정"""
    H_eff = H
    if not math.isfinite(H_eff) or abs(H_eff) < 1e-12:
        H_eff = -1e-12
    return - g / H_eff

def _clip_c(c: float, y: float, x: float, tau: float, eps: float) -> float:
    """소비 범위를 [y+eps, min(cap*X, X/τ)-eps]로 안전 클리핑"""
    tau_eff    = max(tau, 1e-6)
    c_hi_bgt   = x / tau_eff
    c_hi_soft  = CAP_MULT * x if CAP_MULT is not None else float("inf")
    c_hi       = min(c_hi_bgt, c_hi_soft) - 1e-9
    c_lo       = y + eps
    return max(c_lo, min(c, c_hi))

def _frac_to_boundary(c: float, dc: float, y: float, x: float, tau: float, eps: float) -> float:
    """장벽을 넘지 않도록 한 스텝에서 허용되는 최대 배율(alpha_max) 계산"""
    tau_eff   = max(tau, 1e-6)
    c_hi_bgt  = x / tau_eff
    c_hi_soft = CAP_MULT * x if CAP_MULT is not None else float("inf")
    c_hi      = min(c_hi_bgt, c_hi_soft) - 1e-9
    c_lo      = y + eps
    if dc > 0:
        return max(0.0, 0.99 * (c_hi - c) / max(dc, 1e-12))
    elif dc < 0:
        return max(0.0, 0.99 * (c - c_lo) / max(-dc, 1e-12))
    else:
        return 0.0

def _sigmoid_clip(z: float) -> float:
    """수치 안정화를 위한 시그모이드(스칼라)"""
    z = max(min(z, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-z))

# --------------------------
# 메인: 시점별 PMP 사영
# --------------------------
@torch.no_grad()
def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    입력:
      costates['JX'] = (B,2) = [∂J/∂X, ∂J/∂Y]
      states['X']    = (B,2) = [X, Y], states['TmT']=(B,1)=tau (선택)
    반환:
      (B, d+1) = [u_pp (B,d), C_pp (B,1)]

    구현:
      - C는 뉴턴 스텝으로 FOC를 풀어 제약을 만족하는 pp 해로 계산
      - u는 u_base = (1/γ)Σ^{-1}α 를 기준으로, (선택) DW 스케일 적용 후
        NO_U_CAP=0이면 L1축약/음수클램프, NO_U_CAP=1이면 무제약
    """
    device = DEV_CONST
    dtype  = torch.float32

    alpha = ALPHA_CONST.view(-1).to(device=device, dtype=dtype)
    Sigma = SIGMA_CONST.to(device=device, dtype=dtype)
    gamma = float(GAMMA_CONST)
    d     = int(alpha.numel())

    X   = states["X"].to(device=device, dtype=dtype)            # (B,2) = [X, Y]
    tau = states.get("TmT", None)
    if tau is None:
        tau = torch.zeros((X.size(0), 1), device=device, dtype=dtype)
    else:
        tau = tau.to(device=device, dtype=dtype)

    lam   = costates["JX"].to(device=device, dtype=dtype)       # (B,2)
    lam_x = lam[:, 0:1]
    lam_y = lam[:, 1:2]

    B = X.size(0)
    out = torch.empty(B, d + 1, device=device, dtype=dtype)

    eps_in = float(BARRIER_EPS)
    L_cap  = float(L_CAP_CONST)

    # 기준 포트폴리오(머튼 양수해)
    try:
        u_base = (1.0 / max(gamma, 1e-12)) * torch.linalg.solve(Sigma, alpha)
    except RuntimeError:
        u_base = (1.0 / max(gamma, 1e-12)) * torch.cholesky_solve(alpha.unsqueeze(1), torch.linalg.cholesky(Sigma)).squeeze(1)

    for b in range(B):
        x_b   = float(X[b, 0].item())
        y_b   = float(X[b, 1].item())
        tau_b = float(tau[b, 0].item())
        lamx  = float(lam_x[b].item())
        lamy  = float(lam_y[b].item())

        # ---- C 초기값: FOC 닫힌꼴 워밍업 ----
        t_b   = T_CONST - tau_b
        lamx_eff = max(lamx, 1e-12)
        if abs(gamma - 1.0) > 1e-12:
            c_star = (lamx_eff * math.exp(RHO_CONST * max(0.0, t_b))) ** (-1.0 / gamma)
        else:
            c_star = math.exp(- RHO_CONST * max(0.0, t_b)) / lamx_eff
        c_b = _clip_c(max(c_star, y_b + eps_in), y_b, x_b, tau_b, eps_in)

        # trust-region 파라미터
        tr = TR_FRAC * (x_b / max(tau_b, 1e-6))
        step = 1.0

        # ---- 뉴턴 반복 ----
        for _ in range(MAX_ITERS):
            # 효용 도함수
            if abs(gamma - 1.0) > 1e-12:
                dU  = (c_b) ** (-gamma)
                d2U = - gamma * (c_b) ** (-gamma - 1.0)
            else:
                dU  = 1.0 / max(c_b, 1e-12)
                d2U = - 1.0 / (max(c_b, 1e-12) ** 2)

            # 장벽: C≥Y
            g_bar = (eps_in / (c_b - y_b + 1e-12))
            H_bar = - (eps_in / (c_b - y_b + 1e-12) ** 2)

            # 소프트 랫쳇 민감도(λ_Y 항)
            lamy_term = 0.0
            lamy_hess = 0.0
            if USE_LAMY:
                z    = RAT_BETA_PP * (c_b - y_b)
                sig  = _sigmoid_clip(z)
                sig_h= RAT_BETA_PP * sig * (1.0 - sig)
                lamy_term = lamy * sig
                lamy_hess = lamy * sig_h

            # 장벽: C≤cmax
            g_cap = 0.0; H_cap = 0.0
            if CAP_MULT is not None:
                cmax = CAP_MULT * x_b
                g_cap -= (eps_in / (cmax - c_b + 1e-12))
                H_cap += (eps_in / (cmax - c_b + 1e-12) ** 2)

            disc = math.exp(- RHO_CONST * max(0.0, t_b))

            # FOC (λ_Y on/off 포함)
            g_c  = disc * dU  - lamx + lamy_term + g_bar + g_cap
            H_cc = disc * d2U + lamy_hess + H_bar + H_cap - RIDGE_C

            # 뉴턴 스텝
            dc = _safe_newton_step(g_c, H_cc)
            if not math.isfinite(dc):
                dc = 0.0
            dc   = max(-tr, min(dc, tr))
            if abs(dc) < 1e-12:
                break
            alpha_max = _frac_to_boundary(c_b, dc, y_b, x_b, tau_b, eps_in)
            if alpha_max <= 0.0:
                break
            step = min(step, alpha_max)

            c_new = _clip_c(c_b + step * dc, y_b, x_b, tau_b, eps_in)

            # 수렴
            if abs(c_new - c_b) <= 1e-6 * (1.0 + abs(c_b)):
                c_b = c_new
                break

            # backtracking 비슷한 진행/감쇠
            if (c_new - c_b) * dc > 0:
                c_b = c_new
                step = min(1.0, step * 1.2)
            else:
                step *= STEP_DECAY
                c_b = c_new

        # ---- u 결정 ----
        # u_base: (1/γ)Σ^{-1}α  (기저 방향)
        u_b = u_base.clone()

        # DW 스케일 (옵션)
        if APPLY_DW_IN_PP:
            if abs(R_CONST) < 1e-12:
                AF = tau_b
            else:
                AF = (1.0 - math.exp(- R_CONST * tau_b)) / R_CONST
            pvC = AF * c_b
            dw  = max(0.0, min(1.0, 1.0 - pvC / max(x_b, 1e-12)))
            u_b = u_b * dw

        # 제약 모드: L1축약 + 음수 클램프
        if not NO_U_CAP:
            u_b = torch.clamp(u_b, min=0.0)
            s = float(u_b.sum().item())
            if s >= L_cap:
                u_b *= (L_cap - BARRIER_EPS) / (s + 1e-12)

        out[b, :d] = u_b
        out[b, d:d+1] = torch.tensor([c_b], device=device, dtype=dtype)

    return out
