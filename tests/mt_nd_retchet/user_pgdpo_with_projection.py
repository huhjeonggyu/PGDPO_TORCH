# -*- coding: utf-8 -*-
# 파일: tests/mt_nd_retchet/user_pgdpo_with_projection.py
# 목적: PMP 프로젝션(u & C) — 랫칫(Y), 소프트캡, 예산(X/dt)까지 즉시 반영한 안정판
from __future__ import annotations
import os, math, torch

# 코어가 요구하는 심볼
PP_NEEDS = ("JX",)
__all__   = ["PP_NEEDS", "project_pmp"]

# 베이스 파라미터 (run.py가 TEST_PATH를 sys.path에 추가)
from user_pgdpo_base import (
    alpha as ALPHA_CONST, Sigma as SIGMA_CONST,
    gamma as GAMMA_CONST, L_cap as L_CAP_CONST, d as D_CONST,
    r as R_CONST, rho as RHO_CONST, device as DEV_CONST, T as T_CONST, m as M_STEPS_CONST,
    C_SOFT_BETA as RATCHET_BETA_CONST
)

# 하이퍼/장벽
BARRIER_EPS   = float(os.getenv("PGDPO_BARRIER_EPS", "1e-3"))
MAX_ITERS     = int(os.getenv("PGDPO_PROJ_ITERS", "30"))
STEP_DECAY    = float(os.getenv("PGDPO_STEP_DECAY", "0.7"))
RIDGE_C       = float(os.getenv("PGDPO_RIDGE_C", "1e-8"))

# 소비 상한: (i) 소프트캡 cap*X, (ii) 예산 X/dt — cap이 비어있어도 예산 캡은 항상 켭니다.
_cap_env = os.getenv("PGDPO_C_SOFTCAP", "None")
CAP_MULT = None if _cap_env == "None" else float(_cap_env) if _cap_env else None

def _safe_newton_step(g: float, H: float) -> float:
    # 뉴턴 스텝: dc = -g/H, H가 비정상/약하면 보수적으로 음수 곡률 가정
    H_eff = H
    if not math.isfinite(H_eff) or abs(H_eff) < 1e-12:
        H_eff = -1e-12
    return - g / H_eff

def _clip_c(c: float, y: float, x: float, tau: float, eps: float) -> float:
    """소비 범위를 [y+eps, min(cap*X, X/dt)-eps]로 안전 클리핑"""
    # 예산 캡: dt ≈ tau / m
    dt   = max(tau / max(1, int(M_STEPS_CONST)), 1e-6)
    c_hi_budget = x / dt
    c_hi_soft   = CAP_MULT * x if CAP_MULT is not None else float("inf")
    c_hi        = min(c_hi_budget, c_hi_soft) - 1e-9
    c_lo        = y + eps
    return max(c_lo, min(c, c_hi))

@torch.no_grad()
def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    입력:
      costates['JX'] = (B,2) = [∂J/∂X, ∂J/∂Y]
      states['X']    = (B,2) = [X, Y], states['TmT']=(B,1)=tau (선택)
    반환:
      (B, d+1) = [u_pp (B,d), C_pp (B,1)]
    """
    device = DEV_CONST
    dtype  = torch.float32

    alpha = ALPHA_CONST.view(-1).to(device=device, dtype=dtype)
    Sigma = SIGMA_CONST.to(device=device, dtype=dtype)
    gamma = float(GAMMA_CONST)
    d     = int(alpha.numel())

    X   = states["X"].to(device=device, dtype=dtype)            # (B,2)
    tau = states.get("TmT", None)
    if tau is None:
        tau = torch.zeros((X.size(0), 1), device=device, dtype=dtype)
    else:
        tau = tau.to(device=device, dtype=dtype)

    lam   = costates["JX"].to(device=device, dtype=dtype)
    lam_x = lam[:, 0:1]
    lam_y = lam[:, 1:2]

    B = X.size(0)
    out = torch.empty(B, d + 1, device=device, dtype=dtype)

    eps_in = float(BARRIER_EPS)
    L_cap  = float(L_CAP_CONST)
    beta   = float(RATCHET_BETA_CONST)

    # 기준 포트폴리오(양수화 머튼 + L1축약) — 이후 DW로 스케일
    u_base = (1.0 / max(gamma, 1e-12)) * torch.linalg.solve(Sigma, alpha)
    u_base = torch.clamp(u_base, min=eps_in)
    s0 = float(u_base.sum().item())
    if s0 > 0.0 and s0 > L_cap:
        u_base *= (L_cap / s0)

    for b in range(B):
        x_b   = float(X[b, 0].item())
        y_b   = float(X[b, 1].item())
        tau_b = float(tau[b, 0].item())
        lamx  = float(lam_x[b].item())
        lamy  = float(lam_y[b].item())

        # 초기 c: Y 바로 위
        c_b = _clip_c(max(y_b + eps_in, 0.02 * x_b), y_b, x_b, tau_b, eps_in)
        step = 1.0

        for _ in range(MAX_ITERS):
            # 효용 도함수
            if abs(gamma - 1.0) > 1e-12:
                dU  = (c_b) ** (-gamma)
                d2U = - gamma * (c_b) ** (-gamma - 1.0)
            else:
                dU  = 1.0 / max(c_b, 1e-12)
                d2U = - 1.0 / max(c_b, 1e-12) ** 2

            # soft-ratchet의 ∂Y/∂C = σ(β(C-Y))
            sig  = 1.0 / (1.0 + math.exp(-beta * (c_b - y_b)))
            g_bar = (eps_in / (c_b - y_b + 1e-12))
            H_bar = - (eps_in / (c_b - y_b + 1e-12) ** 2)

            # (선택) 소프트캡의 장벽(곡률>0)
            g_cap = 0.0; H_cap = 0.0
            if CAP_MULT is not None:
                cmax = CAP_MULT * x_b
                g_cap -= (eps_in / (cmax - c_b + 1e-12))
                H_cap += (eps_in / (cmax - c_b + 1e-12) ** 2)

            # 할인因子(절대시간 t = T - τ)
            disc = math.exp(- RHO_CONST * max(0.0, (T_CONST - tau_b)))

            # FOC in C: e^{-ρt}u'(C) - λ_X + λ_Y σ + barriers
            g_c  = disc * dU - lamx + lamy * sig + g_bar + g_cap
            H_cc = disc * d2U + lamy * beta * sig * (1.0 - sig) + H_bar + H_cap - RIDGE_C

            dc = _safe_newton_step(g_c, H_cc)
            if abs(dc) < 1e-12: break

            # 라인서치(단순 감쇠) + 경계 내부 보정
            c_new = _clip_c(c_b + step * dc, y_b, x_b, tau_b, eps_in)
            # 수렴 기준
            if abs(c_new - c_b) <= 1e-6 * (1.0 + abs(c_b)):
                c_b = c_new
                break

            # 진행/감쇠
            if (c_new - c_b) * dc > 0:  # 같은 방향이면 유지
                c_b = c_new
                step = min(1.0, step * 1.2)
            else:                        # 반대면 감쇠
                step *= STEP_DECAY
                c_b = c_new

        # 최종 u: DW 스케일
        AF   = (tau_b if abs(R_CONST) < 1e-12 else (1.0 - math.exp(- R_CONST * tau_b)) / R_CONST)
        pvC  = AF * c_b
        dw   = max(0.0, min(1.0, 1.0 - pvC / max(x_b, 1e-12)))
        u_b  = u_base * dw
        # L1 내부 유지
        s = float(u_b.sum().item())
        if s >= L_cap:
            u_b *= (L_cap - eps_in) / (s + 1e-12)
        u_b = torch.clamp(u_b, min=eps_in)

        out[b, :d] = u_b
        out[b, d:d+1] = torch.tensor([c_b], device=device, dtype=dtype)

    return out
