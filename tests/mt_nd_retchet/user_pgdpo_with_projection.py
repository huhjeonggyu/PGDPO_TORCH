# -*- coding: utf-8 -*-
# 파일: tests/mt_nd_retchet/user_pgdpo_with_projection.py
# 목적: PMP 프로젝션(u & C) — 랫칫(Y), 소프트캡, 예산(X/τ)까지 즉시 반영한 안정판
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

# --------------------------
# 하이퍼/장벽 및 토글
# --------------------------
BARRIER_EPS   = float(os.getenv("PGDPO_BARRIER_EPS", "1e-6"))   # 장벽은 약하게
MAX_ITERS     = int(os.getenv("PGDPO_PROJ_ITERS",   "30"))
STEP_DECAY    = float(os.getenv("PGDPO_STEP_DECAY", "0.7"))
RIDGE_C       = float(os.getenv("PGDPO_RIDGE_C",    "1e-8"))

# 랫칫 ∂Y/∂C 항 사용 여부(0=끄기 권장, 1=켜기)
USE_LAMY      = int(os.getenv("PGDPO_USE_LAMY",     "0"))

# trust region 강도(소비 한 step 변화량 ≤ TR_FRAC * X/τ)
TR_FRAC       = float(os.getenv("PGDPO_TR_FRAC",    "0.2"))

# 소비 상한: (i) 소프트캡 cap*X, (ii) 예산 X/τ — cap이 비어있어도 예산 캡은 항상 켭니다.
_cap_env = os.getenv("PGDPO_C_SOFTCAP", "None")
CAP_MULT = None if _cap_env == "None" else float(_cap_env) if _cap_env else None

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

        # ----------------------------
        # 소비 초기값: FOC 닫힌꼴로 워밍업
        # e^{-ρ t} U'(C) = λ_X  ⇒  C* = (e^{ρ t} λ_X)^(-1/γ)
        # (t = T - tau)
        # ----------------------------
        t_b   = T_CONST - tau_b
        disc  = math.exp(- RHO_CONST * max(0.0, t_b))
        lamx_eff = max(lamx, 1e-12)  # 음수/제로 방지
        if abs(gamma - 1.0) > 1e-12:
            c_star = (lamx_eff * math.exp(RHO_CONST * max(0.0, t_b))) ** (-1.0 / gamma)
        else:
            c_star = disc / lamx_eff
        c_b = _clip_c(max(c_star, y_b + eps_in), y_b, x_b, tau_b, eps_in)

        # trust-region: 한 스텝 소비 변화 상한(예산 상한의 TR_FRAC)
        tr = TR_FRAC * (x_b / max(tau_b, 1e-6))
        step = 1.0

        for _ in range(MAX_ITERS):
            # 효용 도함수
            if abs(gamma - 1.0) > 1e-12:
                dU  = (c_b) ** (-gamma)
                d2U = - gamma * (c_b) ** (-gamma - 1.0)
            else:
                dU  = 1.0 / max(c_b, 1e-12)
                d2U = - 1.0 / (max(c_b, 1e-12) ** 2)

            # 랫칫 장벽(+ε ln(C-Y)) ⇒ g_bar=ε/(C-Y), H_bar= -ε/(C-Y)^2
            g_bar = (eps_in / (c_b - y_b + 1e-12))
            H_bar = - (eps_in / (c_b - y_b + 1e-12) ** 2)

            # (선택) ∂Y/∂C 항 — 기본은 끔(USE_LAMY=0)
            sig = 0.0; sig_h = 0.0
            if USE_LAMY:
                sig   = 1.0 / (1.0 + math.exp(-beta * (c_b - y_b)))
                sig_h = beta * sig * (1.0 - sig)

            # (선택) 소프트캡 장벽(+ε ln(cmax-C))
            g_cap = 0.0; H_cap = 0.0
            if CAP_MULT is not None:
                cmax = CAP_MULT * x_b
                g_cap -= (eps_in / (cmax - c_b + 1e-12))
                H_cap += (eps_in / (cmax - c_b + 1e-12) ** 2)

            # FOC in C: e^{-ρt}U'(C) - λ_X + [λ_Y σ] + barriers
            g_c  = disc * dU - lamx + (lamy * sig if USE_LAMY else 0.0) + g_bar + g_cap
            H_cc = disc * d2U + (lamy * sig_h if USE_LAMY else 0.0) + H_bar + H_cap - RIDGE_C

            dc = _safe_newton_step(g_c, H_cc)
            if not math.isfinite(dc):
                dc = 0.0
            # trust-region & fraction-to-boundary
            dc   = max(-tr, min(dc, tr))
            if abs(dc) < 1e-12:
                break
            alpha_max = _frac_to_boundary(c_b, dc, y_b, x_b, tau_b, eps_in)
            if alpha_max <= 0.0:
                break
            step = min(step, alpha_max)

            c_new = _clip_c(c_b + step * dc, y_b, x_b, tau_b, eps_in)

            # 수렴 기준
            if abs(c_new - c_b) <= 1e-6 * (1.0 + abs(c_b)):
                c_b = c_new
                break

            # 진행/감쇠
            if (c_new - c_b) * dc > 0:  # 같은 방향이면 유지/약간 증대
                c_b = c_new
                step = min(1.0, step * 1.2)
            else:                        # 반대면 감쇠
                step *= STEP_DECAY
                c_b = c_new

        # ----------------------------
        # 최종 u: DW 스케일링 (1 - PV(C)/X)
        # PV(C) = C * ((1 - e^{-rτ})/r) 혹은 r≈0이면 τ*C
        # ----------------------------
        if abs(R_CONST) < 1e-12:
            AF = tau_b
        else:
            AF = (1.0 - math.exp(- R_CONST * tau_b)) / R_CONST
        pvC = AF * c_b
        dw  = max(0.0, min(1.0, 1.0 - pvC / max(x_b, 1e-12)))

        u_b = u_base * dw
        # L1 내부 유지
        s = float(u_b.sum().item())
        if s >= L_cap:
            u_b *= (L_cap - eps_in) / (s + 1e-12)
        u_b = torch.clamp(u_b, min=eps_in)

        out[b, :d] = u_b
        out[b, d:d+1] = torch.tensor([c_b], device=device, dtype=dtype)

    return out
