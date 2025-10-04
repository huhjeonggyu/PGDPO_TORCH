# -*- coding: utf-8 -*-
# 파일: tests/mt_nd_retchet/closed_form_ref.py
# 목적:
#   (i) 정태 포트폴리오 u*, (ii) RIM 경계 기반 랫칫 정책(TimeDependentRatchetPolicyRIM),
#   (iii) BW 근사 정책(TimeDependentRatchetPolicyBW), (iv) 외부 빌더 build_cf_with_args(...)
#
# 참고:
#  - JEDC (2018) Lemma/Prop에서 z*(t) 경계의 적분 방정식(식 (14), (B.4)).
#  - RIM: s = T - t 변수에서 \tilde z(s)=z*(T-s) 를 k=1..N 순차로 풀기(사다리꼴 + 이분법).
#  - 다자산: θ_eff = (μ_eff)/σ_eff 사용(완전시장 상수계수 가정).

from __future__ import annotations
import math
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn

__all__ = [
    "solve_static_portfolio",
    "TimeDependentRatchetPolicyBW",
    "TimeDependentRatchetPolicyRIM",
    "build_cf_with_args",
]

# --------------------- 정태 포트폴리오 (Merton형, 비음수 + L1상한) --------------------- #
def solve_static_portfolio(alpha: torch.Tensor,
                           Sigma: torch.Tensor,
                           gamma: float,
                           L_cap: float) -> torch.Tensor:
    """
    u_unc = (1/gamma) * Sigma^{-1} * alpha
    -> 비음수로 투영 후, L1합이 L_cap을 넘으면 비례 축소.
    """
    u_unc = (1.0 / float(gamma)) * torch.linalg.solve(Sigma, alpha)
    u_pos = torch.clamp(u_unc, min=0.0)
    s = float(u_pos.sum().item())
    if s > 0.0 and s > L_cap:
        u_pos = u_pos * (L_cap / s)
    return u_pos

# ----------------------- 보조: 정규 CDF/ d± ----------------------- #
def _norm_cdf(x: float) -> float:
    # 수치적으로 안정적인 정규분포 CDF 근사
    # torch 필요 없는 경량 구현
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _d_pm(theta: float, rho: float, r: float, s: float, y_ratio: float) -> Tuple[float, float]:
    """
    d±(s, y) = [ log(y) + ((ρ - r) ± 0.5 θ^2) s ] / ( θ sqrt(s) )
    """
    s = max(s, 1e-12)
    th2 = theta * theta
    num_base = math.log(max(y_ratio, 1e-300))
    d_minus = (num_base + ((rho - r) - 0.5 * th2) * s) / (theta * math.sqrt(s))
    d_plus  = (num_base + ((rho - r) + 0.5 * th2) * s) / (theta * math.sqrt(s))
    return d_minus, d_plus

# ----------------------- RIM 경계해(z*(t)) ----------------------- #
def _rim_boundary(theta: float, rho: float, r: float, T: float, N: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    s-그리드(0..T)에서 \tilde z(s)=z*(T-s) 수치해.
    귀납식: 각 s_k에서 G(z)=0을 이분법으로 풀며 적분은 사다리꼴.
    G(z) = ∫_0^s e^{-ρ ξ} Φ(d_-(ξ, z/ẑ(s-ξ))) dξ - z ∫_0^s e^{-r ξ} Φ(d_+(ξ, z/ẑ(s-ξ))) dξ
    """
    # s-grid
    s_grid = torch.linspace(0.0, float(T), N+1)
    # 초기값: \tilde z(0)=1
    z_tilde = torch.ones(N+1)

    # 무한수명 경계 근사(상한 브래킷): z_inf = (r/ρ) * λ-/(λ- - 1)
    # 여기서 λ- 는 θ^2/2 λ^2 + ((ρ - r) - θ^2/2) λ - ρ = 0 의 음의 해
    A = 0.5 * (theta * theta)
    B = (rho - r) - 0.5 * (theta * theta)
    C = -rho
    disc = max(B * B - 4.0 * A * C, 0.0)
    lam_minus = (-B - math.sqrt(disc)) / (2.0 * A) if A > 0 else -1.0
    z_inf = (r / max(rho, 1e-12)) * (lam_minus / max(lam_minus - 1.0, 1e-6))
    z_upper_default = max(1.5, 5.0 * abs(z_inf))

    # 적분 보조: 사다리꼴 가중치
    def trapz_weights(k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # ξ_j = s_k - s_j, j=0..k
        s_k = s_grid[k].item()
        s_prev = s_grid[:k+1]
        xi = s_k - s_prev  # (k+1,)
        # 가중치 w: [1,2,2,...,2,1]*Δξ/2
        if k == 0:
            w = torch.tensor([s_k], dtype=s_prev.dtype)  # 단일구간
        else:
            dxi = (s_k - s_grid[k-1].item())
            # 비균등일 수 있어 엄밀한 사다리꼴: 개별 폭
            d = s_prev.clone()
            d[1:] = s_prev[1:] - s_prev[:-1]
            w = d.clone()
            w[1:-1] = (d[1:-1] + d[2:]) / 2.0
            w[0] = d[1] / 2.0 if k >= 1 else s_k
            w[-1] = (s_k - s_prev[-2]) / 2.0 if k >= 1 else s_k
        return xi, w

    # 이분법 보조: G(z)
    def G_of_z(k: int, z: float) -> float:
        xi, w = trapz_weights(k)   # (k+1,)
        # ratio: z / ẑ(s_k - ξ_j) = z / z_tilde[j]
        ratios = z / torch.clamp(z_tilde[:k+1], min=1e-12)
        acc1 = 0.0  # ∫ e^{-ρ ξ} Φ(d_-)
        acc2 = 0.0  # ∫ e^{-r ξ} Φ(d_+)
        for j in range(k+1):
            sxi = float(xi[j].item())
            yratio = float(ratios[j].item())
            dminus, dplus = _d_pm(theta, rho, r, sxi, yratio)
            acc1 += math.exp(-rho * sxi) * _norm_cdf(dminus) * float(w[j].item())
            acc2 += math.exp(-r   * sxi) * _norm_cdf(dplus ) * float(w[j].item())
        return acc1 - z * acc2

    # s=0: 이미 1
    for k in range(1, N+1):
        # 브래킷 선택: [z_lo, z_hi]
        z_lo = 1e-6
        z_hi = z_upper_default
        # 부호 검사 및 이분법
        g_lo = G_of_z(k, z_lo)
        g_hi = G_of_z(k, z_hi)
        # 필요시 상한 늘림
        tries = 0
        while g_lo * g_hi > 0.0 and tries < 5:
            z_hi *= 2.0
            g_hi = G_of_z(k, z_hi)
            tries += 1
        # 이분법
        zL, zH = z_lo, z_hi
        for _ in range(60):
            zm = 0.5 * (zL + zH)
            gm = G_of_z(k, zm)
            if gm == 0.0 or abs(zH - zL) <= 1e-8 * (1.0 + zm):
                z_tilde[k] = zm
                break
            if g_lo * gm <= 0.0:
                zH = zm; g_hi = gm
            else:
                zL = zm; g_lo = gm
        else:
            z_tilde[k] = 0.5 * (zL + zH)

    return s_grid, z_tilde

def _interp_z_of_tau(tau: torch.Tensor, s_grid: torch.Tensor, z_tilde: torch.Tensor) -> torch.Tensor:
    """z*(t)=\tilde z(τ) 선형보간"""
    # tau shape (B,1)
    s = tau.view(-1).clamp_min(0.0).clamp_max(float(s_grid[-1].item()))
    # torch.interp는 2.0부터, 여기선 수동 구현
    s_np = s.detach().cpu().numpy()
    sg = s_grid.detach().cpu().numpy()
    zg = z_tilde.detach().cpu().numpy()
    out = []
    import bisect
    for val in s_np:
        i = bisect.bisect_left(sg, val)
        if i <= 0: out.append(zg[0]); continue
        if i >= len(sg): out.append(zg[-1]); continue
        t0, t1 = sg[i-1], sg[i]
        z0, z1 = zg[i-1], zg[i]
        w = 0.0 if t1 == t0 else (val - t0)/(t1 - t0)
        out.append((1.0 - w) * z0 + w * z1)
    return torch.tensor(out, dtype=tau.dtype, device=tau.device).view_as(tau)

# ----------------------- BW 근사 정책 ----------------------- #
def _Rstar_tau(theta: float, rho: float, r: float, tau: float) -> float:
    if tau <= 0.0:
        return 1.0
    A = 0.5 * (theta ** 2)
    B = (rho - r) - 0.5 * (theta ** 2)
    C = - rho * (1.0 - math.exp(-rho * tau))
    disc = max(B * B - 4.0 * A * C, 0.0)
    if A <= 0.0:
        return 1.0
    n_minus = (-B - math.sqrt(disc)) / (2.0 * A)
    return 1.0 / (1.0 - n_minus)

def _annuity_factor(r: float, tau_t: torch.Tensor) -> torch.Tensor:
    if abs(r) < 1e-12:
        return tau_t.clamp_min(1e-12)
    return (1.0 - torch.exp(torch.tensor(-r, dtype=tau_t.dtype, device=tau_t.device) * tau_t)) / float(r)

def _X_threshold_BW(c: torch.Tensor, tau: torch.Tensor,
                    gamma: float, r: float, Rstar_tau: torch.Tensor) -> torch.Tensor:
    ratio = float(gamma) / torch.clamp(torch.tensor(gamma, dtype=c.dtype, device=c.device) - Rstar_tau, min=1e-6)
    return ratio * c * _annuity_factor(r, tau)

def _solve_c_from_X_BW(X: torch.Tensor, tau: torch.Tensor,
                       gamma: float, r: float, Rstar_tau: torch.Tensor) -> torch.Tensor:
    factor = torch.clamp(torch.tensor(gamma, dtype=X.dtype, device=X.device) - Rstar_tau, min=1e-6) / float(gamma)
    return X * factor / torch.clamp(_annuity_factor(r, tau), min=1e-10)

class TimeDependentRatchetPolicyBW(nn.Module):
    def __init__(self,
                 u_star: torch.Tensor,
                 theta_eff: float,
                 gamma: float,
                 r: float, rho: float, T: float,
                 device: torch.device,
                 alpha_raise: float = 1.0,
                 c_softcap: float | None = None,
                 dw_floor: float = 0.0):
        super().__init__()
        self.register_buffer("u_star", u_star.to(device))
        self.theta_eff = float(theta_eff)
        self.gamma = float(gamma)
        self.r = float(r)
        self.rho = float(rho)
        self.T = float(T)
        self.alpha_raise = float(alpha_raise)
        self.c_softcap = None if c_softcap is None else float(c_softcap)
        self.dw_floor = float(max(0.0, dw_floor))

    def forward(self, **states_dict):
        X = states_dict["X"]                 # (B, 2) : [Wealth, Habit]
        tau = states_dict.get("TmT", None)   # (B, 1) : time-to-maturity
        if tau is None:
            tau = torch.zeros_like(X[:, :1])
        wealth = X[:, :1]; habit  = X[:, 1:2]

        # 배치 R*_τ
        tau_np = tau.detach().cpu().numpy().reshape(-1)
        Rstar_vals = [_Rstar_tau(self.theta_eff, self.rho, self.r, float(max(0.0, t))) for t in tau_np]
        Rstar_tau = torch.tensor(Rstar_vals, dtype=wealth.dtype, device=wealth.device).view(-1, 1)

        # 임계부: X > X_ap(τ,c)면 상승
        X_th = _X_threshold_BW(c=habit, tau=tau, gamma=self.gamma, r=self.r, Rstar_tau=Rstar_tau)
        need_raise = (wealth > X_th)
        c_on_boundary = _solve_c_from_X_BW(X=wealth, tau=tau, gamma=self.gamma, r=self.r, Rstar_tau=Rstar_tau)
        c_raised = self.alpha_raise * c_on_boundary + (1.0 - self.alpha_raise) * habit
        C = torch.where(need_raise, torch.maximum(c_raised, habit), habit)

        # 소프트 캡(선택)
        if self.c_softcap is not None:
            cap = torch.maximum(habit, self.c_softcap * wealth)
            C = torch.minimum(C, cap)

        # 재량부 스케일
        pvC = _annuity_factor(self.r, tau) * C
        dw_share = (wealth - pvC) / wealth.clamp_min(1e-12)
        dw_share = torch.clamp(dw_share, min=self.dw_floor, max=1.0)

        u = self.u_star.unsqueeze(0).expand(wealth.size(0), -1) * dw_share
        return torch.cat([u, C], dim=1)

# ----------------------- RIM 정책 (y가 있으면 정확, 없으면 BW 폴백) ----------------------- #
class TimeDependentRatchetPolicyRIM(nn.Module):
    """
    - z*(t)을 RIM으로 구해두고, (가능하면) y/u'(c) ≤ z*(t) 를 써서 hit-and-raise.
      * CRRA: u'(c)=c^{-γ} ⇒ y c^γ ≤ z*(t) ⇒ C_boundary=(z/y)^{1/γ}.
    - states_dict에 y(=∂J/∂X, λ) 가 없으면 BW 임계부(X기반)로 폴백.
    - 포트폴리오는 u*×DW 비율(1-PV(C)/X).
    """
    def __init__(self,
                 u_star: torch.Tensor,
                 theta_eff: float,
                 gamma: float,
                 r: float, rho: float, T: float,
                 device: torch.device,
                 s_grid: torch.Tensor, z_tilde: torch.Tensor,
                 alpha_raise: float = 1.0,
                 c_softcap: float | None = None,
                 dw_floor: float = 0.0):
        super().__init__()
        self.register_buffer("u_star", u_star.to(device))
        self.theta_eff = float(theta_eff)
        self.gamma = float(gamma)
        self.r = float(r)
        self.rho = float(rho)
        self.T = float(T)
        self.alpha_raise = float(alpha_raise)
        self.c_softcap = None if c_softcap is None else float(c_softcap)
        self.dw_floor = float(max(0.0, dw_floor))
        # 경계표
        self.register_buffer("s_grid", s_grid.to(device))
        self.register_buffer("z_tilde", z_tilde.to(device))

    def _z_of_tau(self, tau: torch.Tensor) -> torch.Tensor:
        return _interp_z_of_tau(tau, self.s_grid, self.z_tilde)

    def forward(self, **states_dict):
        X = states_dict["X"]                 # (B,2) [X,Y]
        tau = states_dict.get("TmT", None)   # (B,1)
        if tau is None: tau = torch.zeros_like(X[:, :1])
        wealth = X[:, :1]; habit = X[:, 1:2]

        # y_t 소스 찾기(있으면 정확 RIM, 없으면 BW 폴백)
        y_shadow: Optional[torch.Tensor] = None
        if "y" in states_dict: y_shadow = states_dict["y"]
        elif "lambda" in states_dict: y_shadow = states_dict["lambda"]
        elif "lamx" in states_dict: y_shadow = states_dict["lamx"]
        elif "LamX" in states_dict: y_shadow = states_dict["LamX"]
        elif "JX" in states_dict and states_dict["JX"] is not None:
            jx = states_dict["JX"]
            y_shadow = jx[:, :1] if jx.ndim == 2 else None

        if y_shadow is not None:
            z_tau = self._z_of_tau(tau)                      # (B,1)
            y_eff = torch.clamp(y_shadow, min=1e-12)
            c_on_boundary = torch.pow(torch.clamp(z_tau / y_eff, min=1e-24), 1.0 / self.gamma)
            need_raise = (y_eff * torch.pow(habit, self.gamma) <= z_tau)
            c_raised = self.alpha_raise * c_on_boundary + (1.0 - self.alpha_raise) * habit
            C = torch.where(need_raise, torch.maximum(c_raised, habit), habit)
        else:
            # BW 폴백
            tau_np = tau.detach().cpu().numpy().reshape(-1)
            Rstar_vals = [_Rstar_tau(self.theta_eff, self.rho, self.r, float(max(0.0, t))) for t in tau_np]
            Rstar_tau = torch.tensor(Rstar_vals, dtype=wealth.dtype, device=wealth.device).view(-1, 1)
            X_th = _X_threshold_BW(c=habit, tau=tau, gamma=self.gamma, r=self.r, Rstar_tau=Rstar_tau)
            need_raise = (wealth > X_th)
            c_on_boundary = _solve_c_from_X_BW(X=wealth, tau=tau, gamma=self.gamma, r=self.r, Rstar_tau=Rstar_tau)
            c_raised = self.alpha_raise * c_on_boundary + (1.0 - self.alpha_raise) * habit
            C = torch.where(need_raise, torch.maximum(c_raised, habit), habit)

        # 소프트 캡(선택)
        if self.c_softcap is not None:
            cap = torch.maximum(habit, self.c_softcap * wealth)
            C = torch.minimum(C, cap)

        # 재량부 스케일
        pvC = _annuity_factor(self.r, tau) * C
        dw_share = (wealth - pvC) / wealth.clamp_min(1e-12)
        dw_share = torch.clamp(dw_share, min=self.dw_floor, max=1.0)

        u = self.u_star.unsqueeze(0).expand(wealth.size(0), -1) * dw_share
        return torch.cat([u, C], dim=1)

# ----------------------------- 외부 빌더 ----------------------------- #
def build_cf_with_args(*,
                       alpha: torch.Tensor, Sigma: torch.Tensor,
                       gamma: float, L_cap: float,
                       rho: float, r: float, T: float,
                       rim_steps: int,                  # RIM 그리드 수 (예: 256)
                       device: torch.device,
                       z_margin: float = 0.0,           # (미사용) 이전 z-기반과 호환
                       alpha_raise: float = 1.0,        # 부분 상승 계수
                       c_softcap: float | None = None,  # 소비 소프트캡 (배수)
                       dw_floor: float = 0.0,           # 재량부 스케일 하한
                       ref_method: str = "RIM"          # "RIM" | "BW"
                       ) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    외부 진입점:
      - u* 계산
      - θ_eff, μ_eff, σ_eff 산출
      - RIM 또는 BW 정책 객체와 부가 정보 반환
    """
    u_star = solve_static_portfolio(alpha, Sigma, gamma, L_cap)

    mu_eff   = float((alpha @ u_star).item())                  # (μ-r)_eff
    sig2_eff = float((u_star @ Sigma @ u_star).item())
    sigma_eff = math.sqrt(max(sig2_eff, 1e-12))
    theta_eff = mu_eff / max(sigma_eff, 1e-12)

    if ref_method.upper() == "RIM":
        s_grid, z_tilde = _rim_boundary(theta=theta_eff, rho=rho, r=r, T=T, N=max(8, int(rim_steps)))
        policy = TimeDependentRatchetPolicyRIM(
            u_star=u_star, theta_eff=theta_eff, gamma=gamma,
            r=r, rho=rho, T=T, device=device,
            s_grid=s_grid.to(device), z_tilde=z_tilde.to(device),
            alpha_raise=alpha_raise, c_softcap=c_softcap, dw_floor=dw_floor
        )
        info = {
            "note": "RIM ratchet boundary + DW scaling (uses y if provided; else BW fallback)",
            "mu_eff": mu_eff,
            "sig_eff": sigma_eff,
            "theta_eff": theta_eff,
            "grid_N": int(rim_steps),
            "z0": float(z_tilde[0].item()),
            "zT": float(z_tilde[-1].item()),
        }
    else:
        policy = TimeDependentRatchetPolicyBW(
            u_star=u_star, theta_eff=theta_eff, gamma=gamma,
            r=r, rho=rho, T=T, device=device,
            alpha_raise=alpha_raise, c_softcap=c_softcap, dw_floor=dw_floor
        )
        info = {
            "note": "BW approximation policy (time-dependent threshold) + DW scaling",
            "mu_eff": mu_eff,
            "sig_eff": sigma_eff,
            "theta_eff": theta_eff,
            "Rstar_tau0": _Rstar_tau(theta_eff, rho, r, 0.0),
            "Rstar_tauT": _Rstar_tau(theta_eff, rho, r, T),
        }
    return policy, info

