# -*- coding: utf-8 -*-
# 파일: tests/mt_nd_retchet/closed_form_ref.py
# 목적:
#   (i) 정태 포트폴리오 u*, (ii) BW 근사 기반 시간의존 임계부 X_ap(t,c),
#   (iii) BW 정책(TimeDependentRatchetPolicyBW), (iv) 외부 빌더 build_cf_with_args(...)
#
# 참고: Jeon–Koo–Shin (2018) JEDC, Proposition 4.1.
#  - R*_τ = 1/(1 - n_-(τ)),  n_-(τ)는 θ^2/2·n^2 + ((ρ-r) - 0.5θ^2)·n - ρ(1 - e^{-ρτ}) = 0 의 음의 해
#  - X_ap(τ,c) = [γ/(γ - R*_τ)] · c · (1 - e^{-rτ}) / r   (r≈0이면 분모를 τ로 대체)
#  - X > X_ap(τ,c) 일 때만 c를 경계로 상향(랫칭: C≥Y).
#  - 포트폴리오 위험노출은 재량부 비율 (1 - PV(c)/X) 로 스케일.

from __future__ import annotations
import math
import torch
import torch.nn as nn

__all__ = [
    "solve_static_portfolio",
    "TimeDependentRatchetPolicyBW",
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

# ----------------------- BW 근사 보조 함수 ----------------------- #
def _Rstar_tau(theta: float, rho: float, r: float, tau: float) -> float:
    """
    Proposition 4.1의 R*_τ.
    θ = (μ-r)/σ (여기서는 유효 포트폴리오 θ_eff 사용)
    n_-은 (θ^2/2) n^2 + ((ρ-r) - 0.5 θ^2) n - ρ (1 - e^{-ρ τ}) = 0 의 음의 해.
    """
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
    """ (1 - e^{-r τ})/r (r≈0이면 τ) """
    if abs(r) < 1e-12:
        return tau_t.clamp_min(1e-12)
    return (1.0 - torch.exp(torch.tensor(-r, dtype=tau_t.dtype, device=tau_t.device) * tau_t)) / float(r)

def _X_threshold_BW(c: torch.Tensor, tau: torch.Tensor,
                    gamma: float, r: float, Rstar_tau: torch.Tensor) -> torch.Tensor:
    """
    X_ap(τ,c) = [γ/(γ - R*_τ)] * c * (1 - e^{-rτ}) / r
    """
    ratio = float(gamma) / torch.clamp(torch.tensor(gamma, dtype=c.dtype, device=c.device) - Rstar_tau, min=1e-6)
    return ratio * c * _annuity_factor(r, tau)

def _solve_c_from_X_BW(X: torch.Tensor, tau: torch.Tensor,
                       gamma: float, r: float, Rstar_tau: torch.Tensor) -> torch.Tensor:
    """
    경계에서 평형 X = X_ap(τ,c)을 만족하도록 c를 역산:
    c = X * (γ - R*_τ)/γ * r/(1-e^{-rτ})   (r≈0이면 분모를 τ로 대체)
    """
    factor = torch.clamp(torch.tensor(gamma, dtype=X.dtype, device=X.device) - Rstar_tau, min=1e-6) / float(gamma)
    return X * factor / torch.clamp(_annuity_factor(r, tau), min=1e-10)

# ----------------------- 정책 클래스 (BW 근사 기반) ----------------------- #
class TimeDependentRatchetPolicyBW(nn.Module):
    """
    - 포트폴리오는 u*를 기본으로 하되, 재량부 비율(1 - PV(c)/X)로 위험노출을 스케일.
    - 소비는 BW 근사 임계부 사용:
        X > X_ap(τ,c_current) 이면 c를 경계로 올림(부분 상승 허용), 아니면 유지.
    - 항상 C >= Y(=과거최대소비) 강제 -> 랫칭 보장.
    - 선택 소프트캡: C <= cap_mult * X (단, cap >= Y 유지).
    """
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

        wealth = X[:, :1]
        habit  = X[:, 1:2]

        # 배치 R*_τ
        tau_np = tau.detach().cpu().numpy().reshape(-1)
        Rstar_vals = [_Rstar_tau(self.theta_eff, self.rho, self.r, float(max(0.0, t))) for t in tau_np]
        Rstar_tau = torch.tensor(Rstar_vals, dtype=wealth.dtype, device=wealth.device).view(-1, 1)

        # 임계부 통과 여부
        X_th = _X_threshold_BW(c=habit, tau=tau, gamma=self.gamma, r=self.r, Rstar_tau=Rstar_tau)
        need_raise = (wealth > X_th)

        # 경계로 '부분 상승' (alpha_raise=1 → 완전 상승)
        c_on_boundary = _solve_c_from_X_BW(X=wealth, tau=tau, gamma=self.gamma, r=self.r, Rstar_tau=Rstar_tau)
        c_raised = self.alpha_raise * c_on_boundary + (1.0 - self.alpha_raise) * habit
        C = torch.where(need_raise, torch.maximum(c_raised, habit), habit)

        # (선택) 소비 소프트 캡: cap ≥ habit 보장
        if self.c_softcap is not None:
            cap = torch.maximum(habit, self.c_softcap * wealth)
            C = torch.minimum(C, cap)

        # 재량부 비율로 위험노출 스케일: λ_dw = max(0, 1 - PV(C)/X)
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
                       rim_steps: int,  # 인터페이스 유지용(미사용)
                       device: torch.device,
                       z_margin: float = 0.0,           # (미사용) 이전 z-기반과 호환
                       alpha_raise: float = 1.0,        # 부분 상승 계수
                       c_softcap: float | None = None,  # 소비 소프트캡 (배수)
                       dw_floor: float = 0.0            # 재량부 스케일의 하한(옵션)
                       ):
    """
    외부 진입점:
      - u* 계산
      - θ_eff, μ_eff, σ_eff 산출
      - BW 정책 객체와 부가 정보 반환
    """
    u_star = solve_static_portfolio(alpha, Sigma, gamma, L_cap)

    mu_eff   = float((alpha @ u_star).item())                  # (μ-r)_eff
    sig2_eff = float((u_star @ Sigma @ u_star).item())
    sigma_eff = math.sqrt(max(sig2_eff, 1e-12))
    theta_eff = mu_eff / max(sigma_eff, 1e-12)

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

