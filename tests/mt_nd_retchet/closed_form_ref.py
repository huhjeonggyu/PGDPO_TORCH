# 파일: tests/mt_nd_retchet/closed_form_ref.py
# 목적: (i) 정태 포트폴리오 u*, (ii) 시간의존 자유경계(RIM), (iii) 준-분석 정책 클래스
# 외부에서는 build_cf_with_args(...)로만 사용하세요. (코어 호출용 무인자 래퍼는 base 파일이 제공)

import math
import numpy as np
import torch
import torch.nn as nn

try:
    from scipy.optimize import brentq
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

SQRT2 = math.sqrt(2.0)
def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))

def solve_static_portfolio(alpha: torch.Tensor, Sigma: torch.Tensor, gamma: float, L_cap: float) -> torch.Tensor:
    """
    정태 Merton형: u* = (1/gamma) Sigma^{-1} alpha, 비음수 및 총합 상한 L_cap 간단 투영.
    """
    u_unc = (1.0 / gamma) * torch.linalg.solve(Sigma, alpha)
    u_pos = u_unc.clamp_min(0.0)
    s = float(u_pos.sum().item())
    return u_pos * (L_cap / s) if s > L_cap and s > 0 else u_pos

def _bisect_root(f, a, b, tol=1e-8, max_it=200):
    fa, fb = f(a), f(b)
    if math.isnan(fa) or math.isnan(fb): return None
    if fa == 0.0: return a
    if fb == 0.0: return b
    if fa * fb > 0.0: return None
    lo, hi, flo, fhi = a, b, fa, fb
    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) * 0.5 < tol:
            return mid
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)

def solve_free_boundary_RIM(mu_eff: float, sig2_eff: float, rho: float, r: float, T: float, N: int):
    """
    Jeon–Koo–Shin(2018) 식(14) 자유경계 적분방정식을 RIM(Recursive Iteration Method)으로 풉니다.
      - s = T - t, z_tilde(0) = 1, 단조 증가 가정.
      - d±(xi,y) = [log y + ((rho-r) ± 0.5 theta^2) xi] / (theta sqrt(xi)),  theta = mu_eff / sigma_eff.
    반환: (z_table, s_grid)
    """
    eps = 1e-12
    sig2_eff = max(sig2_eff, eps)
    sigma_eff = math.sqrt(sig2_eff)
    theta = (mu_eff / sigma_eff) if sigma_eff > 0.0 else 1e-8

    N = max(2, int(N))
    s_grid = np.linspace(0.0, T, N + 1)
    ds = s_grid[1] - s_grid[0]
    z = np.ones(N + 1, dtype=np.float64)  # z[0]=1

    def d_minus(xi, y):
        if xi <= 0.0: return 1e6
        return (math.log(max(y, eps)) + ((rho - r) - 0.5 * theta * theta) * xi) / (theta * math.sqrt(xi))
    def d_plus(xi, y):
        if xi <= 0.0: return -1e6
        return (math.log(max(y, eps)) + ((rho - r) + 0.5 * theta * theta) * xi) / (theta * math.sqrt(xi))

    for j in range(1, N + 1):
        xi_vals = s_grid[1:j+1]

        def F(zj: float) -> float:
            A = 0.0; B = 0.0
            for k in range(1, j + 1):
                xi = xi_vals[k - 1]
                y_ratio = zj / z[j - k]
                w = ds  # 간단 trapezoid
                A += w * math.exp(-rho * xi) * _phi(d_minus(xi, y_ratio))
                B += w * math.exp(-r   * xi) * _phi(d_plus (xi, y_ratio))
            return A - zj * B

        lo = max(z[j - 1], 1.0 + 1e-8)
        hi = max(lo * 2.0, lo + 1e-6)
        f_lo = F(lo); f_hi = F(hi)
        tries = 0
        while (math.isnan(f_lo) or math.isnan(f_hi) or f_lo * f_hi > 0.0) and tries < 40:
            hi *= 2.0; f_hi = F(hi); tries += 1

        if _HAS_SCIPY and (not math.isnan(f_lo)) and (not math.isnan(f_hi)) and (f_lo * f_hi <= 0.0):
            try:
                zj = brentq(F, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=200)
            except Exception:
                zj = _bisect_root(F, lo, hi, tol=1e-8, max_it=200)
        else:
            zj = _bisect_root(F, lo, hi, tol=1e-8, max_it=200)

        if zj is None or not np.isfinite(zj):
            candidates = np.logspace(math.log10(lo), math.log10(max(hi, lo * 10.0)), 40)
            vals = np.array([abs(F(c)) for c in candidates])
            zj = float(candidates[int(np.nanargmin(vals))])

        z[j] = max(zj, lo)
    return z, s_grid

# closed_form_ref.py
class TimeDependentRatchetPolicy(nn.Module):
    def __init__(self, u_star, z_table, s_grid, device, z_margin=0.0, alpha_raise=1.0, c_softcap=None):
        super().__init__()
        self.register_buffer("u_star", u_star.to(device))
        self.T = float(s_grid[-1]); self.N = int(len(s_grid) - 1); self.ds = self.T / self.N
        self.register_buffer("z_table", torch.tensor(z_table, dtype=torch.float32, device=device))
        self.z_margin = float(z_margin)         # <-- 추가
        self.alpha_raise = float(alpha_raise)   # <-- 추가 (부분 상승)
        self.c_softcap = c_softcap if c_softcap is None else float(c_softcap)  # <-- 선택적 소프트 캡

    def _interp_z_tau(self, tau):
        tau = tau.clamp(0.0, self.T - 1e-12)
        idx_f = (tau / self.ds); idx0 = torch.floor(idx_f).long().clamp(0, self.N-1); idx1 = (idx0+1).clamp(0, self.N)
        w1 = (idx_f - idx0.float()).clamp(0.0, 1.0)
        return (1.0-w1)*self.z_table[idx0] + w1*self.z_table[idx1]

    def forward(self, **states_dict):
        X = states_dict['X']; wealth, habit = X[:, :1], X[:, 1:2]
        tau = states_dict.get('TmT', torch.zeros_like(wealth))
        z_tau = self._interp_z_tau(tau) * (1.0 + self.z_margin)    # <-- 경계 보수화
        ratio = wealth / habit.clamp_min(1e-8)

        # 목표치로 '부분 상승' (alpha_raise=1이면 완전 상승, <1이면 완만)
        c_target = wealth / z_tau.clamp_min(1.0 + 1e-8)
        c_raised = self.alpha_raise * c_target + (1.0 - self.alpha_raise) * habit
        C_raise  = torch.max(c_raised, habit)                      # 항상 C>=Y
        C_keep   = habit

        C = torch.where(ratio > z_tau, C_raise, C_keep)

        # 선택: 과도한 폭주만 제한 (예: C <= 2 * X). None이면 적용 안 함.
        if self.c_softcap is not None:
            C = torch.min(C, self.c_softcap * wealth)

        u = self.u_star.unsqueeze(0).expand(wealth.size(0), -1)
        return torch.cat([u, C], dim=1)


# closed_form_ref.py
def build_cf_with_args(*, alpha, Sigma, gamma, L_cap, rho, r, T, rim_steps, device,
                       z_margin=0.0, alpha_raise=1.0, c_softcap=None):
    u_star = solve_static_portfolio(alpha, Sigma, gamma, L_cap)
    mu_eff = float((alpha @ u_star).item()); sig2_eff = float((u_star @ Sigma @ u_star).item())
    z_table, s_grid = solve_free_boundary_RIM(mu_eff, sig2_eff, rho, r, T, rim_steps)
    policy = TimeDependentRatchetPolicy(u_star, z_table, s_grid, device,
                                        z_margin=z_margin, alpha_raise=alpha_raise, c_softcap=c_softcap)
    info = {"note": f"RIM N={len(s_grid)-1}", "mu_eff": mu_eff, "sig_eff": math.sqrt(max(sig2_eff,1e-12)),
            "z_min": float(np.min(z_table)), "z_max": float(np.max(z_table))}
    return policy, info

