# 파일: tests/mt_nd_ratchet/user_pgdpo_base.py
# 모델: 소비 랫칭 모델 (균형 잡힌 시장 생성기를 포함한 최종 완성본)

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# --------------------------- 기본 설정 ---------------------------
d = int(os.getenv("PGDPO_D", 5)); k = 0
epochs = int(os.getenv("PGDPO_EPOCHS", 250)); batch_size = int(os.getenv("PGDPO_BS", 1024))
lr = float(os.getenv("PGDPO_LR", 1e-4)); seed = int(os.getenv("PGDPO_SEED", 42))
T = float(os.getenv("PGDPO_T", 1.5)); m = int(os.getenv("PGDPO_M", 40))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIM_X = 2; DIM_Y = k; DIM_U = d + 1
N_eval_states = int(os.getenv("PGDPO_EVAL_STATES", 100)); CRN_SEED_EU   = int(os.getenv("PGDPO_CRN", 12345))

# --------------------------- 경제 파라미터 ---------------------------
gamma = float(os.getenv("PGDPO_GAMMA", 2.0)); rho = float(os.getenv("PGDPO_RHO", 0.10))
r = float(os.getenv("PGDPO_RF", 0.03)); kappa = float(os.getenv("PGDPO_KAPPA", 1.0))
X0_range = (0.5, 2.0); H0_initial = 0.01; lb_X = 1e-6
L_cap = float(os.getenv("PGDPO_LCAP", 1.0))

# --------------------------- 시장 파라미터 생성 (개선됨) ---------------------------
@torch.no_grad()
def _nearest_spd_correlation(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    C = 0.5 * (C + C.T); evals, evecs = torch.linalg.eigh(C)
    evals = torch.clamp(evals, min=eps); C_spd = (evecs @ torch.diag(evals) @ evecs.T)
    d_ = torch.diag(C_spd).clamp_min(eps).sqrt(); Dinv = torch.diag(1.0 / d_)
    return 0.5 * (Dinv @ C_spd @ Dinv + (Dinv @ C_spd @ Dinv).T)

@torch.no_grad()
def _generate_mu_sigma_balanced(
    d: int, dev, seed: int | None = None, *,
    vol_range=(0.18, 0.28), avg_corr=0.45, jitter=0.04, lam_factor=0.01,
    dirichlet_conc=10.0, target_leverage=None, alpha_mid=0.09,
    noise_std=0.01, hhi_factor=3.0
):
    if seed is not None: torch.manual_seed(seed)
    sigma = torch.empty(d, device=dev).uniform_(*vol_range)
    Psi = torch.full((d, d), float(avg_corr), device=dev); Psi.fill_diagonal_(1.0)
    if jitter > 0:
        N = torch.randn(d, d, device=dev) * jitter
        Psi = _nearest_spd_correlation(Psi + 0.5 * (N + N.T))
    Sigma = torch.diag(sigma) @ Psi @ torch.diag(sigma)
    Sigma = Sigma + lam_factor * Sigma.diag().mean() * torch.eye(d, device=dev)

    w_ref = torch.distributions.Dirichlet(torch.full((d,), float(dirichlet_conc))).sample().to(dev)
    hhi_target = hhi_factor / d
    if (w_ref**2).sum() > hhi_target:
        mix = 0.4; w_ref = (1 - mix) * w_ref + mix * (torch.ones(d, device=dev) / d)

    if target_leverage is not None: s = float(target_leverage)
    else: s = float(alpha_mid / (gamma * (Sigma @ w_ref).mean().clamp_min(1e-8)).item())
    alpha = gamma * s * (Sigma @ w_ref)
    if noise_std > 0:
        alpha += float(noise_std) * alpha.abs().mean() * torch.randn_like(alpha)
        alpha = alpha.clamp_min(1e-4)

    Sigma_inv = torch.linalg.inv(Sigma)
    return {"alpha": alpha, "Sigma": Sigma, "Sigma_inv": Sigma_inv}

# (핵심) 균형 잡힌 시장 생성 함수 호출
params = _generate_mu_sigma_balanced(
    d, device, seed=seed,
    target_leverage=0.7 * L_cap,
    dirichlet_conc=10.0,
    hhi_factor=3.0
)
alpha, Sigma, Sigma_inv = params["alpha"], params["Sigma"], params["Sigma_inv"]
chol_S = torch.linalg.cholesky(Sigma)

# --------------------------- 정책 신경망 ---------------------------
class DirectPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        state_dim = 3; hid = 256
        self.net = nn.Sequential(nn.Linear(state_dim, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, d + 2))
    def forward(self, **states_dict):
        wealth, habit, TmT = states_dict['X'][:, 0:1], states_dict['X'][:, 1:2], states_dict['TmT']
        state_input = torch.cat([wealth, habit / wealth.clamp_min(1e-8), TmT], dim=1)
        raw_output = self.net(state_input)
        u = L_cap * F.softmax(raw_output[:, :d+1], dim=1)[:, :d]
        C = wealth - F.softplus(wealth - (habit + F.softplus(raw_output[:, d+1:])), beta=5.0)
        return torch.cat([u, C], dim=1)

# --------------------------- 초기 상태 및 시뮬레이터 ---------------------------
def sample_initial_states(B, *, rng=None):
    wealth0 = torch.rand((B, 1), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
    X0 = torch.cat([wealth0, torch.full_like(wealth0, H0_initial)], dim=1)
    TmT0 = torch.rand((B, 1), device=device, generator=rng) * T
    return {'X': X0, 'Y': None, 'TmT': TmT0}, TmT0 / float(m)

def simulate(policy, B, *, train=True, rng=None, initial_states_dict=None, random_draws=None, m_steps=None):
    m_eff = m_steps or m
    states, dt = sample_initial_states(B, rng=rng) if initial_states_dict is None else (initial_states_dict, initial_states_dict['TmT'] / m_eff)
    wealth, habit = states['X'][:, 0:1].clamp_min(lb_X), states['X'][:, 1:2]
    Z = random_draws[0] if random_draws is not None else torch.randn(B, m_eff, d, device=device, generator=rng)
    total_utility = torch.zeros((B, 1), device=device)
    for i in range(m_eff):
        t_i = T - (states['TmT'] - i * dt)
        policy_output = policy(**{'X': torch.cat([wealth, habit], dim=1), 'TmT': states['TmT'] - i * dt})
        u, C = policy_output[:, :d], policy_output[:, d:]
        drift = r + (u * alpha).sum(1, keepdim=True) - 0.5 * (u.unsqueeze(1) @ Sigma @ u.unsqueeze(-1)).squeeze(-1) - C / wealth
        dBX = (u @ chol_S * Z[:, i, :]).sum(1, keepdim=True)
        wealth = torch.exp(torch.log(wealth) + drift * dt + dBX * dt.sqrt()).clamp_min(lb_X)
        habit = torch.max(habit, C)
        period_utility = (C.clamp_min(1e-12).pow(1.0 - gamma) - 1.0) / (1.0 - gamma) if gamma != 1.0 else torch.log(C.clamp_min(1e-12))
        total_utility += torch.exp(-rho * t_i) * period_utility * dt
    terminal_utility = (wealth.pow(1.0 - gamma) - 1.0) / (1.0 - gamma) if gamma != 1.0 else torch.log(wealth)
    total_utility += torch.exp(torch.tensor(-rho * T, device=device)) * kappa * terminal_utility
    return total_utility.view(-1)

# --------------------------- 준-분석적 해 계산 로직 ---------------------------
def _solve_static_portfolio(alpha, Sigma, gamma, L_cap):
    u_unc = (1.0 / gamma) * torch.linalg.solve(Sigma, alpha)
    u_pos = u_unc.clamp_min(0.0); s = float(u_pos.sum().item())
    return u_pos * (L_cap / s) if s > L_cap and s > 0 else u_pos
def _solve_ratchet_free_boundary(params_np, z_low=1.01, z_high=200.0, z_min=1e-5):
    g, rh, r, mu, s2 = [float(params_np[k]) for k in ['gamma', 'rho', 'r', 'mu_eff', 'sig2_eff']]; s2 = max(s2, 1e-12)
    def ode(z, y): return np.array([y[1], (rh*y[0] - 1/(1-g) - ((r+mu)*z-1)*y[1])/(0.5*s2*z*z)])
    def ic(zs): A1 = rh*zs - (1-g)*((r+mu)*zs-1-0.5*s2*g*zs); vp=(1-rh)/(1e-12 if abs(A1)<1e-12 else A1); return (1+zs*vp)/(1-g), vp
    def shoot(zs): v0, vp0 = ic(zs); sol=solve_ivp(ode, [zs, z_min], [v0, vp0], method='LSODA'); return sol.y[1,-1]*sol.t[-1] if sol.success else np.nan
    grid = np.geomspace(z_low, z_high, 60); vals = np.array([shoot(zs) for zs in grid])
    fin = np.where(np.isfinite(vals))[0]
    if not fin.size: return grid[0]
    sgn, idx = np.sign(vals[fin]), np.where(np.sign(vals[fin][:-1]) * np.sign(vals[fin][1:]) < 0)[0]
    if idx.size: return float(brentq(shoot, grid[fin[idx[0]]], grid[fin[idx[0]+1]]))
    return float(grid[fin[np.nanargmin(np.abs(vals[fin]))]])
class ClosedFormPolicy(nn.Module):
    def __init__(self, u_star, z_star): super().__init__(); self.register_buffer("u_star", u_star); self.z_star = z_star
    def forward(self, **states_dict):
        wealth, habit = states_dict['X'][:, 0:1], states_dict['X'][:, 1:2]
        u = self.u_star.unsqueeze(0).expand(wealth.size(0), -1); z = wealth/habit.clamp_min(1e-8)
        C = torch.min(torch.where(z > self.z_star, wealth / self.z_star, habit), wealth)
        return torch.cat([u, C], dim=1)

def build_closed_form_policy():
    if not _HAS_SCIPY: return None, {"note": "scipy not found"}
    try:
        print("✅ Computing semi-analytical benchmark for ratchet model...")
        u_star = _solve_static_portfolio(alpha, Sigma, gamma, L_cap)
        params_np = {'gamma':gamma,'rho':rho,'r':r,'mu_eff':(alpha@u_star).item(),'sig2_eff':(u_star@Sigma@u_star).item()}
        z_star = _solve_ratchet_free_boundary(params_np)
        print(f"   ... Found critical ratio z* = {z_star:.4f}")
        return ClosedFormPolicy(u_star.to(device), z_star), {"note": f"Semi-analytical solution with z*={z_star:.4f}"}
    except Exception as e:
        print(f"[WARN] Failed to compute semi-analytical solution: {e}"); return None, {"note": "failed to compute"}

# --------------------------- 시각화 스키마 ---------------------------
def get_traj_schema():
    u_labels = [f"u_{i+1}" for i in range(d)] + ["Consumption"]
    x_labels = ["Wealth", "Habit"]
    return {
        "roles": {"X": {"dim": DIM_X, "labels": x_labels}, "U": {"dim": DIM_U, "labels": u_labels}},
        "views": [
            {"name": "Consumption_Path", "block": "U", "mode": "indices", "indices": [d], "ylabel": "Consumption (C)"},
            {"name": "State_Variables", "block": "X", "mode": "indices", "indices": [0, 1], "ylabel": "State Value"},
        ],
        "sampling": {"Bmax": 5}
    }