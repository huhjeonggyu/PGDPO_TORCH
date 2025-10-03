# 파일: tests/mt_nd_retchet/user_pgdpo_base.py
# 목적: 학습 베이스 + (코어 호환) 무인자 build_closed_form_policy() 제공

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- CF 모듈 임포트 (상대/절대 폴백) ---
try:
    from .closed_form_ref import build_cf_with_args
except Exception:
    from tests.mt_nd_retchet.closed_form_ref import build_cf_with_args

# --------------------------- 기본 설정 ---------------------------
d = int(os.getenv("PGDPO_D", 5)); k = 0
epochs = int(os.getenv("PGDPO_EPOCHS", 250)); batch_size = int(os.getenv("PGDPO_BS", 1024))
lr = float(os.getenv("PGDPO_LR", 1e-4)); seed = int(os.getenv("PGDPO_SEED", 42))
T = float(os.getenv("PGDPO_T", 1.5)); m = int(os.getenv("PGDPO_M", 40))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIM_X = 2; DIM_Y = k; DIM_U = d + 1
N_eval_states = int(os.getenv("PGDPO_EVAL_STATES", 100))
CRN_SEED_EU = int(os.getenv("PGDPO_CRN", 12345))
USE_CLOSED_FORM = bool(int(os.getenv("PGDPO_CF", "1")))
RIM_STEPS = int(os.getenv("PGDPO_RIM_STEPS", str(max(200, m))))

# --------------------------- 경제 파라미터 ---------------------------
gamma = float(os.getenv("PGDPO_GAMMA", 3.0))
rho   = float(os.getenv("PGDPO_RHO",   0.04))
r     = float(os.getenv("PGDPO_RF",    0.01))
kappa = float(os.getenv("PGDPO_KAPPA", 1.0))
X0_range = (0.5, 2.0)
H0_RATIO = float(os.getenv("PGDPO_H0_RATIO", 0.6))
lb_X = 1e-6
L_cap = float(os.getenv("PGDPO_LCAP", 1.0))

# --------------------------- 시장 파라미터 ---------------------------
@torch.no_grad()
def _nearest_spd_correlation(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    C = 0.5 * (C + C.T); evals, evecs = torch.linalg.eigh(C)
    evals = torch.clamp(evals, min=eps); C_spd = (evecs @ torch.diag(evals) @ evecs.T)
    d_ = torch.diag(C_spd).clamp_min(eps).sqrt(); Dinv = torch.diag(1.0 / d_)
    return 0.5 * (Dinv @ C_spd @ Dinv + (Dinv @ C_spd @ Dinv).T)

@torch.no_grad()
def _generate_mu_sigma_balanced(
    d: int, dev, seed: int | None = None, *,
    vol_range=(0.16, 0.26), avg_corr=0.35, jitter=0.035, lam_factor=0.01,
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

params = _generate_mu_sigma_balanced(
    d, device, seed=seed,
    target_leverage=0.7 * L_cap,
    dirichlet_conc=10.0,
    hhi_factor=3.0
)
alpha, Sigma, Sigma_inv = params["alpha"], params["Sigma"], params["Sigma_inv"]
chol_S = torch.linalg.cholesky(Sigma)

# --------------------------- 학습 정책 ---------------------------
class DirectPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        state_dim = 3; hid = 256
        self.net = nn.Sequential(
            nn.Linear(state_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, d + 2)
        )
    def forward(self, **states_dict):
        wealth, habit, TmT = states_dict['X'][:, 0:1], states_dict['X'][:, 1:2], states_dict['TmT']
        s_in = torch.cat([wealth, habit / wealth.clamp_min(1e-8), TmT], dim=1)
        raw = self.net(s_in)
        u = L_cap * F.softmax(raw[:, :d+1], dim=1)[:, :d]
        proposed = habit + F.softplus(raw[:, d+1:])
        C = torch.max(proposed, habit)
        return torch.cat([u, C], dim=1)

# --------------------------- 초기 상태/시뮬레이터 ---------------------------
def sample_initial_states(B, *, rng=None):
    wealth0 = torch.rand((B, 1), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
    habit0  = (wealth0 * H0_RATIO).clamp_min(1e-6)
    X0 = torch.cat([wealth0, habit0], dim=1)
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
        out = policy(**{'X': torch.cat([wealth, habit], dim=1), 'TmT': states['TmT'] - i * dt})
        u, C = out[:, :d], out[:, d:]
        drift = r + (u * alpha).sum(1, keepdim=True) - 0.5 * (u.unsqueeze(1) @ Sigma @ u.unsqueeze(-1)).squeeze(-1) - C / wealth
        dBX = (u @ chol_S * Z[:, i, :]).sum(1, keepdim=True)
        wealth = torch.exp(torch.log(wealth) + drift * dt + dBX * dt.sqrt()).clamp_min(lb_X)
        habit = torch.max(habit, C)
        period_utility = (C.clamp_min(1e-12).pow(1.0 - gamma) - 1.0) / (1.0 - gamma) if gamma != 1.0 else torch.log(C.clamp_min(1e-12))
        total_utility += torch.exp(-rho * t_i) * period_utility * dt
    terminal_utility = (wealth.pow(1.0 - gamma) - 1.0) / (1.0 - gamma) if gamma != 1.0 else torch.log(wealth)
    total_utility += torch.exp(torch.tensor(-rho * T, device=device)) * kappa * terminal_utility
    return total_utility.view(-1)

# --------------------------- (코어 호환) 무인자 CF 빌더 ---------------------------
Z_MARGIN    = float(os.getenv("PGDPO_Z_MARGIN", "0.15"))   # 경계 15% 상향
ALPHA_RAISE = float(os.getenv("PGDPO_ALPHA_RAISE", "0.7")) # 70%만 올리고 나머지는 Y 유지
C_SOFTCAP   = os.getenv("PGDPO_C_SOFTCAP", "None")
C_SOFTCAP   = None if C_SOFTCAP=="None" else float(C_SOFTCAP)

def build_closed_form_policy():
    return build_cf_with_args(
        alpha=alpha, Sigma=Sigma, gamma=gamma, L_cap=L_cap,
        rho=rho, r=r, T=T, rim_steps=RIM_STEPS, device=device,
        z_margin=Z_MARGIN, alpha_raise=ALPHA_RAISE, c_softcap=C_SOFTCAP
    )

# (선택) 로딩 확인용 프리로드
CF_POLICY, CF_INFO = (None, {})
if USE_CLOSED_FORM:
    try:
        CF_POLICY, CF_INFO = build_closed_form_policy()
        print(f"[CF] loaded? {CF_POLICY is not None} | note={CF_INFO['note']}, "
              f"mu={CF_INFO['mu_eff']:.4f}, sig={CF_INFO['sig_eff']:.4f}, "
              f"z_min={CF_INFO.get('z_min','?')}, z_max={CF_INFO.get('z_max','?')}")
    except Exception as e:
        print(f"[WARN] CF preload failed: {e}")

# --------------------------- 시각화 스키마 ---------------------------
def get_traj_schema():
    u_labels = [f"u_{i+1}" for i in range(d)] + ["Consumption"]
    x_labels = ["Wealth (X)", "Ratchet Y"]
    return {
        "roles": {"X": {"dim": DIM_X, "labels": x_labels}, "U": {"dim": DIM_U, "labels": u_labels}},
        "views": [
            {"name": "Consumption_Path", "block": "U", "mode": "indices", "indices": [d], "ylabel": "Consumption (C)"},
            {"name": "State_Variables", "block": "X", "mode": "indices", "indices": [0, 1], "ylabel": "State Value"},
        ],
        "sampling": {"Bmax": 5}
    }