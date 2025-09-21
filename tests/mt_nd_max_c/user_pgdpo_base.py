# 파일: tests/mt_nd_max_c/user_pgdpo_base.py
# 모델: 소비 상한 모델 (균형 잡힌 시장 생성기를 포함한 최종 완성본)

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- 기본 설정 ---------------------------
d = int(os.getenv("PGDPO_D", 5)); k = 0
epochs = int(os.getenv("PGDPO_EPOCHS", 200)); batch_size = int(os.getenv("PGDPO_BS", 1024))
lr = float(os.getenv("PGDPO_LR", 1e-4)); seed = int(os.getenv("PGDPO_SEED", 42))
T = float(os.getenv("PGDPO_T", 1.5)); m = int(os.getenv("PGDPO_M", 20))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIM_X = 1; DIM_Y = k; DIM_U = d + 1
N_eval_states = int(os.getenv("PGDPO_EVAL_STATES", 100)); CRN_SEED_EU = int(os.getenv("PGDPO_CRN", 12345))

# --------------------------- 경제 파라미터 ---------------------------
gamma = float(os.getenv("PGDPO_GAMMA", 2.0)); rho   = float(os.getenv("PGDPO_RHO", 0.10))
r     = float(os.getenv("PGDPO_RF", 0.03)); kappa = float(os.getenv("PGDPO_KAPPA", 1.0))
alpha_rel = float(os.getenv("PGDPO_C_ALPHA", 0.30)); L_cap = float(os.getenv("PGDPO_LCAP", 1.0))
X0_range = (0.1, 3.0); lb_X = 1e-6

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
        N = torch.randn(d, d, device=dev) * jitter; Psi = _nearest_spd_correlation(Psi + 0.5 * (N + N.T))
    Sigma = torch.diag(sigma) @ Psi @ torch.diag(sigma)
    Sigma = Sigma + lam_factor * Sigma.diag().mean() * torch.eye(d, device=dev)
    w_ref = torch.distributions.Dirichlet(torch.full((d,), float(dirichlet_conc))).sample().to(dev)
    if (w_ref**2).sum() > (hhi_factor / d):
        w_ref = (0.6 * w_ref) + 0.4 * (torch.ones(d, device=dev) / d)
    if target_leverage is not None: s = float(target_leverage)
    else: s = float(alpha_mid / (gamma * (Sigma @ w_ref).mean().clamp_min(1e-8)).item())
    alpha = gamma * s * (Sigma @ w_ref)
    if noise_std > 0:
        alpha += float(noise_std) * alpha.abs().mean() * torch.randn_like(alpha)
        alpha = alpha.clamp_min(1e-4)
    Sigma_inv = torch.linalg.inv(Sigma)
    return {"alpha": alpha, "Sigma": Sigma, "Sigma_inv": Sigma_inv}

# --- (핵심 수정) 균형 잡힌 시장 생성 함수 호출 ---
params = _generate_mu_sigma_balanced(
    d, device, seed=seed,
    target_leverage=0.7 * L_cap,
    dirichlet_conc=10.0,
    hhi_factor=3.0
)
alpha, Sigma, Sigma_inv = params["alpha"], params["Sigma"], params["Sigma_inv"]
chol_S = torch.linalg.cholesky(Sigma)
u_closed_form_unc = (1.0 / gamma) * (Sigma_inv @ alpha)

# --------------------------- 유틸 및 정책 (이전과 동일) ---------------------------
@torch.no_grad()
def _proj_simplex_leq(v: torch.Tensor, mass: float) -> torch.Tensor:
    vpos = v.clamp_min(0.0); s = float(vpos.sum().item())
    if s <= mass: return vpos
    u_sorted, _ = torch.sort(vpos, descending=True); cssv = torch.cumsum(u_sorted, dim=0) - mass
    j = torch.arange(1, v.numel()+1, device=v.device, dtype=v.dtype); cond = u_sorted > (cssv / j)
    rho_idx = torch.nonzero(cond, as_tuple=False)[-1].item() if cond.any() else v.numel() - 1
    theta = cssv[rho_idx] / (rho_idx + 1)
    return (vpos - theta).clamp_min(0.0)

class DirectPolicy(nn.Module):
    def __init__(self):
        super().__init__(); state_dim = DIM_X + DIM_Y + 1; hid = 200
        self.net = nn.Sequential(nn.Linear(state_dim, hid), nn.LeakyReLU(), nn.Linear(hid, hid), nn.LeakyReLU(), nn.Linear(hid, d + 2))
    def forward(self, **states_dict):
        z = torch.cat([states_dict["X"], states_dict["TmT"]], dim=1)
        raw_output = self.net(z)
        u = L_cap * F.softmax(raw_output[:, :d+1], dim=1)[:, :d]
        C = states_dict["X"] * alpha_rel * torch.sigmoid(raw_output[:, d+1:])
        return torch.cat([u, C], dim=1)

class WrappedCFPolicy(nn.Module):
    def __init__(self, u_star, alpha_rel, c_frac):
        super().__init__(); self.register_buffer("u_star", u_star); self.alpha_rel = alpha_rel; self.c_frac = c_frac
    def forward(self, **states_dict):
        B, X = states_dict["X"].shape[0], states_dict["X"]
        u = self.u_star.unsqueeze(0).expand(B, -1)
        C = torch.minimum(self.c_frac * X, self.alpha_rel * X)
        return torch.cat([u, C], dim=1)

def build_closed_form_policy():
    u_star = _proj_simplex_leq(u_closed_form_unc.clone(), L_cap)
    return WrappedCFPolicy(u_star.to(device), alpha_rel, 0.50).to(device), None

# --------------------------- 초기 상태 및 시뮬레이터 (이전과 동일) ---------------------------
def sample_initial_states(B, *, rng=None):
    X0 = torch.rand((B, DIM_X), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
    TmT0 = torch.rand((B, 1), device=device, generator=rng) * T
    return {"X": X0, "TmT": TmT0, "Y": None}, TmT0 / float(m)

def simulate(policy, B, *, train=True, rng=None, initial_states_dict=None, random_draws=None, m_steps=None):
    m_eff = m_steps or m
    states, dt = sample_initial_states(B, rng=rng) if initial_states_dict is None else (initial_states_dict, initial_states_dict['TmT']/m_eff)
    Z = random_draws[0] if random_draws is not None else torch.randn(B, m_eff, d, device=device, generator=rng)
    logX, TmT0 = torch.log(states["X"].clamp_min(lb_X)), states["TmT"]
    util_cons = torch.zeros((B, 1), device=device)
    for i in range(m_eff):
        t_i = T - (TmT0 - i * dt); cur_states = {"X": torch.exp(logX), "TmT": TmT0 - i * dt}
        policy_output = policy(**cur_states); u, C = policy_output[:, :d], policy_output[:, d:]
        drift = r + (u * alpha).sum(1, keepdim=True) - 0.5 * (u.unsqueeze(1)@Sigma@u.unsqueeze(-1)).squeeze(-1) - C/torch.exp(logX)
        dBX = (u @ chol_S * Z[:, i, :]).sum(1, keepdim=True)
        logX += drift * dt + dBX * dt.sqrt()
        uC = (C.clamp_min(1e-12).pow(1-gamma)-1)/(1-gamma) if gamma!=1.0 else torch.log(C.clamp_min(1e-12))
        util_cons += torch.exp(-rho * t_i) * uC * dt
    uT = (torch.exp(logX).pow(1-gamma)-1)/(1-gamma) if gamma!=1.0 else logX
    return (util_cons + kappa * math.exp(-rho * T) * uT).view(-1)

def get_traj_schema():
    u_labels = [f"u_{i+1}" for i in range(d)] + ["Consumption"]
    return {
        "roles": {"X": {"dim": DIM_X, "labels": ["Wealth"]}, "U": {"dim": DIM_U, "labels": u_labels}},
        "views": [
            {"name": "Consumption_Path", "block": "U", "mode": "indices", "indices": [d], "ylabel": "Consumption (C)"},
            {"name": "Wealth_Path", "block": "X", "mode": "indices", "indices": [0], "ylabel": "Wealth (X)"},
        ], "sampling": {"Bmax": 5}
    }