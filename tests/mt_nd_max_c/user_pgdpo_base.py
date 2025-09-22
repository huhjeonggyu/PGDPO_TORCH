# 파일: tests/mt_nd_max_c/user_pgdpo_base.py
# 모델: 소비 상한 모델 (소비 비율 + 투자 비율 모두 학습 가능; U_MODE 토글) - 상대 상한 버전

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- 기본 설정 ---------------------------
d = int(os.getenv("PGDPO_D", 5)); k = 0
epochs = int(os.getenv("PGDPO_EPOCHS", 200)); batch_size = int(os.getenv("PGDPO_BS", 1024))
lr = float(os.getenv("PGDPO_LR", 1e-4)); seed = int(os.getenv("PGDPO_SEED", 24))
T = float(os.getenv("PGDPO_T", 1.5)); m = int(os.getenv("PGDPO_M", 20))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIM_X = 1; DIM_Y = k; DIM_U = d + 1
N_eval_states = int(os.getenv("PGDPO_EVAL_STATES", 100)); CRN_SEED_EU = int(os.getenv("PGDPO_CRN", 1352))

# --------------------------- 경제 파라미터 ---------------------------
gamma = float(os.getenv("PGDPO_GAMMA", 2.0))
rho   = float(os.getenv("PGDPO_RHO", 0.10))
r     = float(os.getenv("PGDPO_RF", 0.03))
kappa = float(os.getenv("PGDPO_KAPPA", 1.0))

# 상대(부 비례) 소비 상한: C_max(X) = c_frac * X
c_frac    = float(os.getenv("PGDPO_C_FRAC", 0.7))
L_cap     = float(os.getenv("PGDPO_LCAP", 3.0))   # u 안정화를 위한 소프트 박스

X0_range = (0.1, 3.0); lb_X = 1e-6

# --------------------------- 시장 파라미터 생성 ---------------------------
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
    if (w_ref**2).sum() > (hhi_factor / d):
        w_ref = (0.6 * w_ref) + 0.4 * (torch.ones(d, device=dev) / d)
    if target_leverage is not None:
        s = float(target_leverage)
    else:
        s = float(alpha_mid / (gamma * (Sigma @ w_ref).mean().clamp_min(1e-8)).item())
    alpha = gamma * s * (Sigma @ w_ref)
    if noise_std > 0:
        alpha += float(noise_std) * alpha.abs().mean() * torch.randn_like(alpha)
        alpha = alpha.clamp_min(1e-4)
    Sigma_inv = torch.linalg.inv(Sigma)
    return {"alpha": alpha, "Sigma": Sigma, "Sigma_inv": Sigma_inv}

params = _generate_mu_sigma_balanced(
    d, device, seed=seed, target_leverage=None,
    dirichlet_conc=10.0, hhi_factor=3.0
)
alpha, Sigma, Sigma_inv = params["alpha"], params["Sigma"], params["Sigma_inv"]
chol_S = torch.linalg.cholesky(Sigma)

# --------------------------- U 학습 모드 토글 ---------------------------
U_MODE = os.getenv("PGDPO_U_MODE", "free")       # "free" | "residual" | "const"
U_RESID_SCALE = float(os.getenv("PGDPO_U_RESID_SCALE", "0.2"))
U_CLAMP = float(os.getenv("PGDPO_U_CLAMP", str(L_cap)))  # 마지막 안전 클램프 범위

# --------------------------- 유틸 및 정책 ---------------------------

class DirectPolicy(nn.Module):
    """
    신경망이 '소비 비율'과 '투자 비율'을 학습.
      - 소비: C = c_frac * X * sigmoid(f(X, TmT))
      - 투자:
          U_MODE="const"    : u = u_Merton (상수)
          U_MODE="residual" : u = u_Merton + s * tanh(g(X, TmT))
          U_MODE="free"     : u = L_cap * tanh(g(X, TmT))
    """
    def __init__(self):
        super().__init__()
        state_dim = DIM_X + DIM_Y + 1  # X, TmT
        hid = 200
        self.consumption_net = nn.Sequential(
            nn.Linear(state_dim, hid), nn.LeakyReLU(),
            nn.Linear(hid, hid), nn.LeakyReLU(),
            nn.Linear(hid, 1)
        )
        # 머튼 상수 벡터
        u_star = (1.0 / gamma) * (Sigma_inv @ alpha)  # (d,)
        self.register_buffer("u_const", u_star)

        if U_MODE in {"residual", "free"}:
            self.investment_net = nn.Sequential(
                nn.Linear(state_dim, hid), nn.LeakyReLU(),
                nn.Linear(hid, hid), nn.LeakyReLU(),
                nn.Linear(hid, d)
            )
        self.two_stage = True  # 코어 호환용

    def forward_consumption(self, **states_dict):
        z = torch.cat([states_dict["X"], states_dict["TmT"]], dim=1)  # (B,2)
        ratio = torch.sigmoid(self.consumption_net(z))                 # (B,1) ∈ (0,1)
        C = c_frac * states_dict["X"] * ratio                          # 상대 상한
        return C

    def forward_investment(self, **states_dict):
        B = states_dict["X"].shape[0]
        z_inv = torch.cat([states_dict["X"], states_dict["TmT"]], dim=1)
        u_raw = self.investment_net(z_inv)
        u = U_CLAMP * torch.tanh(u_raw)
        return u

    def forward(self, **states_dict):
        C = self.forward_consumption(**states_dict)
        u = self.forward_investment(**states_dict)
        return torch.cat([u, C], dim=1)

class WrappedCFPolicy(nn.Module):
    """
    해석적 참조정책(상대 상한 유지):
      u_cf = (1/gamma) Σ^{-1} α  (상수)
      C_cf(t,X) = min{ m(t), c_frac } * X
    """
    def __init__(self, u_star, c_frac, gamma, r, rho, theta_sq, kappa, T):
        super().__init__()
        self.register_buffer("u_star", u_star)  # (d,)
        self.c_frac = float(c_frac)
        self.gamma = float(gamma); self.r = float(r); self.rho = float(rho)
        self.theta_sq = float(theta_sq); self.kappa = float(kappa); self.T = float(T)
        self.m_star = ( self.rho - (1.0 - self.gamma) * ( self.r + self.theta_sq / (2.0 * self.gamma) ) ) / self.gamma
        self.m_T = (self.kappa if self.kappa > 0.0 else 1e-12) ** (-1.0 / self.gamma)

    def forward(self, **states_dict):
        X  = states_dict["X"]           # (B,1)
        tau = states_dict["TmT"]        # (B,1)  남은시간 = T - t
        m_star = torch.tensor(self.m_star, device=X.device, dtype=X.dtype)
        m_T    = torch.tensor(self.m_T,    device=X.device, dtype=X.dtype)
        denom  = 1.0 + (m_star / m_T - 1.0) * torch.exp(-m_star * tau)
        m_t    = (m_star / denom).clamp_min(1e-12)                     # (B,1)
        C = torch.minimum(m_t, torch.full_like(X, self.c_frac)) * X
        u = self.u_star.unsqueeze(0).expand(X.size(0), -1)
        return torch.cat([u, C], dim=1)

def build_closed_form_policy():
    u_star = (1.0 / gamma) * (Sigma_inv @ alpha)
    theta_sq = (alpha.view(1, -1) @ Sigma_inv @ alpha.view(-1, 1)).item()
    print(f"✅ Finite-horizon CF policy: using time-varying m(t),  C_frac={c_frac:.4f}")
    pol = WrappedCFPolicy(
        u_star.to(device), c_frac, gamma, r, rho, theta_sq, kappa, T
    ).to(device)
    return pol, None

# --------------------------- 초기 상태 및 시뮬레이터 ---------------------------
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
        t_i = T - (TmT0 - i * dt)
        current_X = torch.exp(logX)
        cur_states = {"X": current_X, "TmT": TmT0 - i * dt}

        # 1) 소비 결정
        out = policy(**cur_states)
        u, C = out[:, :d], out[:, d:]
        C = torch.min(C, current_X)  # 수치 안전

        # 흐름효용
        if gamma != 1.0:
            uC = (C.clamp_min(1e-12).pow(1-gamma) - 1.0) / (1.0 - gamma)
        else:
            uC = torch.log(C.clamp_min(1e-12))
        util_cons += torch.exp(-rho * t_i) * uC * dt

        # 2) 소비 차감 후 투자 적용
        current_X = current_X - C * dt
        current_X = torch.max(torch.ones_like(current_X) * 1e-3, current_X)
        cur_states = {"X": current_X, "TmT": TmT0 - i * dt}
        out = policy(**cur_states)
        u, _ = out[:, :d], out[:, d:]

        # 로그-부 진화
        drift = r + (u * alpha).sum(1, keepdim=True) \
                  - 0.5 * (u.unsqueeze(1) @ Sigma @ u.unsqueeze(-1)).squeeze(-1)
        dBX = (u @ chol_S * Z[:, i, :]).sum(1, keepdim=True)
        logX += drift * dt + dBX * dt.sqrt()
        current_X = torch.exp(logX)
        current_X = torch.max(torch.ones_like(current_X) * 1e-3, current_X)
        logX = torch.log(current_X)

    # 말기효용
    if gamma != 1.0:
        uT = (torch.exp(logX).pow(1-gamma) - 1.0) / (1.0 - gamma)
    else:
        uT = logX

    return (util_cons + kappa * math.exp(-rho * T) * uT).view(-1)

def get_traj_schema():
    u_labels = [f"u_{i+1}" for i in range(d)] + ["Consumption"]
    return {
        "roles": {"X": {"dim": DIM_X, "labels": ["Wealth"]}, "U": {"dim": DIM_U, "labels": u_labels}},
        "views": [
            {"name": "Consumption_Path", "block": "U", "mode": "indices", "indices": [d], "ylabel": "Consumption (C)"},
            {"name": "Wealth_Path", "block": "X", "mode": "indices", "indices": [0], "ylabel": "Wealth (X)"},
        ],
        "sampling": {"Bmax": 5},
        "legend_labels": {"cf": "Analytical Solution", "learn": "Learned Policy"}
    }