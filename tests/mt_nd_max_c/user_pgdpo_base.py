# 파일: tests/mt_nd_max_c/user_pgdpo_base.py
# 모델: 소비 상한 모델 (신경망이 소비 비율을 학습) - 최종 버전

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- 기본 설정 (변경 없음) ---------------------------
d = int(os.getenv("PGDPO_D", 5)); k = 0
epochs = int(os.getenv("PGDPO_EPOCHS", 200)); batch_size = int(os.getenv("PGDPO_BS", 1024))
lr = float(os.getenv("PGDPO_LR", 1e-4)); seed = int(os.getenv("PGDPO_SEED", 24))
T = float(os.getenv("PGDPO_T", 1.5)); m = int(os.getenv("PGDPO_M", 20))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIM_X = 1; DIM_Y = k; DIM_U = d + 1
N_eval_states = int(os.getenv("PGDPO_EVAL_STATES", 100)); CRN_SEED_EU = int(os.getenv("PGDPO_CRN", 1352))

# --------------------------- 경제 파라미터 (변경 없음) ---------------------------
gamma = float(os.getenv("PGDPO_GAMMA", 2.0)); rho   = float(os.getenv("PGDPO_RHO", 0.10))
r     = float(os.getenv("PGDPO_RF", 0.03)); kappa = float(os.getenv("PGDPO_KAPPA", 1.0))
C_abs_cap = float(os.getenv("PGDPO_C_CAP", 10.15))
L_cap = float(os.getenv("PGDPO_LCAP", 3.0)) 
X0_range = (0.1, 3.0); lb_X = 1e-6

# --------------------------- 시장 파라미터 생성 (변경 없음) ---------------------------
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

params = _generate_mu_sigma_balanced(
    d, device, seed=seed, target_leverage=L_cap,
    dirichlet_conc=10.0, hhi_factor=3.0
)
alpha, Sigma, Sigma_inv = params["alpha"], params["Sigma"], params["Sigma_inv"]
chol_S = torch.linalg.cholesky(Sigma)

# --------------------------- 유틸 및 정책 ---------------------------

class DirectPolicy(nn.Module):
    """
    두 단계 정책:
      1) consumption_net: (X, TmT) -> C (절대 소비)
      2) investment_net: (X_eff, TmT) -> u  where  X_eff = X - C  (기본)
         - 환경변수 PGDPO_TWO_STAGE_MODE=dt 로 설정하면 X_eff = X - C*dt 로 변경
    """
    def __init__(self):
        super().__init__()
        state_dim = DIM_X + DIM_Y + 1
        hid = 200
        # 1) 소비망: 스칼라 출력(logit) -> sigmoid -> 비율 -> 절대 소비
        self.consumption_net = nn.Sequential(
            nn.Linear(state_dim, hid), nn.LeakyReLU(),
            nn.Linear(hid, hid), nn.LeakyReLU(),
            nn.Linear(hid, 1)
        )
        # 2) 투자망: d차원 출력 -> tanh -> [-L_cap, L_cap] 스케일
        self.investment_net  = nn.Sequential(
            nn.Linear(state_dim, hid), nn.LeakyReLU(),
            nn.Linear(hid, hid), nn.LeakyReLU(),
            nn.Linear(hid, d)
        )
        # 시뮬레이터가 두 단계 경로를 사용할지 식별용 플래그
        self.two_stage = True

    # --- Stage 1: C(X, TmT)
    def forward_consumption(self, **states_dict):
        z = torch.cat([states_dict["X"], states_dict["TmT"]], dim=1)
        ratio = torch.sigmoid(self.consumption_net(z))  # (0,1)
        C_prop = states_dict["X"] * ratio
        C = torch.minimum(C_prop, torch.full_like(C_prop, C_abs_cap))
        return C

    # --- Stage 2: u(X_eff, TmT)
    def forward_investment(self, **states_dict):
        z_inv = torch.cat([states_dict["X"], states_dict["TmT"]], dim=1)
        u_raw = self.investment_net(z_inv)
        u = L_cap * torch.tanh(u_raw)
        return u

    def forward(self, **states_dict):
        C = self.forward_consumption(**states_dict)

        # 효과적 웰스 계산 규칙: 기본은 X - C,  환경변수로 dt 모드 선택 가능
        mode = os.getenv("PGDPO_TWO_STAGE_MODE", "abs").lower()  # 'abs' | 'dt'
        X = states_dict["X"]
        if mode == "dt":
            # heuristic: 현재 step의 dt를 TmT/m로 근사
            dt_guess = (states_dict["TmT"] / m).clamp_min(0.0)
            X_eff = (X - C * dt_guess).clamp_min(lb_X)
        else:
            X_eff = (X - C).clamp_min(lb_X)

        u = self.forward_investment(X=X_eff, TmT=states_dict["TmT"])
        return torch.cat([u, C], dim=1)


class WrappedCFPolicy(nn.Module):
    def __init__(self, u_star, c_cap_abs, c_frac):
        super().__init__(); self.register_buffer("u_star", u_star); self.c_cap_abs = c_cap_abs; self.c_frac = c_frac
    
    def forward(self, **states_dict):
        B, X = states_dict["X"].shape[0], states_dict["X"]
        u = self.u_star.unsqueeze(0).expand(B, -1)
        C = torch.minimum(self.c_frac * X, torch.full_like(X, self.c_cap_abs))
        return torch.cat([u, C], dim=1)

def build_closed_form_policy():
    u_star = (1.0 / gamma) * (Sigma_inv @ alpha)
    theta_sq = (alpha.view(1, -1) @ Sigma_inv @ alpha.view(-1, 1)).item()
    m_star = (rho - (1 - gamma) * (r + theta_sq / (2 * gamma))) / gamma
    print(f"✅ Analytical policy computed: m*={m_star:.4f}, C_abs_cap={C_abs_cap:.4f}")
    return WrappedCFPolicy(u_star.to(device), C_abs_cap, m_star).to(device), None

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

    TWO_STAGE_MODE = os.getenv("PGDPO_TWO_STAGE_MODE", "abs").lower()  # 'abs' | 'dt'

    for i in range(m_eff):
        t_i = T - (TmT0 - i * dt)
        current_X = torch.exp(logX)
        cur_states = {"X": current_X, "TmT": TmT0 - i * dt}

        if getattr(policy, "two_stage", False):
            # 1) 소비 먼저
            C = policy.forward_consumption(**cur_states)
            C = torch.min(C, current_X)  # 파산 방지

            # 2) 소비 반영한 웰스에서 투자 결정
            if TWO_STAGE_MODE == "dt":
                X_eff = (current_X - C * dt).clamp_min(lb_X)
            else:
                X_eff = (current_X - C).clamp_min(lb_X)

            u = policy.forward_investment(X=X_eff, TmT=cur_states["TmT"])
        else:
            # 단일 네트워크 경로(기존)
            out = policy(**cur_states)
            u, C = out[:, :d], out[:, d:]
            C = torch.min(C, current_X)  # 파산 방지

        # --- 동학 업데이트(소비는 연속시간 흐름으로 logX에서 -C/X * dt 반영) ---
        drift = r + (u * alpha).sum(1, keepdim=True) \
                  - 0.5 * (u.unsqueeze(1) @ Sigma @ u.unsqueeze(-1)).squeeze(-1) \
                  - C / current_X
        dBX = (u @ chol_S * Z[:, i, :]).sum(1, keepdim=True)
        logX += drift * dt + dBX * dt.sqrt()

        # 효용 적분
        uC = (C.clamp_min(1e-12).pow(1-gamma)-1)/(1-gamma) if gamma != 1.0 else torch.log(C.clamp_min(1e-12))
        util_cons += torch.exp(-rho * t_i) * uC * dt

    uT = (torch.exp(logX).pow(1-gamma)-1)/(1-gamma) if gamma != 1.0 else logX
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
        # ✨ 수정된 부분: 범례(legend)의 이름을 직접 지정하는 항목 추가
        "legend_labels": {
            "cf": "Analytical Solution", # 'cf' 정책의 라벨을 지정합니다.
            "learn": "Learned Policy"
        }
    }