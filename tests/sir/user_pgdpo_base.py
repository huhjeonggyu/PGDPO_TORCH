# user_pgdpo_base.py for the Bensoussan-Park (2024) Model
# 역할: 확률적 노동 소득이 있는 최적 소비/투자 문제의 모든 사용자 정의 요소를 설정합니다.
import os
import torch
import torch.nn as nn

# --- 모델별 설정 및 환경변수 오버라이드 블록 ---

# 1. 모델 고유의 기본값을 설정합니다.
# ✨ Park 모델은 d=1, k=1 로 차원이 고정되므로, 환경변수에서 읽지 않습니다.
d = 1
k = 1
epochs = 250
batch_size = 1024
lr = 1e-4
seed = 42

# 2. 변경 가능한 하이퍼파라미터만 환경변수로부터 덮어씁니다.
epochs = int(os.getenv("PGDPO_EPOCHS", epochs))
batch_size = int(os.getenv("PGDPO_BATCH_SIZE", batch_size))
lr = float(os.getenv("PGDPO_LR", lr))
seed = int(os.getenv("PGDPO_SEED", seed))

# --- 블록 끝 ---


# ==============================================================================
# ===== (A) 사용자 정의 영역: 모델 차원, 파라미터, 하이퍼파라미터 =====
# ==============================================================================

# --------------------------- Model Dimensions ---------------------------
# d = 1  <-- 상단 블록에서 고정
# k = 1  <-- 상단 블록에서 고정
DIM_X = 1
DIM_Y = k
DIM_U = d + 1 # 제어 u = [투자 비중(d), 소비율(1)]

# --------------------------- Config ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seed = 42 <-- 상단 블록에서 제어

# --------------------------- Market & Utility Parameters ---------------------------
beta = 0.01
r = 0.06
alpha_market = 0.1
sigma_market = 0.25
mu_income = 0.05
rho_income = 0.02

theta = torch.tensor([(alpha_market - r) / sigma_market], device=device)
sigma = torch.tensor([[sigma_market]], device=device)
Sigma_inv = torch.inverse(sigma @ sigma.T)

# --------------------------- Simulation & Training Parameters ---------------------------
T = 1.5
m = 20
X0_range = (0.5, 2.0)
z0_range = (0.01, 1.5)
u_cap = 10.
lb_X = 1e-5
# epochs = 250 <-- 상단 블록에서 제어
# batch_size = 1024 <-- 상단 블록에서 제어
# lr = 1e-4 <-- 상단 블록에서 제어
N_eval_states = 2000
CRN_SEED_EU = 12345
Y0_range = None

# ==============================================================================
# ===== (B) 사용자 정의 영역: 정책 네트워크 및 모델 동역학 =====
# ==============================================================================
class DirectPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        state_dim = DIM_X + DIM_Y + 1
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, 128), nn.LeakyReLU(),
            nn.Linear(128, DIM_U)
        )

    def forward(self, **states_dict):
        x_in = torch.cat([states_dict['X'], states_dict['Y'], states_dict['TmT']], dim=1)
        u = self.net(x_in)
        investment_alloc = torch.clamp(u[:, :d], -u_cap, u_cap)
        consumption_rate = torch.clamp(u[:, d:], 1e-4, u_cap)
        return torch.cat([investment_alloc, consumption_rate], dim=1)

def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    X0 = torch.rand((B, DIM_X), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
    z0 = torch.rand((B, 1), device=device, generator=rng) * (z0_range[1] - z0_range[0]) + z0_range[0]
    Y0 = z0 * X0
    TmT0 = torch.rand((B, 1), device=device, generator=rng) * T
    dt_vec = TmT0 / float(m)
    states = {'X': X0, 'Y': Y0, 'TmT': TmT0}
    return states, dt_vec

def simulate(policy, B, *, train=True, rng=None, initial_states_dict=None, random_draws=None, m_steps=None):
    m_eff = m_steps if m_steps is not None else m
    
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states, dt = initial_states_dict, initial_states_dict['TmT'] / float(m_eff)
        
    X = states['X'].clamp_min(lb_X)
    Y = states['Y']
    
    if random_draws is None:
        ZX = torch.randn(B, m_eff, d, device=device, generator=rng)
        ZY = torch.randn(B, m_eff, k, device=device, generator=rng)
    else:
        ZX, ZY = random_draws

    cumulative_log_consumption = 0.0
    
    for i in range(m_eff):
        current_states = {'X': X, 'Y': Y, 'TmT': states['TmT'] - i * dt}
        
        u = policy(**current_states)
        investment_alloc = u[:, :d]
        consumption_rate = u[:, d:]

        market_risk_premium = (investment_alloc @ sigma) * theta
        drift_X = X * (r + market_risk_premium - consumption_rate) + Y
        
        diffusion_X = X * (investment_alloc @ sigma) * ZX[:, i, :]
        X = X + drift_X * dt + diffusion_X * dt.sqrt()
        X = X.clamp_min(lb_X)

        drift_Y = Y * mu_income
        diffusion_Y = Y * rho_income * ZY[:, i, :]
        Y = Y + drift_Y * dt + diffusion_Y * dt.sqrt()

        time_t = T - (states['TmT'] - i * dt)
        consumption_value = consumption_rate * X
        log_consumption = torch.log(consumption_value.clamp_min(1e-8))
        cumulative_log_consumption += torch.exp(-beta * time_t) * log_consumption * dt

    return cumulative_log_consumption

def build_closed_form_policy():
    return None, None