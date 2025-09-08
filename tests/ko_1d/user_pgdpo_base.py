# user_pgdpo_base.py for 1-dimensional Kim-Omberg Problem
# 역할: 1차원 Kim-Omberg 모델의 모든 사용자 정의 요소를 설정합니다.
#       (고정된 표준 파라미터 사용)
import os
import torch
import torch.nn as nn
from closed_form_ref import precompute_ABC, ClosedFormPolicy

# --- 모델별 설정 및 환경변수 오버라이드 블록 ---

# 1. 모델 고유의 기본값을 설정합니다.
# ✨ ko_1d 모델은 d=1, k=1 로 차원이 고정되므로, 환경변수에서 읽지 않습니다.
d = 1
k = 1 
epochs = 200
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
# d = 1  # ✨ 1개의 위험 자산 <-- 상단 블록에서 고정
# k = 1  # ✨ 1개의 확률 팩터 <-- 상단 블록에서 고정

DIM_X = 1
DIM_Y = k
DIM_U = d

# --------------------------- Config ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seed = 42 <-- 상단 블록에서 제어

# --------------------------- Market & Utility Parameters ---------------------------
r = 0.03
gamma = 2.0

# --------------------------- ✨ Fixed Canonical Parameters ---------------------------
# Kim-Omberg 논문 등에서 자주 사용되는 표준 파라미터 값들을 상수로 직접 정의합니다.
params_val = {
    'kappa_Y': torch.tensor([[2.0]]),      # 팩터의 평균 회귀 속도
    'theta_Y': torch.tensor([0.2]),        # 팩터의 장기 평균
    'sigma_Y': torch.tensor([[0.3]]),      # 팩터의 변동성
    'sigma':   torch.tensor([0.2]),        # 자산의 변동성
    'alpha':   torch.tensor([[1.0]]),      # 자산-팩터 민감도
    'rho_Y':   torch.tensor([[0.1]]),      # 자산-팩터 상관관계
    'Psi':     torch.tensor([[1.0]]),
    'Phi_Y':   torch.tensor([[1.0]]),
}
params = {key: val.to(device) for key, val in params_val.items()}

# 프레임워크에서 사용할 파라미터 형태로 변환
params['Sigma'] = torch.diag(params['sigma']) @ params['Psi'] @ torch.diag(params['sigma'])
params['Sigma_inv'] = torch.linalg.inv(params['Sigma'])
params['block_corr'] = torch.tensor([[1.0, params['rho_Y'].item()], [params['rho_Y'].item(), 1.0]], device=device)
params['cholesky_L'] = torch.linalg.cholesky(params['block_corr'])

# 전역 변수로 추출
alpha, sigma, kappa_Y, theta_Y, sigma_Y, rho_Y, Psi, Phi_Y = \
    [params[key] for key in ['alpha', 'sigma', 'kappa_Y', 'theta_Y', 'sigma_Y', 'rho_Y', 'Psi', 'Phi_Y']]
Sigma, Sigma_inv, block_corr, cholesky_L = \
    params['Sigma'], params['Sigma_inv'], params['block_corr'], params['cholesky_L']

# --------------------------- Simulation & Training Parameters ---------------------------
T = 1.0
m = 20
Y_min_vec = theta_Y - 3 * torch.diag(sigma_Y)
Y_max_vec = theta_Y + 3 * torch.diag(sigma_Y)
Y0_range = (Y_min_vec, Y_max_vec)
X0_range = (0.5, 1.5)
u_cap = 10.0
lb_X  = 1e-5
# epochs = 200 <-- 상단 블록에서 제어
# batch_size = 1024 <-- 상단 블록에서 제어
# lr = 1e-4 <-- 상단 블록에서 제어
N_eval_states = 2000
CRN_SEED_EU = 12345

# ==============================================================================
# ===== (B) 사용자 정의 영역: 정책 네트워크, 시뮬레이션, 분석적 해 =====
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
        x_in = torch.cat([states_dict['X'], states_dict['TmT'], states_dict['Y']], dim=1)
        return torch.clamp(self.net(x_in), -u_cap, u_cap)

def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    X0 = torch.rand((B, DIM_X), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
    Y0 = Y0_range[0].unsqueeze(0) + (Y0_range[1] - Y0_range[0]).unsqueeze(0) * torch.rand((B, DIM_Y), device=device, generator=rng)
    TmT0 = torch.rand((B, 1), device=device, generator=rng) * T
    dt_vec = TmT0 / float(m)
    states = {'X': X0, 'Y': Y0, 'TmT': TmT0}
    return states, dt_vec

def simulate(policy, B, *, train=True, rng=None, initial_states_dict=None, random_draws=None, m_steps=None):
    m_eff = m if m_steps is None else m_steps
    states, dt = sample_initial_states(B, rng=rng) if initial_states_dict is None else (initial_states_dict, initial_states_dict['TmT'] / float(m_eff))
    logX, Y = states['X'].clamp_min(lb_X).log(), states['Y']
    
    if random_draws is None:
        Z = torch.randn(B, m_eff, d + k, device=device, generator=rng)
        Z_corr = torch.einsum('bmd,dn->bmn', Z, cholesky_L)
        ZX, ZY = Z_corr[..., :d], Z_corr[..., d:]
    else:
        ZX, ZY = random_draws

    for i in range(m_eff):
        current_states = {'X': logX.exp(), 'Y': Y, 'TmT': states['TmT'] - i * dt}
        u = policy(**current_states)
        
        risk_premium = alpha @ Y.unsqueeze(-1)
        drift_term = r + (u * sigma * risk_premium.squeeze(-1)).sum(1, keepdim=True)
        var_term = torch.einsum('bi,bi->b', (u * sigma) @ Psi, (u * sigma)).view(-1, 1)
        
        Y_drift = (theta_Y - Y) @ kappa_Y.T
        dBY = (sigma_Y @ ZY[:, i, :].T).T
        Y = Y + Y_drift * dt + dBY * dt.sqrt()
        
        dBX = (u * sigma * ZX[:, i, :]).sum(1, keepdim=True)
        logX = logX + (drift_term - 0.5 * var_term) * dt + dBX * dt.sqrt()

    XT = logX.exp().clamp_min(lb_X)
    return (XT.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)

class WrappedCFPolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
    def forward(self, **states_dict):
        return self.policy.forward(**states_dict)

def build_closed_form_policy():
    ode_solution = precompute_ABC(params, T, gamma)
    cf_policy = ClosedFormPolicy(ode_solution, params, T, gamma, u_cap).to(device)
    return WrappedCFPolicy(cf_policy), ode_solution