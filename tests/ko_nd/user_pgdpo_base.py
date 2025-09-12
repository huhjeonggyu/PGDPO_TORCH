# user_pgdpo_base.py
# 역할: 다차원 모델의 모든 사용자 정의 요소를 설정합니다.
#       (1) 모델 차원 및 파라미터, (2) 정책 신경망, (3) 시뮬레이션 동역학
import os
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import dirichlet

# ver2 프레임워크와 연동될 분석적 해법 모듈 import
from closed_form_ref import precompute_ABC, ClosedFormPolicy

# --- 모델별 설정 및 환경변수 오버라이드 블록 ---

# 1. 모델 고유의 기본값을 설정합니다.
d = 5
k = 3
epochs = 200
batch_size = 1024
lr = 1e-4
seed = 7

# 2. 환경변수가 존재하면 기본값을 덮어씁니다.
d = int(os.getenv("PGDPO_D", d))
k = int(os.getenv("PGDPO_K", k))
epochs = int(os.getenv("PGDPO_EPOCHS", epochs))
batch_size = int(os.getenv("PGDPO_BATCH_SIZE", batch_size))
lr = float(os.getenv("PGDPO_LR", lr))
seed = int(os.getenv("PGDPO_SEED", seed))

# --- 블록 끝 ---


# ==============================================================================
# ===== (A) 사용자 정의 영역: 모델 차원, 파라미터, 하이퍼파라미터 =====
# ==============================================================================

# --------------------------- Model Dimensions ---------------------------
# d: 자산 수, k: 팩터 수
# d = 5 <-- 상단 블록에서 제어
# k = 3 <-- 상단 블록에서 제어
DIM_X = 1      # 자산 X는 부(wealth)를 의미하므로 1차원
DIM_Y = k      # 외생 변수 Y는 k개의 팩터
DIM_U = d      # 제어 u는 d개 자산에 대한 투자 비율

# --------------------------- Config ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seed = 7 # pgdpo_base.py에서 이 변수를 사용해 실제 시드를 설정 <-- 상단 블록에서 제어

# --------------------------- Market & Utility Parameters ---------------------------
r = 0.03
gamma = 2.0

# --------------------------- Simulation Parameters ---------------------------
T = 1.
m = 40

# --------------------------- State & Control Ranges ---------------------------
X0_range = (0.1, 3.0)
u_cap = 10.
lb_X  = 1e-5

# --------------------------- Training Hyperparameters ---------------------------
# epochs     = 200 <-- 상단 블록에서 제어
# batch_size = 1024 <-- 상단 블록에서 제어
# lr         = 1e-4 <-- 상단 블록에서 제어

# --------------------------- Evaluation Parameters ---------------------------
N_eval_states = 2000
CRN_SEED_EU   = 12345

# --------------------------- Parameter Generation ---------------------------
def _generate_market_params(d, k, r, dev, seed=None):
    """다차원 시장 파라미터를 생성하고 torch 텐서로 반환합니다."""
    # pgdpo_base.py에서 이미 NumPy/PyTorch 시드를 모두 설정했지만,
    # 이 함수가 독립적으로도 결정적으로 작동하도록 시드를 한 번 더 명시합니다.
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 팩터 동역학 파라미터 (k x k, k)
    kappa_Y = torch.diag(torch.linspace(2.0, 2.0 + (k-1)*0.5, k)) if k > 0 else torch.empty(0,0)
    theta_Y = torch.linspace(0.2, 0.4, k) if k > 0 else torch.empty(0)
    sigma_Y = torch.diag(torch.linspace(0.3, 0.5, k)) if k > 0 else torch.empty(0,0)

    # 자산별 파라미터 (d, d x k)
    sigma = torch.linspace(0.2, 0.4, d)
    # ===== [핵심 수정 2] =====
    # Scipy 난수 생성에 random_state를 명시하여 재현성 보장
    alpha_np = dirichlet.rvs([1.0] * k, size=d, random_state=seed) if k > 0 else np.zeros((d,0))
    alpha = torch.tensor(alpha_np, dtype=torch.float32)

    # 상관관계 구조 (d x d, k x k, d x k)
    beta_corr = torch.empty(d).uniform_(-0.8, 0.8)
    Psi = torch.outer(beta_corr, beta_corr); Psi.fill_diagonal_(1.0)
    
    if k > 0:
        Z_Y = torch.randn(k, max(1, k)); corr_Y = Z_Y @ Z_Y.T
        d_inv_sqrt_Y = torch.diag(1.0 / torch.sqrt(torch.diag(corr_Y).clamp(min=1e-8)))
        Phi_Y = d_inv_sqrt_Y @ corr_Y @ d_inv_sqrt_Y; Phi_Y.fill_diagonal_(1.0)
        rho_Y = torch.empty(d, k).uniform_(-0.2, 0.2)
    else:
        Phi_Y = torch.empty(0,0)
        rho_Y = torch.empty(d,0)

    # 전체 상관 행렬 (Positive-Definite 보정 포함)
    block_corr = torch.zeros((d + k, d + k), dtype=torch.float32)
    block_corr[:d, :d], block_corr[d:, d:] = Psi, Phi_Y
    if k > 0:
        block_corr[:d, d:], block_corr[d:, :d] = rho_Y, rho_Y.T
    
    eigvals = torch.linalg.eigvalsh(block_corr)
    if eigvals.min() < 1e-6:
        block_corr += (abs(eigvals.min()) + 1e-4) * torch.eye(d + k)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(block_corr)))
        block_corr = D_inv_sqrt @ block_corr @ D_inv_sqrt
        Psi = block_corr[:d, :d]
        if k > 0:
            Phi_Y, rho_Y = block_corr[d:, d:], block_corr[:d, d:]

    params = {
        'kappa_Y': kappa_Y, 'theta_Y': theta_Y, 'sigma_Y': sigma_Y,
        'sigma': sigma, 'alpha': alpha, 'Phi_Y': Phi_Y, 'Psi': Psi, 'rho_Y': rho_Y
    }
    params['Sigma'] = torch.diag(params['sigma']) @ params['Psi'] @ torch.diag(params['sigma'])
    params['Sigma_inv'] = torch.linalg.inv(params['Sigma'])
    params['block_corr'] = block_corr
    params['cholesky_L'] = torch.linalg.cholesky(params['block_corr'])
    
    return {p_key: v.to(dev) for p_key, v in params.items()}

# 파라미터 생성 실행
params = _generate_market_params(d, k, r, device, seed)

# 자주 사용하는 파라미터를 전역 변수로 추출
alpha, sigma, kappa_Y, theta_Y, sigma_Y, rho_Y, Psi, Phi_Y = [params[key] for key in ['alpha', 'sigma', 'kappa_Y', 'theta_Y', 'sigma_Y', 'rho_Y', 'Psi', 'Phi_Y']]
Sigma, Sigma_inv, block_corr, cholesky_L = params['Sigma'], params['Sigma_inv'], params['block_corr'], params['cholesky_L']

# Y의 평균 범위를 계산 (상태 샘플링에 사용)
if k > 0:
    Y_min_vec = theta_Y - 3 * torch.diag(sigma_Y)
    Y_max_vec = theta_Y + 3 * torch.diag(sigma_Y)
    Y0_range = (Y_min_vec, Y_max_vec)
else:
    Y0_range = None


# ==============================================================================
# ===== (B) 사용자 정의 영역: 정책 네트워크 및 모델 동역학 =====
# ==============================================================================

class DirectPolicy(nn.Module):
    """다차원 입출력을 동적으로 처리하는 정책 신경망입니다."""
    def __init__(self):
        super().__init__()
        state_dim = DIM_X + DIM_Y + 1
        self.net = nn.Sequential(
            nn.Linear(state_dim, 200), nn.LeakyReLU(),
            nn.Linear(200, 200), nn.LeakyReLU(),
            nn.Linear(200, DIM_U)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, gain=0.8)

    def forward(self, **states_dict):
        states_to_cat = [states_dict['X'], states_dict['TmT']]
        if 'Y' in states_dict and k > 0:
            states_to_cat.append(states_dict['Y'])
        x_in = torch.cat(states_to_cat, dim=1)
        u = self.net(x_in)
        return torch.clamp(u, -u_cap, u_cap)

def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    """다차원 초기 상태 (X, Y, TmT)를 샘플링하고 딕셔너리로 반환합니다."""
    X0 = torch.rand((B, DIM_X), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
    
    Y0 = None
    if DIM_Y > 0:
        Y_min, Y_max = Y0_range
        Y0 = Y_min.unsqueeze(0) + (Y_max - Y_min).unsqueeze(0) * torch.rand((B, DIM_Y), device=device, generator=rng)
        
    TmT0 = torch.rand((B, 1), device=device, generator=rng) * T
    dt_vec = TmT0 / float(m)

    states = {'X': X0, 'TmT': TmT0}
    if Y0 is not None:
        states['Y'] = Y0
    return states, dt_vec

def simulate(policy, B, *, train=True, rng=None, initial_states_dict=None, random_draws=None, m_steps=None):
    """다차원 SDE를 시뮬레이션합니다."""
    m_eff = m_steps if m_steps is not None else m
    
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states, dt = initial_states_dict, initial_states_dict['TmT'] / float(m_eff)
        
    logX = states['X'].clamp_min(lb_X).log()
    Y = states.get('Y')
    
    if random_draws is None:
        uncorrelated_Z = torch.randn(B, m_eff, d + k, device=device, generator=rng)
        Z = torch.einsum('bmd,dn->bmn', uncorrelated_Z, cholesky_L)
        ZX, ZY = Z[:, :, :d], Z[:, :, d:]
    else:
        ZX, ZY = random_draws

    for i in range(m_eff):
        current_states = {'X': logX.exp(), 'TmT': states['TmT'] - i * dt}
        if Y is not None: current_states['Y'] = Y
        
        u = policy(**current_states)
        
        if Y is not None:
            risk_premium = alpha @ Y.unsqueeze(-1)
            drift_term = r + (u * sigma * risk_premium.squeeze(-1)).sum(1, keepdim=True)
            var_term = torch.einsum('bi,bi->b', (u * sigma) @ Psi, (u * sigma)).view(-1, 1)
            Y_drift = (theta_Y - Y) @ kappa_Y.T
            dBY = (sigma_Y @ ZY[:, i, :].T).T
            Y = Y + Y_drift * dt + dBY * dt.sqrt()
        else: # Merton case k=0
            drift_term = r + (u * alpha).sum(1, keepdim=True)
            var_term = torch.einsum('bi,bi->b', u @ Sigma, u).view(-1, 1)

        dBX = (u * sigma * ZX[:, i, :]).sum(1, keepdim=True)
        logX = logX + (drift_term - 0.5 * var_term) * dt + dBX * dt.sqrt()

    XT = logX.exp().clamp_min(lb_X)
    return (XT.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)

class WrappedCFPolicy(nn.Module):
    """ClosedFormPolicy가 ver2의 표준 입력 `**states_dict`를 받도록 감싸는 래퍼 클래스"""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
    def forward(self, **states_dict):
        return self.policy.forward(**states_dict)

def build_closed_form_policy():
    """분석적 해를 계산하고, ver2 프레임워크와 호환되는 정책 모듈을 빌드합니다."""
    if k == 0: # Merton case
        u_star = ((1.0 / gamma) * (torch.linalg.inv(Sigma) @ alpha.T)).T if alpha.numel() > 0 else torch.zeros(d, device=device)
        from mt_nd.user_pgdpo_base import WrappedCFPolicy as MertonCF
        return MertonCF(u_star), None

    ode_solution = precompute_ABC(params, T, gamma)
    cf_policy = ClosedFormPolicy(ode_solution, params, T, gamma, u_cap).to(device)
    return WrappedCFPolicy(cf_policy), ode_solution