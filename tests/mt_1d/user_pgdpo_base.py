# user_pgdpo_base.py for 1-dimensional Merton Problem
# 역할: 1차원 머튼 모델의 모든 사용자 정의 요소를 설정합니다.
#       (고정된 표준 파라미터 사용)
import os
import torch
import torch.nn as nn

# --- 모델별 설정 및 환경변수 오버라이드 블록 ---

# 1. 모델 고유의 기본값을 설정합니다.
# ✨ mt_1d 모델은 d=1, k=0 으로 차원이 고정되므로, 환경변수에서 읽지 않습니다.
d = 1
k = 0
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
# d = 1  # ✨ 1차원 머튼 문제 <-- 상단 블록에서 고정
# k = 0  # ✨ 팩터 없음     <-- 상단 블록에서 고정

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
# 파라미터 생성 함수 대신, 합리적인 값들을 상수로 직접 정의합니다.
mu_const = 0.09     # 위험 자산의 기대 수익률: 9%
sigma_const = 0.20  # 위험 자산의 변동성: 20%

# 프레임워크에서 사용할 파라미터 형태로 변환
alpha = torch.tensor([mu_const - r], device=device) # 리스크 프리미엄 (mu - r)
Sigma = torch.tensor([[sigma_const**2]], device=device) # 공분산 행렬 (d=1 이므로 분산)
Sigma_inv = torch.tensor([[1.0 / (sigma_const**2)]], device=device)

# 분석적 해 (u*) 미리 계산: u* = (mu - r) / (gamma * sigma^2)
u_closed_form_scalar = (mu_const - r) / (gamma * sigma_const**2)
u_closed_form = torch.tensor([u_closed_form_scalar], device=device)

print(f"✅ 1D Merton model loaded. Analytical solution u* = {u_closed_form_scalar:.4f}")

# --------------------------- Simulation & Training Parameters ---------------------------
T = 1.0
m = 20
X0_range = (0.5, 1.5)
u_cap = 10.
lb_X  = 1e-5
# epochs = 200 <-- 상단 블록에서 제어
# batch_size = 1024 <-- 상단 블록에서 제어
# lr = 1e-4 <-- 상단 블록에서 제어
N_eval_states = 2000
CRN_SEED_EU = 12345
Y0_range = None

# ==============================================================================
# ===== (B) 사용자 정의 영역: 정책 네트워크 및 모델 동역학 =====
# ==============================================================================
# (참고: 아래 함수들은 N차원 머튼 모델의 것과 동일합니다.
#        d=1, k=0 설정에 맞게 자동으로 동작합니다.)
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
        states_to_cat = [states_dict['X'], states_dict['TmT']]
        if 'Y' in states_dict and states_dict['Y'] is not None and k > 0:
            states_to_cat.append(states_dict['Y'])
        x_in = torch.cat(states_to_cat, dim=1)
        u = self.net(x_in)
        return torch.clamp(u, -u_cap, u_cap)

def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    X0 = torch.rand((B, DIM_X), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
    Y0 = None
    if DIM_Y > 0:
        Y_min, Y_max = Y0_range
        Y0 = Y_min.unsqueeze(0) + (Y_max - Y_min).unsqueeze(0) * torch.rand((B, DIM_Y), device=device, generator=rng)
    TmT0 = torch.rand((B, 1), device=device, generator=rng) * T
    dt_vec = TmT0 / float(m)
    states = {'X': X0, 'TmT': TmT0}
    if Y0 is not None: states['Y'] = Y0
    return states, dt_vec

def simulate(policy, B, *, train=True, rng=None, initial_states_dict=None, random_draws=None, m_steps=None):
    m_eff = m_steps if m_steps is not None else m
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states, dt = initial_states_dict, initial_states_dict['TmT'] / float(m_eff)
    logX = states['X'].clamp_min(lb_X).log()
    if random_draws is None:
        Z = torch.randn(B, m_eff, d, device=device, generator=rng)
    else:
        Z = random_draws[0]
    for i in range(m_eff):
        current_states = {'X': logX.exp(), 'TmT': states['TmT'] - i * dt}
        u = policy(**current_states)
        drift_term = r + (u * alpha).sum(1, keepdim=True)
        var_term = torch.einsum('bi,bi->b', u @ Sigma, u).view(-1, 1)
        dBX = (torch.einsum('bi,ij->bj', u, torch.linalg.cholesky(Sigma)) * Z[:, i, :]).sum(1, keepdim=True)
        logX = logX + (drift_term - 0.5 * var_term) * dt + dBX * dt.sqrt()
    XT = logX.exp().clamp_min(lb_X)
    return (XT.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)

class WrappedCFPolicy(nn.Module):
    def __init__(self, u_star):
        super().__init__()
        # u_star: (d,) 머튼 상수 벡터 (비율 정책)
        self.register_buffer("u_star", u_star.view(-1))

    def forward(self, **states_dict):
        # 배치 크기에 맞춰 (B, d)로 확장
        B = states_dict["X"].shape[0]
        return self.u_star.unsqueeze(0).expand(B, -1)

def build_closed_form_policy():
    # u_closed_form: (d,) = (1/γ) Σ^{-1} α (위에서 이미 계산됨)
    cf_policy = WrappedCFPolicy(u_closed_form).to(device)
    return cf_policy, None  # ode_solution 없음