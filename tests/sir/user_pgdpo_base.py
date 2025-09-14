# user_pgdpo_base.py 
# ✨ FIX: 지역 간 이동(Mobility)이 포함된 전염병(SIR) 모델로 수정
import os
import torch
import torch.nn as nn

# --- 모델별 설정 및 환경변수 오버라이드 블록 ---

# 1. 모델 고유의 기본값을 설정합니다.
# d: 지역(region)의 수. S,I,R 3개 구획이 있으므로 총 상태 차원은 3*d가 됩니다.
d = 3
k = 0  # 이 모델은 외부 요인(k)이 없습니다.
epochs = 250
batch_size = 1024
lr = 1e-4
seed = 42

# 2. 변경 가능한 하이퍼파라미터만 환경변수로부터 덮어씁니다.
d = int(os.getenv("PGDPO_D", d))
epochs = int(os.getenv("PGDPO_EPOCHS", epochs))
batch_size = int(os.getenv("PGDPO_BATCH_SIZE", batch_size))
lr = float(os.getenv("PGDPO_LR", lr))
seed = int(os.getenv("PGDPO_SEED", seed))
# k는 이 모델의 정의에 따라 0으로 유지됩니다.

# --- 블록 끝 ---


# ==============================================================================
# ===== (A) 사용자 정의 영역: 모델 차원, 파라미터, 하이퍼파라미터 =====
# ==============================================================================

# --------------------------- Model Dimensions ---------------------------
N_regions = d
DIM_X = 3 * N_regions  # 상태 X = [S1,I1,R1, S2,I2,R2, ...]
DIM_Y = k
DIM_U = N_regions      # 제어 u = [u1, u2, ...] (지역별 백신 접종률)

# --------------------------- Config ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------- Epidemic & Utility Parameters ---------------------------
beta = 0.25      # 감염률 (Infection rate)
gamma = 0.1      # 회복률 (Recovery rate)
A_cost = 5.0     # 감염자 1명당 발생하는 사회적 비용 (기존 1.0에서 5.0으로 상향)
B_cost = torch.full((N_regions,), 0.5, device=device) # 백신 접종 비용

# ✨ FIX: 지역 간 이동 행렬 (Mobility Matrix, M) 정의
# M[i, j]는 j지역에서 i지역으로의 인구 이동률을 의미합니다.
# 예시: 인접 지역으로만 5%씩 이동하는 순환 모델
M = torch.zeros(N_regions, N_regions, device=device)
mobility_rate = 0.05
for i in range(N_regions):
    M[i, (i - 1 + N_regions) % N_regions] = mobility_rate
    M[i, (i + 1) % N_regions] = mobility_rate

# --------------------------- Simulation & Training Parameters ---------------------------
T = 2.0
m = 50
X0_range = (0.8, 1.2)  # 초기 S인구는 1.0 근처에서 시작
u_cap = 1.0
lb_X  = 1e-5
N_eval_states = 2000
CRN_SEED_EU = 12345
Y0_range = None # Y 없음

# ==============================================================================
# ===== (B) 사용자 정의 영역: 정책 네트워크 및 모델 동역학 =====
# ==============================================================================
class DirectPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        state_dim = DIM_X + DIM_Y + 1
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, DIM_U)
        )

    def forward(self, **states_dict):
        # Y가 없으므로 X와 TmT만 사용
        x_in = torch.cat([states_dict['X'], states_dict['TmT']], dim=1)
        u = self.net(x_in)
        # 백신 접종률은 [0, u_cap] 사이의 값을 가집니다.
        return torch.clamp(u, 0.0, u_cap)

def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    X0 = torch.zeros(B, DIM_X, device=device)
    # 초기에는 대부분 Susceptible(S), 일부만 Infected(I)
    for i in range(N_regions):
        X0[:, i*3] = torch.rand((B,), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0] # S
        X0[:, i*3+1] = torch.rand((B,), device=device, generator=rng) * 0.05 # I
    
    TmT0 = torch.rand((B, 1), device=device, generator=rng) * T
    dt_vec = TmT0 / float(m)
    states = {'X': X0, 'Y': None, 'TmT': TmT0}
    return states, dt_vec

def simulate(policy, B, *, train=True, rng=None, initial_states_dict=None, random_draws=None, m_steps=None):
    m_eff = m_steps if m_steps is not None else m
    
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states, dt = initial_states_dict, initial_states_dict['TmT'] / float(m_eff)
        
    X = states['X'].clamp_min(lb_X)
    
    Z = None
    if random_draws is not None:
        provided_noise = random_draws[0]
        if provided_noise.shape[-1] == DIM_X:
            Z = provided_noise
            
    if Z is None:
        Z = torch.randn(B, m_eff, DIM_X, device=device, generator=rng)

    total_cost = torch.zeros(B, 1, device=device)
    
    for i in range(m_eff):
        current_states = {'X': X, 'TmT': states['TmT'] - i * dt}
        
        u = policy(**current_states)

        S = X[:, 0::3]
        I = X[:, 1::3]
        R = X[:, 2::3]

        infection_cost = A_cost * I.sum(dim=1, keepdim=True)
        control_cost = 0.5 * (u * u * B_cost.view(1, -1)).sum(dim=1, keepdim=True)
        total_cost += (infection_cost + control_cost) * dt

        infection_term = beta * S * I
        
        S_inflow = S @ M.T 
        S_outflow = S * M.sum(dim=0, keepdim=True)
        I_inflow = I @ M.T
        I_outflow = I * M.sum(dim=0, keepdim=True)
        R_inflow = R @ M.T
        R_outflow = R * M.sum(dim=0, keepdim=True)
        
        dS = -infection_term - u * S + (S_inflow - S_outflow)
        dI = infection_term - gamma * I + (I_inflow - I_outflow)
        dR = gamma * I + u * S + (R_inflow - R_outflow)
        
        noise_S = Z[:, i, 0::3] * dt.sqrt()
        noise_I = Z[:, i, 1::3] * dt.sqrt()
        noise_R = Z[:, i, 2::3] * dt.sqrt()

        # ✨ FIX: Replace in-place operations (+=) with out-of-place assignments (= ... + ...)
        # This preserves the computation graph for the backward pass.
        S_new = S + dS * dt + noise_S
        I_new = I + dI * dt + noise_I
        R_new = R + dR * dt + noise_R
        
        # Combine the new S, I, R parts back into a single state tensor X
        X_new = torch.zeros_like(X)
        X_new[:, 0::3] = S_new
        X_new[:, 1::3] = I_new
        X_new[:, 2::3] = R_new

        X = X_new.clamp_min(lb_X)

    return -total_cost

def build_closed_form_policy():
    # 이 모델은 해석적 해가 없습니다.
    return None, {"note": "no true closed-form; sir model"}