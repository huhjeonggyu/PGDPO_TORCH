# tests/sir/user_pgdpo_base.py
# ✨ FIX: 모든 실행 모드에서 노이즈 차원을 일관성 있게 처리하도록 수정

import os
import torch
import torch.nn as nn

# --- 모델별 설정 및 환경변수 오버라이드 블록 ---
d = 3
k = 0
epochs = 250
batch_size = 1024
lr = 1e-4
seed = 42

d = int(os.getenv("PGDPO_D", d))
epochs = int(os.getenv("PGDPO_EPOCHS", epochs))
batch_size = int(os.getenv("PGDPO_BATCH_SIZE", batch_size))
lr = float(os.getenv("PGDPO_LR", lr))
seed = int(os.getenv("PGDPO_SEED", seed))
# --- 블록 끝 ---


# ==============================================================================
# ===== (A) 사용자 정의 영역: 모델 차원, 파라미터, 하이퍼파라미터 =====
# ==============================================================================

# --------------------------- Model Dimensions ---------------------------
N_regions = d
DIM_X = 3 * N_regions
DIM_Y = k
DIM_U = N_regions

# --------------------------- Config ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------- Epidemic & Utility Parameters (방역 강화 버전) ---------------------------
beta = 0.25
gamma = 0.1
A_cost = 25.0
B_cost = torch.full((N_regions,), 0.1, device=device)
I_max = 0.4 * N_regions

M = torch.zeros(N_regions, N_regions, device=device)
mobility_rate = 0.05
for i in range(N_regions):
    M[i, (i - 1 + N_regions) % N_regions] = mobility_rate
    M[i, (i + 1) % N_regions] = mobility_rate

# --------------------------- Simulation & Training Parameters ---------------------------
T = 2.0
m = 50
X0_range = (0.8, 1.2)
u_cap = 10.0
lb_X  = 1e-5
N_eval_states = 100
CRN_SEED_EU = 12345
Y0_range = None

def calculate_rt(S: torch.Tensor) -> torch.Tensor:
    return beta * S / gamma

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
        x_in = torch.cat([states_dict['X'], states_dict['TmT']], dim=1)
        u = self.net(x_in)
        return torch.clamp(u, 0.0, u_cap)

def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    X0 = torch.zeros(B, DIM_X, device=device)
    for i in range(N_regions):
        X0[:, i*3] = torch.rand((B,), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
        X0[:, i*3+1] = torch.rand((B,), device=device, generator=rng) * 0.05
    
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
    
    # ✨ [수정] --run, --residual 모드에서 d차원 노이즈를 받아도 DIM_X (3*d) 차원으로 확장
    if random_draws is not None:
        noise_source = random_draws[0]
        # 입력된 노이즈의 차원이 d와 같다면 (run, residual 모드), 3*d로 확장
        if noise_source.shape[-1] == d:
            Z = torch.zeros(B, m_eff, DIM_X, device=device, dtype=noise_source.dtype)
            # 동일한 노이즈를 S, I, R에 각각 적용 (간소화된 가정)
            Z[:, :, 0::3] = noise_source / (3**0.5)
            Z[:, :, 1::3] = noise_source / (3**0.5)
            Z[:, :, 2::3] = noise_source / (3**0.5)
        else: # 입력된 노이즈가 이미 DIM_X 차원이라면 (base 모드)
            Z = noise_source
    else: # random_draws가 없으면 (base 모드) 직접 DIM_X 차원 노이즈 생성
        Z = torch.randn(B, m_eff, DIM_X, device=device, generator=rng)

    total_cost = torch.zeros(B, 1, device=device)
    
    for i in range(m_eff):
        current_states = {'X': X, 'TmT': states['TmT'] - i * dt}
        u = policy(**current_states)

        S = X[:, 0::3]; I = X[:, 1::3]; R = X[:, 2::3]

        infection_cost = A_cost * I.sum(dim=1, keepdim=True)
        control_cost = 0.5 * (u * u * B_cost.view(1, -1)).sum(dim=1, keepdim=True)
        total_cost += (infection_cost + control_cost) * dt

        infection_term = beta * S * I
        
        S_inflow = S @ M.T; S_outflow = S * M.sum(dim=0, keepdim=True)
        I_inflow = I @ M.T; I_outflow = I * M.sum(dim=0, keepdim=True)
        R_inflow = R @ M.T; R_outflow = R * M.sum(dim=0, keepdim=True)
        
        dX = torch.zeros_like(X)
        dX[:, 0::3] = -infection_term - u * S + (S_inflow - S_outflow)
        dX[:, 1::3] = infection_term - gamma * I + (I_inflow - I_outflow)
        dX[:, 2::3] = gamma * I + u * S + (R_inflow - R_outflow)
        
        noise = Z[:, i, :] * dt.sqrt() # 이제 Z는 항상 [B, m_eff, 3*d] 크기
        X = X + dX * dt + noise
        X = X.clamp_min(lb_X)

    return -total_cost

def build_closed_form_policy():
    return None, {"note": "no true closed-form; sir model"}

def get_traj_schema():
    x_labels = []
    for i in range(N_regions):
        x_labels.extend([f"S_{i+1}", f"I_{i+1}", f"R_{i+1}"])

    return {
        "roles": {
            "X": {"dim": DIM_X, "labels": x_labels},
            "U": {"dim": DIM_U, "labels": [f"Vaccine_Rate_{i+1}" for i in range(N_regions)]},
            "Rt": {"dim": 1, "labels": ["Effective_Rt"]},
            "Costate_I": {"dim": 1, "labels": ["Shadow_Price_of_Infection"]},
        },
        "views": [
            {
                "name": "Rt_and_Infection_Trajectory",
                "block": ["X", "Rt"], 
                "mode": "custom_sir_rt_plot",
                "ylabel": "Value",
                "h_lines": [{"y": 1.0, "label": "R_t = 1 Threshold", "color": "red", "ls": "--"}]
            },
            {
                "name": "Costate_vs_Policy",
                "block": ["U", "Costate_I"],
                "mode": "dual_axis",
                "ylabels": ["Control (Vaccination)", "Shadow Price (Costate)"]
            },
            {
                "name": "Infected_Population_vs_Capacity",
                "block": "X",
                "mode": "custom_sir_i_plot",
                "ylabel": "Infected Population",
                "h_lines": [{"y": I_max, "label": f"ICU Capacity ({I_max:.2f})", "color": "purple", "ls": "-."}]
            }
        ]
    }