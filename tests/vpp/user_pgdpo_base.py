# tests/vpp/user_pgdpo_base.py
import os
import torch
import torch.nn as nn
import numpy as np

# --- 모델별 설정 및 환경변수 오버라이드 블록 ---
d = 10
k = 0
epochs = 500
batch_size = 1024
lr = 1e-4
seed = 42

d = int(os.getenv("PGDPO_D", d))
epochs = int(os.getenv("PGDPO_EPOCHS", epochs))
batch_size = int(os.getenv("PGDPO_BATCH_SIZE", batch_size))
lr = float(os.getenv("PGDPO_LR", lr))
seed = int(os.getenv("PGDPO_SEED", seed))
# --- 블록 끝 ---

# ================== (A) Dimensions & global config ==================
DIM_X, DIM_Y, DIM_U = d, k, d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRN_SEED_EU = 777
N_eval_states = 2048

torch.manual_seed(seed); np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ================== (B) Problem parameters (수정됨) ==================
T = 1.0
m = 40
dt_const = T / m

# ✨ 1. (신규) 상태 추종 페널티 가중치
# 각 배터리의 SoC가 목표치(x_target)에서 벗어날 때 부과되는 페널티의 강도
state_penalty_weight = 5.0 
x_target = 0.5

# 이종(Heterogeneous) 비용 파라미터 R_i
R_diag = torch.empty(d, device=device).uniform_(0.2, 1.0)
R_matrix = torch.diag(R_diag)

# 변동성 있는 전력 가격 모델 P(t)
def price_fn(t_tensor: torch.Tensor) -> torch.Tensor:
    price = 25 * torch.sin(2 * torch.pi * t_tensor / T) + \
            15 * torch.sin(4 * torch.pi * t_tensor / T) + 30
    return price.clamp_min(0.1)

# 이종(Heterogeneous) 노이즈 파라미터
vols_W = torch.empty(d, device=device).uniform_(0.3, 0.7)

u_min, u_max = -1000.0, 1000.0

def draw_dW(B: int, dt_val: float):
    scale = torch.sqrt(torch.tensor(dt_val, device=device))
    return torch.randn(B, d, device=device) * vols_W * scale

class DirectPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        state_dim = DIM_X + DIM_Y + 1
        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, DIM_U)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, **states_dict) -> torch.Tensor:
        X   = states_dict["X"]
        TmT = states_dict["TmT"]
        x_in = torch.cat([X, TmT], dim=1)
        u = self.net(x_in)
        return torch.clamp(u, u_min, u_max)

# ================== (E) Simulator (수정됨) ==================
def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    X0 = torch.rand((B, DIM_X), device=device, generator=rng) * 0.6 + 0.2
    TmT0 = torch.rand((B, 1), device=device, generator=rng) * T
    dt_vec = torch.full_like(TmT0, dt_const)
    return {'X': X0, 'TmT': TmT0}, dt_vec

def simulate(policy, B, *, train=True, rng=None, initial_states_dict=None, random_draws=None, m_steps=None):
    """
    ✨ 목표 함수에 '상태 추종 페널티' 추가
    - Reward per step: (Price * sum(u)) - 0.5 * u'Ru - 0.5 * weight * ||x - x_target||^2
    """
    m_eff = m_steps if m_steps is not None else m
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states, dt = initial_states_dict, torch.full_like(initial_states_dict['TmT'], T / float(m_eff))

    X = states['X']
    total_profit = torch.zeros(B, 1, device=device)

    for i in range(m_eff):
        t_current = T - (states['TmT'] - i * dt)
        current_states = {'X': X, 'TmT': states['TmT'] - i * dt}
        u = policy(**current_states)

        # 수익 계산 로직
        price = price_fn(t_current)
        revenue = price * torch.sum(u, dim=1, keepdim=True)
        op_cost = 0.5 * torch.einsum('bi,ij,bj->b', u, R_matrix, u).unsqueeze(1)
        
        # ✨ 2. (신규) 상태 페널티 계산
        # 현재 상태 X가 목표치 x_target에서 벗어난 정도에 따라 페널티 부과
        state_penalty = 0.5 * state_penalty_weight * torch.sum((X - x_target)**2, dim=1, keepdim=True)

        # ✨ 3. (수정) 최종 단계별 보상 계산
        profit_step = revenue - op_cost - state_penalty
        total_profit += profit_step * dt

        # SDE step
        dW = draw_dW(B, float(dt_const))
        X = X - u * dt + dW
        # X = torch.clamp(X, 0.0, 1.0) # 필요시 SoC 제약 추가

    # 프레임워크가 보상(reward)을 최대화하도록 수익(profit)을 직접 반환
    return total_profit

# ================== (F) Closed-form (수정됨) ==================
class AnalyticalMyopicPolicy(nn.Module):
    """
    ✨ 수정된 해석적 해: u_i(t) = P(t) / R_i (Co-state 무시)
    """
    def __init__(self):
        super().__init__()
        # R_diag는 (d,) 형태이므로 (1,d)로 브로드캐스팅 가능하게 만듦
        self.register_buffer("inv_R_diag", 1.0 / R_diag.view(1, -1))

    def forward(self, **states_dict):
        t = T - states_dict['TmT']
        price = price_fn(t)
        # 각 배터리는 자신의 비용(R_i)에만 반응
        u = price * self.inv_R_diag
        return torch.clamp(u, u_min, u_max)

def build_closed_form_policy():
    print("✅ Analytical VPP myopic policy for HETEROGENEOUS units loaded.")
    return AnalyticalMyopicPolicy().to(device), None

def ref_signals_fn(t_np: np.ndarray) -> dict:
    """
    비교를 위해 그래프에 함께 표시할 외부 신호(가격)를 반환합니다.
    """
    # price_fn은 tensor를 입력받으므로 numpy 배열을 tensor로 변환했다가 다시 변환합니다.
    price_t = price_fn(torch.from_numpy(t_np).float().to(device))
    return {"Price": price_t.cpu().numpy()}

R_INFO = {"R": R_matrix}

def get_traj_schema():
    """
    어떤 변수를 어떻게 시각화할지 정의하는 스키마를 반환합니다.
    """
    return {
        "roles": {
            "X": {"dim": int(DIM_X), "labels": [f"SoC_{i+1}" for i in range(int(DIM_X))]},
            "U": {"dim": int(DIM_U), "labels": [f"u_{i+1}"   for i in range(int(DIM_U))]},
        },
        "views": [
            # ✨ 새로운 그래프 뷰 정의
            {"name": "Arbitrage_Strategy", "mode": "tracking_vpp", "ylabel": "Price / Power"},
            {"name": "U_Rnorm", "mode": "u_rnorm", "block": "U", "ylabel": "||u||_R"},
        ],
        "sampling": {"Bmax": 5}
    }