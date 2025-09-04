# user_pgdpo_base.py for N-dimensional Merton Problem
# 역할: 다차원 머튼 모델의 모든 사용자 정의 요소를 설정합니다.
#       (1) 모델 차원 및 파라미터, (2) 정책 신경망, (3) 시뮬레이션 동역학

import torch
import torch.nn as nn

# ==============================================================================
# ===== (A) 사용자 정의 영역: 모델 차원, 파라미터, 하이퍼파라미터 =====
# ==============================================================================

# --------------------------- Model Dimensions ---------------------------
# d: 자산 수, k: 팩터 수
d = 5  # 5차원 머튼 문제로 설정
k = 0  # ✨ 머튼 문제의 핵심: 팩터 없음 (k=0)

DIM_X = 1      # 자산 X는 부(wealth)를 의미하므로 1차원
DIM_Y = k      # 외생 변수 Y 없음 (0차원)
DIM_U = d      # 제어 u는 d개 자산에 대한 투자 비율

# --------------------------- Config ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42

# --------------------------- Market & Utility Parameters ---------------------------
r = 0.03
gamma = 2.0

# --------------------------- Simulation Parameters ---------------------------
T = 1.5
m = 20

# --------------------------- State & Control Ranges ---------------------------
X0_range = (0.1, 3.0)
u_cap = 10.
lb_X  = 1e-5

# --------------------------- Training Hyperparameters ---------------------------
epochs     = 200
batch_size = 1024
lr         = 1e-4

# --------------------------- Evaluation Parameters ---------------------------
N_eval_states = 2000
CRN_SEED_EU   = 12345

# --------------------------- Parameter Generation ---------------------------
def _generate_merton_params(d, r, dev, seed=None):
    """다차원 머튼 문제의 파라미터 (mu, Sigma)를 생성합니다."""
    if seed is not None:
        torch.manual_seed(seed)

    # 상수 형태의 기대수익률 벡터 (mu)와 공분산 행렬 (Sigma) 생성
    # alpha는 리스크 프리미엄(mu - r)을 의미
    alpha = torch.empty(d).uniform_(0.05, 0.15) 

    # Positive-definite 공분산 행렬 Sigma 생성
    L = torch.randn(d, d) * 0.5
    Sigma = L @ L.T + torch.diag(torch.empty(d).uniform_(0.02, 0.05)) # 대각 성분에 값을 더해 안정성 확보
    Sigma_inv = torch.linalg.inv(Sigma)

    params = {'alpha': alpha, 'Sigma': Sigma, 'Sigma_inv': Sigma_inv}
    
    # 분석적 해 계산을 위한 파라미터 추가
    # u* = (1/gamma) * Sigma_inv * alpha
    params['u_closed_form'] = (1.0 / gamma) * (params['Sigma_inv'] @ params['alpha'])

    return {k: v.to(dev) for k, v in params.items()}

# 파라미터 생성 실행
params = _generate_merton_params(d, r, device, seed)

# 자주 사용하는 파라미터를 전역 변수로 추출
alpha, Sigma, Sigma_inv = params['alpha'], params['Sigma'], params['Sigma_inv']
u_closed_form = params['u_closed_form']

# Y가 없으므로 Y0_range는 None
Y0_range = None

# ==============================================================================
# ===== (B) 사용자 정의 영역: 정책 네트워크 및 모델 동역학 =====
# ==============================================================================
# 
# 참고: 아래의 DirectPolicy, sample_initial_states, simulate, build_closed_form_policy는
#       Kim-Omberg 예제의 것과 동일합니다. 프레임워크가 k=0인 경우를
#       자동으로 처리하도록 설계되었기 때문에 수정할 필요가 없습니다.
#       (예: Y가 없으면 입력으로 사용하지 않고, 시뮬레이션 시 헤징항을 계산하지 않음)
#
class DirectPolicy(nn.Module):
    """정책 신경망입니다."""
    def __init__(self):
        super().__init__()
        # Y가 없으므로 state_dim에서 DIM_Y가 0이 되어 자동으로 처리됨
        state_dim = DIM_X + DIM_Y + 1
        self.net = nn.Sequential(
            nn.Linear(state_dim, 200), nn.LeakyReLU(),
            nn.Linear(200, 200), nn.LeakyReLU(),
            nn.Linear(200, DIM_U)
        )

    def forward(self, **states_dict):
        states_to_cat = [states_dict['X'], states_dict['TmT']]
        # Y가 없으면 이 부분은 실행되지 않음
        if 'Y' in states_dict and states_dict['Y'] is not None:
            states_to_cat.append(states_dict['Y'])
        x_in = torch.cat(states_to_cat, dim=1)
        u = self.net(x_in)
        return torch.clamp(u, -u_cap, u_cap)

def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    """초기 상태 (X, TmT)를 샘플링합니다."""
    X0 = torch.rand((B, DIM_X), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
    
    # DIM_Y = 0 이므로 Y0는 None
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
    """머튼 문제의 SDE를 시뮬레이션합니다."""
    m_eff = m_steps if m_steps is not None else m
    
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states, dt = initial_states_dict, initial_states_dict['TmT'] / float(m_eff)
        
    logX = states['X'].clamp_min(lb_X).log()
    
    if random_draws is None:
        # Y가 없으므로 d차원 브라운 운동만 필요
        Z = torch.randn(B, m_eff, d, device=device, generator=rng)
    else:
        Z = random_draws[0] # ZY는 없음

    for i in range(m_eff):
        current_states = {'X': logX.exp(), 'TmT': states['TmT'] - i * dt}
        u = policy(**current_states)
        
        # Y가 없으므로 Merton 동역학을 따름
        drift_term = r + (u * alpha).sum(1, keepdim=True)
        var_term = torch.einsum('bi,bi->b', u @ Sigma, u).view(-1, 1)

        dBX = (torch.einsum('bi,ij->bj', u, torch.linalg.cholesky(Sigma)) * Z[:, i, :]).sum(1, keepdim=True)
        logX = logX + (drift_term - 0.5 * var_term) * dt + dBX * dt.sqrt()

    XT = logX.exp().clamp_min(lb_X)
    return (XT.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)


class WrappedCFPolicy(nn.Module):
    def __init__(self, u_star: torch.Tensor):
        super().__init__()
        # u_star: (d,) — 학습/이동과 무관하므로 buffer로 저장
        self.register_buffer("u_star", u_star.view(-1))  # (d,)

    def forward(self, **states_dict):
        # 배치 크기에 맞춰 (B, d)로 확장해 반환 (상수 비율 정책)
        B = states_dict["X"].shape[0]
        return self.u_star.unsqueeze(0).expand(B, -1)


def build_closed_form_policy():
    # u* = (1/γ) Σ^{-1} α  (모양을 (d,)로 보장)
    u_star = ((1.0 / gamma) * (Sigma_inv @ alpha)).view(-1).to(device)
    cf_policy = WrappedCFPolicy(u_star).to(device)
    return cf_policy, None  # ODE 솔루션 없음