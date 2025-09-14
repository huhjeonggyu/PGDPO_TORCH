# tests/harvest/user_pgdpo_base.py
# Harvesting (§5.4) — PG-DPO baseline (rng 호환 simulate, myopic을 참조정책으로 노출)
import os
import math
import torch
import torch.nn as nn

# --- 모델별 설정 및 환경변수 오버라이드 블록 ---

# 1. 모델 고유의 기본값을 설정합니다.
d = 5
k = 0  # ✨ Harvest 모델은 k=0 으로 고정되어야 합니다.
epochs = 200
batch_size = 256
lr = 1e-4
seed = 2025

# 2. 변경 가능한 파라미터만 환경변수로부터 덮어씁니다.
d = int(os.getenv("PGDPO_D", d))
epochs = int(os.getenv("PGDPO_EPOCHS", epochs))
batch_size = int(os.getenv("PGDPO_BATCH_SIZE", batch_size))
lr = float(os.getenv("PGDPO_LR", lr))
seed = int(os.getenv("PGDPO_SEED", seed))
# k는 이 모델의 정의에 따라 0으로 유지됩니다.

# --- 블록 끝 ---


# --------------------------- Device / seed ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)

# --------------------------- Horizon / steps -------------------------
T = 2.0
m = 50  # steps
CRN_SEED_EU = 24680  # common random numbers for eval

# >>> 코어 RMSE 헬퍼가 기대하는 평가 샘플 수
N_eval_states = 100

# --------------------------- Dimensions ------------------------------
# ✨ FIX: 이 섹션에서 모든 차원 변수를 순서에 맞게 정의합니다.
N_species = d
du = N_species  # control dim: harvesting rate per species

# 프레임워크 호환성을 위한 핵심 차원 변수
DIM_X = d
DIM_Y = k
DIM_U = du

# --------------------------- Problem params --------------------------

def _generate_harvesting_params_structured(d, device):
    """
    구조화된 생태계 파라미터를 생성합니다. (먹이 사슬 + 환경 충격)
    """
    # 1. 상호작용 행렬 (Alpha): 먹이 사슬 구조
    #    - A -> B -> C (A는 B를 먹고, B는 C를 먹는다)
    #    - Alpha[i, j]는 j가 i에게 미치는 영향
    Alpha = torch.zeros(d, d, device=device)
    for i in range(d - 1):
        # 포식자(i)는 피식자(i+1)의 성장을 방해 (-)
        Alpha[i, i + 1] = -0.05
        # 피식자(i+1)는 포식자(i)의 성장을 도움 (+)
        Alpha[i + 1, i] = 0.03
    
    # 2. 확률론적 상관관계 (Noise Correlation):
    #    - 모든 종이 0.3 정도의 약한 양의 상관관계를 가짐 (공통 환경 요인)
    noise_corr = torch.full((d, d), 0.3, device=device)
    noise_corr.fill_diagonal_(1.0)
    # Cholesky 분해를 위해 Positive-definite 행렬로 보정
    noise_corr += 1e-5 * torch.eye(d, device=device)
    cholesky_L = torch.linalg.cholesky(noise_corr)

    # 나머지 파라미터는 기존과 유사하게 생성
    price = torch.empty(d, device=device).uniform_(0.8, 1.2)
    R_diag = torch.empty(d, device=device).uniform_(0.1, 1.0)
    Qx_diag = torch.empty(d, device=device).uniform_(0.05, 0.2)
    r_g = torch.empty(d, device=device).uniform_(0.05, 0.15)
    K = torch.empty(d, device=device).uniform_(1.0, 2.0)
    sigma_x = torch.empty(d, device=device).uniform_(0.03, 0.08)

    return {
        "price": price, "R_diag": R_diag, "Qx_diag": Qx_diag,
        "r_g": r_g, "K": K, "Alpha": Alpha, "sigma_x": sigma_x,
        "cholesky_L": cholesky_L
    }

# 구조화된 파라미터 생성 함수 호출
params = _generate_harvesting_params_structured(d, device)

# 자주 사용하는 변수들을 전역으로 할당
price = params["price"]
R_diag = params["R_diag"]
R_mat  = torch.diag(R_diag)
R_inv  = torch.diag(1.0 / R_diag.clamp_min(1e-8))
R_inv_diag = 1.0 / R_diag.clamp_min(1e-8)
Qx_diag = params["Qx_diag"]
Qx = torch.diag(Qx_diag)
r_g = params["r_g"]
K = params["K"]
Alpha = params["Alpha"]
sigma_x = params["sigma_x"]
cholesky_L = params["cholesky_L"]

# Bounds
u_cap = 2.0
lb_X  = 0.0
X_ref = 0.5 * K

# --------------------------- RNG helpers -----------------------------
def make_generator(seed_val: int | None):
    g = torch.Generator(device=device)
    if seed_val is not None:
        g.manual_seed(int(seed_val))
    return g

def sample_TmT(B: int, *, rng: torch.Generator | None = None):
    return torch.rand((B,1), device=device, generator=rng).clamp_min(1e-6) * T

def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    # start near K/2 with some variability, strictly positive
    X0 = (0.5 * K + 0.2 * K * torch.randn(B, d, device=device, generator=rng)).clamp_min(1e-3)
    TmT0 = sample_TmT(B, rng=rng)
    return {"X": X0, "TmT": TmT0}, None

# --------------------------- Policy (residual) -----------------------
class DirectPolicy(nn.Module):
    """
    Residual on top of myopic: u_my = R^{-1} diag(X) price.
    Network learns bounded residual; final u is clamped to [0, u_cap].
    Input: [X, TmT] -> residual
    """
    def __init__(self):
        super().__init__()
        h = 256
        self.net = nn.Sequential(
            nn.Linear(d + 1, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, du),
        )
        # small init around 0 to not distort myopic too much initially
        for m_ in self.net.modules():
            if isinstance(m_, nn.Linear):
                nn.init.kaiming_uniform_(m_.weight, a=math.sqrt(5))
                if m_.bias is not None:
                    nn.init.zeros_(m_.bias)

    def forward(self, **states_dict):
        X   = states_dict["X"]
        TmT = states_dict["TmT"]
        z   = torch.cat([X, TmT], dim=1)
        residual = self.net(z)

        # myopic baseline: R^{-1} diag(X) price  =  (X * price) ⊙ R_inv_diag
        u_my = (X * price.view(1, -1)) * R_inv_diag.view(1, -1)
        u    = u_my + residual
        return torch.clamp(u, 0.0, u_cap)

# --------------------------- Myopic policy (참조정책) ----------------
class MyopicPolicy(nn.Module):
    """u_my = R^{-1} diag(X) price, clamped to [0, u_cap]."""
    def __init__(self):
        super().__init__()
    def forward(self, **states_dict):
        X = states_dict["X"]
        u_my = (X * price.view(1, -1)) * R_inv_diag.view(1, -1)
        return torch.clamp(u_my, 0.0, u_cap)

# --------------------------- Simulator --------------------------------
def simulate(policy: nn.Module,
             B: int,
             *,
             train: bool = True,
             rng: torch.Generator | None = None,
             initial_states_dict: dict | None = None,
             random_draws = None,
             m_steps: int | None = None):
    """
    Returns utility per path, shape (B,1): sum_t [ harvest - 0.5 u^T R u - 0.5 (X-Xref)^T Q (X-Xref) ] * dt
    Compatible with core: accepts rng=Generator and typical kwargs.
    """
    # turn grad on if training OR external X requires grad (for costates)
    need_grad = bool(train)
    if initial_states_dict is not None and "X" in initial_states_dict:
        need_grad = need_grad or bool(getattr(initial_states_dict["X"], "requires_grad", False))

    with torch.set_grad_enabled(need_grad):
        steps = int(m_steps or m)

        if initial_states_dict is None:
            g0 = rng if isinstance(rng, torch.Generator) else make_generator(seed if train else CRN_SEED_EU)
            states, _ = sample_initial_states(B, rng=g0)
            X   = states["X"]
            TmT = states["TmT"]
        else:
            # Use as-is to keep autograd path intact
            X   = initial_states_dict["X"]
            TmT = initial_states_dict["TmT"]
            if X.device.type != device.type or TmT.device.type != device.type:
                raise RuntimeError(f"Initial states are on the wrong device type. Expected '{device.type}', but got X:'{X.device.type}' and TmT:'{TmT.device.type}'")

        # per-sample dt via time-to-go (keeps per-sample horizons)
        dt = (TmT / float(steps)).view(-1, 1)

        # ✨ FIX: 상관관계를 반영하도록 노이즈 생성 방식 변경
        if random_draws is not None:
            uncorrelated_Z = random_draws[0] if isinstance(random_draws, tuple) else random_draws
            uncorrelated_Z = uncorrelated_Z.to(device)
        else:
            g = rng if isinstance(rng, torch.Generator) else make_generator(seed if train else CRN_SEED_EU)
            uncorrelated_Z = torch.randn(B, steps, d, device=device, generator=g)

        # Cholesky 행렬을 곱해 상관관계를 갖는 노이즈(ZX) 생성
        ZX = torch.einsum('bmd,dn->bmn', uncorrelated_Z, cholesky_L)

        U_acc = torch.zeros((B,1), device=device)

        for i in range(steps):
            tau = TmT - i * dt
            
            # ✨ FIX: 누락되었던 정책 호출 코드를 여기에 추가합니다.
            u = policy(X=X, TmT=tau)  # [B,du], already clamped [0,u_cap]

            # reward pieces
            harvest = ((u * X) * price.view(1, -1)).sum(dim=1, keepdim=True)
            quad_u  = 0.5 * torch.einsum('bi,ij,bj->b', u, R_mat, u).unsqueeze(1)
            x_err   = X - X_ref.view(1, -1)
            quad_x  = 0.5 * torch.einsum('bi,ij,bj->b', x_err, Qx, x_err).unsqueeze(1)
            reward  = harvest - quad_u - quad_x
            U_acc   = U_acc + reward * dt

            # dynamics: logistic + interactions - harvest, plus multiplicative noise
            growth = r_g.view(1, -1) * X * (1.0 - X / K.view(1, -1))
            inter  = (X @ Alpha.T) * X
            drift  = growth + inter - u * X

            # ✨ FIX: 상관관계가 적용된 노이즈(ZX) 사용
            dW = ZX[:, i, :] * torch.sqrt(dt)
            X  = (X + drift * dt + sigma_x.view(1, -1) * X * dW).clamp(lb_X, None)

        return U_acc

# --------------------------- "Closed-form" 대용 ----------------------
def build_closed_form_policy():
    """
    하베스팅에는 진짜 closed-form이 없으므로,
    코어의 비교 함수와 호환되도록 myopic 정책을 '참조 정책'으로 반환.
    """
    cf = MyopicPolicy()
    meta = {"note": "no true closed-form; using myopic as reference"}
    return cf, meta

# dummy vars for core compatibility (일부 러너가 import함)
gamma = 1.0
r = 0.0

# ✨ FIX: harvest 모델을 위한 시각화 스키마 정의 함수 (수익률 변수명 변경)
def get_traj_schema():
    """
    harvest 모델의 궤적 시각화 방법을 정의합니다.
    """
    # d의 현재 값에 따라 동적으로 라벨 생성
    x_labels = [f"Biomass_{i+1}" for i in range(d)]
    u_labels = [f"Harvest_Rate_{i+1}" for i in range(d)]

    return {
        "roles": {
            "X": {"dim": d, "labels": x_labels},
            "U": {"dim": du, "labels": u_labels},
            # ✨ 수확 수익(HarvestingIncome)의 역할을 정의합니다.
            "HarvestingIncome": {"dim": 1, "labels": ["Harvesting_Income"]},
        },
        "views": [
            # ✨ 시간에 따른 순간 수확 수익을 그리는 뷰를 추가합니다.
            {
                "name": "Instantaneous_Harvesting_Income",
                "block": "HarvestingIncome",
                "mode": "indices",
                "indices": [0], # 1차원이므로 0번 인덱스만 사용
                "ylabel": "Instantaneous Income"
            },
            # (선택) 기존의 X, U 뷰도 유지할 수 있습니다.
            {
                "name": "Biomass_First_Component",
                "block": "X",
                "mode": "indices",
                "indices": [0], # 첫 번째 종의 바이오매스
                "ylabel": "Biomass X[0]"
            },
            {
                "name": "Harvest_Rate_First_Component",
                "block": "U",
                "mode": "indices",
                "indices": [0], # 첫 번째 종의 수확률
                "ylabel": "Harvest Rate U[0]"
            }
        ],
        "sampling": {"Bmax": 5}
    }