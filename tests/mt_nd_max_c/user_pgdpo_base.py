# user_pgpdo_base.py — Multi-asset Merton with relative consumption cap C <= alpha_rel * X
# 구조는 tests/mt_nd_short와 동일 계열이되, 소비를 포함합니다.
# - 정책: 포트폴리오 u(>=0, 합<=L_cap) + 소비 C(t,x) = alpha_rel * x * sigmoid(v)
# - 시뮬레이터: d log X = ( r + u·alpha - 0.5 u^T Σ u - C/X ) dt + (u^T Σ^{1/2} dW)
# - 목적함수: ∑ e^{-rho t_k} U(C_k) Δt + kappa * e^{-rho T} U(X_T)  (CRRA)
#
# 코어 러너(core/pgdpo_*.py)와 호환을 위해 다음 심볼/함수들을 제공합니다:
#   device, T, m, d, k, DIM_X, DIM_Y, DIM_U, epochs, batch_size, lr, CRN_SEED_EU
#   sample_initial_states(B, rng), simulate(policy, B, ...)
#   DirectPolicy (forward -> u), WrappedCFPolicy, build_closed_form_policy()
#
# 주의: 파일명은 러너가 import하는 user_pgdpo_base.py와 일치해야 합니다.
# 본 샘플은 요청명(user_pgpdo_base.py)에 맞춰 생성했지만, 실제 사용 시 파일명을 user_pgdpo_base.py로 변경하세요.

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- 기본 설정 ---------------------------
# 자산/팩터 차원 및 학습 하이퍼파라미터(환경변수로 덮어쓰기 가능)
d = int(os.getenv("PGDPO_D", 5))
k = int(os.getenv("PGDPO_K", 0))   # 본 베이스는 외생상태 미사용
epochs = int(os.getenv("PGDPO_EPOCHS", 200))
batch_size = int(os.getenv("PGDPO_BS", 1024))
lr = float(os.getenv("PGDPO_LR", 1e-4))
seed = int(os.getenv("PGDPO_SEED", 42))

# 시간 격자
T = float(os.getenv("PGDPO_T", 1.5))
m = int(os.getenv("PGDPO_M", 20))

# 디바이스
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 상태/제어 차원
DIM_X = 1     # wealth X
DIM_Y = k     # 외생 Y (미사용)
DIM_U = d + 1 # risky weights

# 평가/난수
N_eval_states = int(os.getenv("PGDPO_EVAL_STATES", 100))
CRN_SEED_EU   = int(os.getenv("PGDPO_CRN", 12345))

# --------------------------- 경제 파라미터 ---------------------------
gamma = float(os.getenv("PGDPO_GAMMA", 2.0))         # CRRA
rho   = float(os.getenv("PGDPO_RHO", 0.10))          # 시간할인
r     = float(os.getenv("PGDPO_RF", 0.03))           # 무위험
kappa = float(os.getenv("PGDPO_KAPPA", 1.0))         # 유산항 가중치

# 소비 상대상한 계수: C <= alpha_rel * X
alpha_rel = float(os.getenv("PGDPO_C_ALPHA", 0.30))

# 포트폴리오 제약: u >= 0, sum(u) <= L_cap  (무차입)
L_cap = float(os.getenv("PGDPO_LCAP", 1.0))

# 초기부/수치 안정
X0_range = (0.1, 3.0)
lb_X = 1e-6

# --------------------------- 파라미터 생성 ---------------------------
@torch.no_grad()
def _nearest_spd_correlation(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    C = 0.5 * (C + C.T)
    evals, evecs = torch.linalg.eigh(C)
    evals = torch.clamp(evals, min=eps)
    C_spd = (evecs @ torch.diag(evals) @ evecs.T)
    d = torch.diag(C_spd).clamp_min(eps).sqrt()
    Dinv = torch.diag(1.0 / d)
    C_corr = Dinv @ C_spd @ Dinv
    return 0.5 * (C_corr + C_corr.T)

@torch.no_grad()
def _generate_mu_sigma_simple(
    d: int,
    dev,
    seed: int | None = None,
    *,
    vol_range=(0.18, 0.28),
    avg_corr=0.55,
    jitter=0.06,
    alpha_range=(0.03, 0.15),
    lam_factor=0.02,
):
    if seed is not None: torch.manual_seed(seed)
    # (1) Sigma
    sigma = torch.empty(d, device=dev).uniform_(*vol_range)
    Psi = torch.full((d, d), float(avg_corr), device=dev)
    Psi.fill_diagonal_(1.0)
    if jitter and jitter > 0:
        N = torch.randn(d, d, device=dev) * jitter
        Psi = _nearest_spd_correlation(Psi + 0.5 * (N + N.T))
    Sigma = torch.diag(sigma) @ Psi @ torch.diag(sigma)
    lam = lam_factor * Sigma.diag().mean().item()
    Sigma = Sigma + lam * torch.eye(d, device=dev)
    # (2) alpha = mu - r
    alpha = torch.empty(d, device=dev).uniform_(*alpha_range)
    Sigma_inv = torch.linalg.inv(Sigma)
    return {"alpha": alpha, "Sigma": Sigma, "Sigma_inv": Sigma_inv}

# 파라미터 구체화
params = _generate_mu_sigma_simple(d, device, seed=seed)
alpha, Sigma, Sigma_inv = params["alpha"].to(device), params["Sigma"].to(device), params["Sigma_inv"].to(device)
chol_S = torch.linalg.cholesky(Sigma)

# 닫힌형(무제약) 기준 u*: (1/gamma) Σ^{-1} α
u_closed_form_unc = (1.0 / gamma) * (Sigma_inv @ alpha)

# --------------------------- 유틸 ---------------------------
def make_generator(seed_local: int | None = None) -> torch.Generator:
    g = torch.Generator(device=device)
    if seed_local is not None:
        g.manual_seed(int(seed_local))
    return g

# Condat(2016) 기반: v를 {u>=0, sum(u)<=mass}에 대한 근방(유클리드) 투영
@torch.no_grad()
def _proj_simplex_leq(v: torch.Tensor, mass: float) -> torch.Tensor:
    vpos = v.clamp_min(0.0)
    s = float(vpos.sum().item())
    if s <= mass:  # 이미 만족
        return vpos
    # equality simplex로 투영
    u_sorted = torch.sort(vpos, descending=True).values
    cssv = torch.cumsum(u_sorted, dim=0) - mass
    j = torch.arange(1, v.numel()+1, device=v.device, dtype=v.dtype)
    cond = u_sorted > (cssv / j)
    if cond.any():
        rho_idx = int(torch.nonzero(cond, as_tuple=False)[-1].item())
        theta = cssv[rho_idx] / float(rho_idx + 1)
    else:
        theta = cssv[-1] / float(v.numel())
    return (vpos - theta).clamp_min(0.0)

# --------------------------- 정책 ---------------------------
class DirectPolicy(nn.Module):
    """
    포트폴리오 + 소비 신경망(하나의 모듈로 통합).
    - forward(**states)는 u와 C가 결합된 (B, d+1) 텐서를 반환합니다.
    """
    def __init__(self):
        super().__init__()
        state_dim = DIM_X + DIM_Y + 1  # X, (Y), TmT
        hid = 200

        # 이제 하나의 네트워크가 투자와 소비 모두에 대한 출력을 생성합니다.
        self.net = nn.Sequential(
            nn.Linear(state_dim, hid), nn.LeakyReLU(),
            nn.Linear(hid, hid), nn.LeakyReLU(),
            nn.Linear(hid, DIM_U) # 출력 차원을 d+1로 변경
        )

        # 마지막 제약 매핑에 쓰는 softplus beta (부드러움 조절)
        self.softplus_beta = 2.0

    def _concat_states(self, states_dict: dict) -> torch.Tensor:
        parts = [states_dict["X"], states_dict["TmT"]]
        if "Y" in states_dict and states_dict["Y"] is not None:
            parts.insert(1, states_dict["Y"])
        return torch.cat(parts, dim=1)

    def forward(self, **states_dict):
        z = self._concat_states(states_dict)
        raw_output = self.net(z) # (B, d+1)

        # 1. 투자(u) 파트 분리 및 제약 적용
        u_raw = raw_output[:, :d] # 앞 d개는 투자
        u_pos = F.softplus(u_raw, beta=self.softplus_beta)
        B = u_pos.size(0)
        u = torch.stack([_proj_simplex_leq(u_pos[b], L_cap) for b in range(B)], dim=0)

        # 2. 소비(C) 파트 분리 및 제약 적용
        v = raw_output[:, d:] # 마지막 1개는 소비 로짓
        frac = torch.sigmoid(v)      # (0,1) 사이 값으로 변환
        X = states_dict["X"]
        C = alpha_rel * X * frac     # C <= alpha_rel * X 제약 적용

        # 3. 투자(u)와 소비(C)를 하나의 텐서로 결합하여 반환
        return torch.cat([u, C], dim=1)

class WrappedCFPolicy(nn.Module):
    """
    상수형 기준 투자 정책 u_star와 비례 소비 정책 C를 결합하여 반환합니다.
    """
    def __init__(self, u_star: torch.Tensor, alpha_rel: float, c_frac: float):
        super().__init__()
        self.register_buffer("u_star", u_star.view(-1))  # (d,)
        self.alpha_rel = alpha_rel
        self.c_frac = c_frac

    def forward(self, **states_dict):
        B = states_dict["X"].shape[0]
        
        # 1. 투자 u 생성
        u = self.u_star.unsqueeze(0).expand(B, -1)

        # 2. 소비 C 계산 (MyopicPolicy와 동일한 로직)
        X = states_dict["X"]
        C_prop = self.c_frac * X
        C_cap  = self.alpha_rel * X
        C = torch.minimum(C_prop, C_cap)

        # 3. u와 C를 (B, d+1) 텐서로 결합하여 반환
        return torch.cat([u, C], dim=1)

def build_closed_form_policy():
    """
    평가용 기준 정책(벤치마크)을 반환합니다.
    이제 투자(u)와 소비(C)를 모두 포함합니다.
    """
    # 1. 기준 투자 포트폴리오 계산
    u0 = u_closed_form_unc.clone()
    u_star = _proj_simplex_leq(u0, L_cap)

    # 2. 기준 소비를 위한 상수 정의
    # (user_pgdpo_residual.py의 C_FRAC 기본값과 동일하게 설정하여 일관성 유지)
    C_FRAC_BENCHMARK = 0.50 

    # 3. 수정된 WrappedCFPolicy를 사용하여 정책 객체 생성
    cf = WrappedCFPolicy(u_star.to(device), alpha_rel, C_FRAC_BENCHMARK).to(device)
    
    # meta 정보는 없음 (그대로 None 반환)
    return cf, None

# --------------------------- 초기상태 표본 ---------------------------
@torch.no_grad()
def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    X0 = torch.rand((B, DIM_X), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
    Y0 = None  # k=0
    TmT0 = torch.rand((B, 1), device=device, generator=rng) * T
    dt_vec = TmT0 / float(m)
    states = {"X": X0, "TmT": TmT0}
    if Y0 is not None:
        states["Y"] = Y0
    return states, dt_vec

# --------------------------- 시뮬레이터 ---------------------------
def simulate(
    policy: nn.Module,
    B: int,
    *,
    train: bool = True,
    rng: torch.Generator | None = None,
    initial_states_dict: dict | None = None,
    random_draws: tuple[torch.Tensor, torch.Tensor] | None = None,
    m_steps: int | None = None,
):
    """
    반환: 경로별 목적함수 값 U (B,)
    목적함수: sum e^{-rho t} U(C) dt + kappa e^{-rho T} U(X_T),  U(z)=(z^{1-gamma}-1)/(1-gamma)
    """
    m_eff = int(m_steps) if (m_steps is not None) else m
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states, dt = initial_states_dict, initial_states_dict["TmT"] / float(m_eff)

    if random_draws is None:
        Z = torch.randn(B, m_eff, d, device=device, generator=rng)
    else:
        Z = random_draws[0]

    logX = states["X"].clamp_min(lb_X).log()
    TmT0 = states["TmT"]

    util_cons = torch.zeros((B, 1), device=device)

    for i in range(m_eff):
        dt_i = dt
        t_i = T - (TmT0 - i * dt_i)
        cur_states = {"X": logX.exp(), "TmT": TmT0 - i * dt_i}

        # --- (수정된 부분) ---
        # 정책 호출 한 번으로 u와 C를 모두 받습니다.
        policy_output = policy(**cur_states)  # (B, d+1)
        u = policy_output[:, :d]              # 투자 분리
        C = policy_output[:, d:]              # 소비 분리
        # --- (여기까지 수정) ---

        drift = r + torch.einsum("bi,i->b", u, alpha).view(-1,1)
        var = torch.einsum("bi,ij,bj->b", u, Sigma, u).view(-1,1)
        dBX = (torch.einsum("bi,ij->bj", u, chol_S) * Z[:, i, :]).sum(1, keepdim=True)

        logX = logX + (drift - 0.5 * var - C / logX.exp().clamp_min(lb_X)) * dt_i + dBX * dt_i.sqrt()

        if gamma == 1.0:
            uC = torch.log(C.clamp_min(1e-12))
        else:
            uC = (C.clamp_min(1e-12).pow(1.0 - gamma) - 1.0) / (1.0 - gamma)
        disc = torch.exp(-rho * t_i).clamp_max(1e6)
        util_cons = util_cons + disc * uC * dt_i

    XT = logX.exp().clamp_min(lb_X)
    if gamma == 1.0:
        uT = torch.log(XT)
    else:
        uT = (XT.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)
    util = util_cons + (kappa * math.exp(-rho * T)) * uT

    return util.view(-1)

def get_traj_schema():
    """
    mt_nd_max_c 모델의 궤적 시각화 방법을 정의합니다.
    """
    # d의 현재 값에 따라 동적으로 라벨 생성
    u_labels = [f"u_{i+1}" for i in range(d)] + ["Consumption"]

    return {
        "roles": {
            "X": {"dim": DIM_X, "labels": ["Wealth"]},
            # 제어 U의 차원이 d+1이고, 마지막이 소비임을 명시
            "U": {"dim": DIM_U, "labels": u_labels}, 
        },
        "views": [
            # 시간에 따른 소비 궤적을 그리는 뷰 추가
            {
                "name": "Consumption_Path",
                "block": "U",
                "mode": "indices",
                "indices": [d], # d+1개 중 마지막 인덱스(d)가 소비
                "ylabel": "Consumption (C)"
            },
            # 기존의 첫 번째 투자 비중 뷰
            {
                "name": "Portfolio_Weight_First_Component",
                "block": "U",
                "mode": "indices",
                "indices": [0],
                "ylabel": "Portfolio Weight u[0]"
            },
            # 자산 뷰
            {
                "name": "Wealth_Path",
                "block": "X",
                "mode": "indices",
                "indices": [0],
                "ylabel": "Wealth (X)"
            }
        ],
        "sampling": {"Bmax": 5}
    }

# ------------------------------------------------------------
# 모듈 export
# ------------------------------------------------------------
__all__ = [
    "device", "T", "m", "d", "k", "DIM_X", "DIM_Y", "DIM_U",
    "epochs", "batch_size", "lr", "CRN_SEED_EU",
    "alpha", "Sigma", "Sigma_inv", "u_closed_form_unc",
    "sample_initial_states", "simulate",
    "DirectPolicy", "WrappedCFPolicy", "build_closed_form_policy",
    "make_generator"
]
