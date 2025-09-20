# user_pgdpo_base.py for N-dimensional Merton Problem (no-short version)
# 역할: 다차원 머튼 모델의 사용자 정의 요소
# - 정책 신경망: Softplus 출력으로 공매도 금지(u >= 0)
# - 닫힌형 정책: 공매도 금지 KKT 활성집합(조각별 닫힌형)로 u* 계산

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 모델별 설정 및 환경변수 오버라이드 블록 ---

# 1. 기본값
d = 5
k = 0  # Merton: 외생 상태 없음
epochs = 200
batch_size = 1024
lr = 1e-4
seed = 42

# 2. 환경변수로 일부만 덮기
d = int(os.getenv("PGDPO_D", d))
epochs = int(os.getenv("PGDPO_EPOCHS", epochs))
batch_size = int(os.getenv("PGDPO_BATCH_SIZE", batch_size))
lr = float(os.getenv("PGDPO_LR", lr))
seed = int(os.getenv("PGDPO_SEED", seed))
# k는 0 유지

# ==============================================================================
# ===== (A) 차원/파라미터/하이퍼 =====
# ==============================================================================

DIM_X = 1
DIM_Y = k
DIM_U = d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 시장/효용
r = 0.03
gamma = 2.0  # 상수 gamma (코스테이트 추정 대신 상수 사용)

# 시뮬레이션
T = 1.5
m = 20

# 상태/제어 범위
X0_range = (0.1, 3.0)
u_cap = 10.0
lb_X  = 1e-5

# 평가
N_eval_states = 100
CRN_SEED_EU   = 12345

# --------------------------- 파라미터 생성 ---------------------------
def _nearest_spd_correlation(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    C = 0.5 * (C + C.T)
    evals, evecs = torch.linalg.eigh(C)
    evals = torch.clamp(evals, min=eps)
    C_spd = (evecs @ torch.diag(evals) @ evecs.T)
    d = torch.diag(C_spd).clamp_min(eps).sqrt()
    Dinv = torch.diag(1.0 / d)
    C_corr = Dinv @ C_spd @ Dinv
    return 0.5 * (C_corr + C_corr.T)


def _generate_mu_sigma_simple(
    d: int,
    dev,
    seed: int | None = None,
    *,
    vol_range=(0.18, 0.28),
    avg_corr=0.60,               # ↑ 상관을 약간 높여 숏이 잘 생기게
    jitter=0.08,
    alpha_range=(0.04, 0.20),
    short_frac=0.30,             # 헤지 그룹 비율(대략 30%)
    alpha_hedge_range=(-0.05, 0.02),  # 헤지 그룹의 약/음수 프리미엄
    enforce_shorts: bool = True, # 최소 한 개 이상 숏 강제
    target_neg: int = 1,         # 원하는 최소 음수 비중 개수
    max_tries: int = 8,          # 재시도 횟수
):
    if seed is not None:
        torch.manual_seed(seed)

    def _sample_sigma_corr(_avg_corr: float, _jitter: float):
        # 변동도, 상관
        sigma = torch.empty(d, device=dev).uniform_(*vol_range)
        Psi = torch.full((d, d), float(_avg_corr), device=dev)
        Psi.fill_diagonal_(1.0)
        if _jitter and _jitter > 0:
            N = torch.randn(d, d, device=dev) * _jitter
            Psi = _nearest_spd_correlation(Psi + 0.5 * (N + N.T))
        # 공분산(릿지 포함)
        Sigma = torch.diag(sigma) @ Psi @ torch.diag(sigma)
        lam = 1e-3 * Sigma.diag().mean().item()
        Sigma = Sigma + lam * torch.eye(d, device=dev)
        return sigma, Sigma

    # 재시도 루프: 숏이 안 나오면 상관/지터를 조금씩 올려가며 재샘플
    _corr, _jit = avg_corr, jitter
    for _ in range(max_tries):
        sigma, Sigma = _sample_sigma_corr(_corr, _jit)
        Sigma_inv = torch.linalg.inv(Sigma)

        # 기본 α 샘플
        alpha = torch.empty(d, device=dev).uniform_(*alpha_range)

        # 헤지 그룹에 약/음수 α 부여 (셔플)
        if short_frac > 0:
            m = max(1, int(d * short_frac))
            idx = torch.randperm(d, device=dev)[:m]
            alpha[idx] = torch.empty(m, device=dev).uniform_(*alpha_hedge_range)

        # 무제약 머튼 해
        u_unc = (1.0 / gamma) * (Sigma_inv @ alpha)

        if (not enforce_shorts) or (u_unc < 0).sum().item() >= target_neg:
            return {
                "alpha": alpha,
                "Sigma": Sigma,
                "Sigma_inv": Sigma_inv,
                "u_closed_form_unc": u_unc,
            }

        # 실패 시 다음 시도에서 숏이 더 잘 생기도록 상관/지터 소폭 ↑
        _corr = min(0.90, _corr + 0.10)
        _jit  = min(0.15, _jit + 0.02)

    # 마지막 시도 결과라도 반환
    return {
        "alpha": alpha,
        "Sigma": Sigma,
        "Sigma_inv": Sigma_inv,
        "u_closed_form_unc": u_unc,
    }

# (여기 한 줄만 교체)
params = _generate_mu_sigma_simple(d, device, seed)
alpha, Sigma, Sigma_inv = params['alpha'], params['Sigma'], params['Sigma_inv']
u_closed_form_unc = params['u_closed_form_unc']

Y0_range = None

# ==============================================================================
# ===== (B) 정책 네트워크 및 동역학 =====
# ==============================================================================

class DirectPolicy(nn.Module):
    """Softplus로 공매도 금지(u >= 0). 필요 시 상한은 u_cap로 클램프(선택)."""
    def __init__(self, use_cap: bool = False):
        super().__init__()
        state_dim = DIM_X + DIM_Y + 1  # X, (Y), TmT
        self.net = nn.Sequential(
            nn.Linear(state_dim, 200), nn.LeakyReLU(),
            nn.Linear(200, 200), nn.LeakyReLU(),
            nn.Linear(200, DIM_U)
        )
        self.use_cap = use_cap

    def forward(self, **states_dict):
        states_to_cat = [states_dict['X'], states_dict['TmT']]
        if 'Y' in states_dict and states_dict['Y'] is not None and DIM_Y > 0:
            states_to_cat.append(states_dict['Y'])
        x_in = torch.cat(states_to_cat, dim=1)  # (B, state_dim)

        logits = self.net(x_in)                 # (B, d)
        u = F.softplus(logits)                  # (B, d), u >= 0 (공매도 금지)

        # 필요하면 상한 도입 (부드러운 상한을 원하면 sigmoid*cap로 바꿀 수 있음)
        if self.use_cap and torch.isfinite(torch.tensor(u_cap)):
            u = torch.clamp(u, max=float(u_cap))

        return u

def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    """초기 상태 (X, TmT) 샘플링"""
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
    """머튼 SDE 시뮬레이션 (로그-부호화)"""
    m_eff = m_steps if m_steps is not None else m

    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states, dt = initial_states_dict, initial_states_dict['TmT'] / float(m_eff)

    logX = states['X'].clamp_min(lb_X).log()

    if random_draws is None:
        Z = torch.randn(B, m_eff, d, device=device, generator=rng)  # d차원
    else:
        Z = random_draws[0]

    chol_S = torch.linalg.cholesky(Sigma)  # 상수 Σ

    for i in range(m_eff):
        current_states = {'X': logX.exp(), 'TmT': states['TmT'] - i * dt}
        u = policy(**current_states)       # (B, d), u >= 0

        drift_term = r + (u * alpha).sum(1, keepdim=True)
        var_term = torch.einsum('bi,bi->b', u @ Sigma, u).view(-1, 1)

        # dB: u * (Σ^{1/2} dW)
        dBX = (torch.einsum('bi,ij->bj', u, chol_S) * Z[:, i, :]).sum(1, keepdim=True)
        logX = logX + (drift_term - 0.5 * var_term) * dt + dBX * dt.sqrt()

    XT = logX.exp().clamp_min(lb_X)
    return (XT.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)

# ==============================================================================
# ===== (C) 공매도 금지 닫힌형(KKT 활성집합) =====
# ==============================================================================

@torch.no_grad()
def _solve_nonneg_qp_closed_form(alpha_vec: torch.Tensor,
                                 Sigma_mat: torch.Tensor,
                                 gamma_val: float,
                                 tol: float = 1e-12,
                                 max_iter: int = 30) -> torch.Tensor:
    """
    max_{u >= 0} alpha^T u - (gamma/2) u^T Sigma u 의 유일해.
    KKT 활성집합 반복:
      - 활성집합 A에 대해 u_A = (1/gamma) Σ_AA^{-1} alpha_A, u_{A^c}=0
      - 비활성 i에 대해 α_i - γ Σ_{iA} u_A <= 0
    """
    n = alpha_vec.shape[0]

    # 무제약 해로 초기화
    try:
        u = (1.0 / gamma_val) * torch.linalg.solve(Sigma_mat, alpha_vec)
    except RuntimeError:
        L = torch.linalg.cholesky(Sigma_mat)
        u = (1.0 / gamma_val) * torch.cholesky_solve(alpha_vec.unsqueeze(-1), L).squeeze(-1)

    active = (u > 0).clone()

    for _ in range(max_iter):
        if active.sum() == 0:
            return torch.zeros_like(u)

        idx = active.nonzero(as_tuple=True)[0]
        alpha_A = alpha_vec.index_select(0, idx)
        Sigma_AA = Sigma_mat.index_select(0, idx).index_select(1, idx)
        # u_A
        try:
            u_A = (1.0 / gamma_val) * torch.linalg.solve(Sigma_AA, alpha_A)
        except RuntimeError:
            L_AA = torch.linalg.cholesky(Sigma_AA)
            u_A = (1.0 / gamma_val) * torch.cholesky_solve(alpha_A.unsqueeze(-1), L_AA).squeeze(-1)

        u_new = torch.zeros_like(u)
        u_new[idx] = u_A

        # 음수 좌표 제거
        newly_negative = (u_new < -tol)
        if newly_negative.any():
            active = active & (~newly_negative)
            u = torch.clamp_min(u_new, 0.0)
            continue

        # KKT 검사 (비활성 위반 활성화)
        Sigma_iA = Sigma_mat.index_select(1, idx)   # (n, |A|)
        red = Sigma_iA @ u_A                        # (n,)
        kkt = alpha_vec - gamma_val * red
        violated = (~active) & (kkt > tol)
        if violated.any():
            active = active | violated
            u = torch.clamp_min(u_new, 0.0)
            continue

        u = torch.clamp_min(u_new, 0.0)
        break

    return torch.clamp_min(u, 0.0)

class WrappedCFPolicy(nn.Module):
    def __init__(self, u_star: torch.Tensor):
        super().__init__()
        self.register_buffer("u_star", u_star.view(-1))  # (d,)

    def forward(self, **states_dict):
        B = states_dict["X"].shape[0]
        return self.u_star.unsqueeze(0).expand(B, -1)

def build_closed_form_policy():
    """공매도 금지 닫힌형: u* >= 0 (상수 정책)"""
    u_star = _solve_nonneg_qp_closed_form(alpha, Sigma, gamma).to(device).view(-1)
    cf_policy = WrappedCFPolicy(u_star).to(device)
    return cf_policy, None