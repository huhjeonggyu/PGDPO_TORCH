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
    avg_corr=0.50,
    jitter=0.06,
    alpha_range=(0.05, 0.22),        # ← 살짝 상향
    short_frac=0.25,
    alpha_hedge_range=(-0.03, 0.02),
    enforce_shorts: bool = True,
    target_neg: int = 1,
    max_tries: int = 6,

    # 합(∑u) 목표: ‘조금 더’ 높게
    enforce_sum: bool = True,
    sum_target: float | None = None,
    sum_range=(0.40, 0.75),          # ← 상향
    sum_bias_exp: float = 0.7,       # ← 상단 쪽으로 편향 (0<exp<1이면 상단 편향)

    # 분산화/집중 완화
    rho_div: float = 0.35,
    wmax_cap: float = 0.70,

    # 숏 통제
    short_mass_frac: float = 0.15,
    short_count_max_frac: float = 0.40,

    # 공분산 정규화
    lam_factor: float = 2e-3,        # ← 릿지 약화(기존 3e-3)
):
    if seed is not None:
        torch.manual_seed(seed)

    def _sample_sigma_corr(_avg_corr: float, _jitter: float):
        sigma = torch.empty(d, device=dev).uniform_(*vol_range)
        Psi = torch.full((d, d), float(_avg_corr), device=dev)
        Psi.fill_diagonal_(1.0)
        if _jitter and _jitter > 0:
            N = torch.randn(d, d, device=dev) * _jitter
            Psi = _nearest_spd_correlation(Psi + 0.5 * (N + N.T))
        Sigma = torch.diag(sigma) @ Psi @ torch.diag(sigma)
        lam = lam_factor * Sigma.diag().mean().item()
        Sigma = Sigma + lam * torch.eye(d, device=dev)
        return Sigma

    _corr, _jit = avg_corr, jitter

    for _ in range(max_tries):
        Sigma = _sample_sigma_corr(_corr, _jit)
        Sigma_inv = torch.linalg.inv(Sigma)

        alpha = torch.empty(d, device=dev).uniform_(*alpha_range)
        if short_frac > 0:
            m = max(1, int(d * short_frac))
            idx = torch.randperm(d, device=dev)[:m]
            alpha[idx] = torch.empty(m, device=dev).uniform_(*alpha_hedge_range)

        u_unc = (1.0 / gamma) * (Sigma_inv @ alpha)

        if (not enforce_shorts) or (u_unc < 0).sum().item() >= target_neg:
            break
        _corr = min(0.80, _corr + 0.08)
        _jit  = min(0.12, _jit + 0.02)

    # 합 목표 정렬(상단 편향 샘플링)
    if enforce_sum:
        if sum_target is None:
            a, b = map(float, sum_range)
            r = torch.rand((), device=dev)
            r = r ** sum_bias_exp          # 0<exp<1 → 상단 편향
            S_target = (a + (b - a) * r).item()
        else:
            S_target = float(sum_target)

        one = torch.ones(d, device=dev)
        S_curr = float(u_unc.sum().item())
        deltaS = S_target - S_curr
        alpha = alpha + gamma * deltaS * (Sigma @ one) / float(d)
        u = (1.0 / gamma) * (Sigma_inv @ alpha)
    else:
        u = u_unc.clone()

    # 단순체(합=S_target) 투영과 블렌딩 → 집중 완화 (합 보존)
    def _proj_simplex(v, mass):
        vpos = v.clamp_min(0.0)
        s = vpos.sum()
        if s <= 1e-12:
            return torch.full_like(v, mass / d)
        u_sorted, _ = torch.sort(vpos, descending=True)
        cssv = torch.cumsum(u_sorted, 0) - mass
        j = torch.arange(1, d + 1, device=dev, dtype=v.dtype)
        cond = u_sorted > (cssv / j)
        if cond.any():
            rho = int(torch.nonzero(cond, as_tuple=False)[-1].item())
            theta = cssv[rho] / float(rho + 1)
        else:
            theta = cssv[-1] / float(d)
        return (vpos - theta).clamp_min(0.0)

    S_target_cur = float(u.sum().item())
    u_pos = _proj_simplex(u, S_target_cur)
    u = (1.0 - rho_div) * u + rho_div * u_pos

    # 숏 질량/개수 제한 (합 보존)
    neg = (-u).clamp_min(0.0)
    neg_mass = float(neg.sum().item())
    cap_mass = float(short_mass_frac * S_target_cur)
    if neg_mass > cap_mass + 1e-12:
        s = cap_mass / (neg_mass + 1e-12)
        delta = (1.0 - s) * neg
        u[u < 0] = -s * neg[u < 0]
        pos_idx = (u > 0)
        pos_mass = float(u[pos_idx].sum().item())
        if pos_mass > 1e-12:
            u[pos_idx] -= u[pos_idx] * (delta.sum() / pos_mass)

    max_neg_cnt = int(short_count_max_frac * d)
    neg_idx = (u < 0).nonzero(as_tuple=False).view(-1)
    if neg_idx.numel() > max_neg_cnt:
        vals = u[neg_idx].abs()
        _, order = torch.sort(vals)
        kill = neg_idx[order[:(neg_idx.numel() - max_neg_cnt)]]
        freed = (-u[kill]).sum()
        u[kill] = 0.0
        pos_idx = (u > 0)
        pos_mass = float(u[pos_idx].sum().item())
        if pos_mass > 1e-12:
            u[pos_idx] -= u[pos_idx] * (freed / pos_mass)

    # 최대 비중 캡 + 재분배(합 보존)
    if wmax_cap is not None:
        over = (u > wmax_cap)
        if over.any():
            excess = (u[over] - wmax_cap).sum()
            u[over] = wmax_cap
            pos_idx = (u > 0) & (~over)
            pos_mass = float(u[pos_idx].sum().item())
            if pos_mass > 1e-12:
                u[pos_idx] -= u[pos_idx] * (excess / pos_mass)
            else:
                u += excess / d

    # 최종 α 역매핑
    alpha = gamma * (Sigma @ u)
    Sigma_inv = torch.linalg.inv(Sigma)

    return {
        "alpha": alpha,
        "Sigma": Sigma,
        "Sigma_inv": Sigma_inv,
        "u_closed_form_unc": u,
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