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
# - 우선 u^2 패널티 반영 버전(full)을 시도, 실패시 기존 모듈로 폴백
try:
    from closed_form_ref_full import precompute_ABC, ClosedFormPolicy
    _CF_FULL = True
except ImportError:
    from closed_form_ref import precompute_ABC, ClosedFormPolicy
    _CF_FULL = False

# --- 모델별 설정 및 환경변수 오버라이드 블록 ---

# 1. 모델 고유의 기본값을 설정합니다.
d = 5
k = 3
epochs = 200
batch_size = 1024
lr = 1e-4
seed = 7

# (중복 표기 정리)
epochs     = 200
batch_size = 1024
lr         = 1e-4
seed       = 7

# (NEW) Quadratic penalty coefficient for ||u||^2 (작은 값 권장)
epsilon = 1e-1

# 2. 환경변수가 존재하면 기본값을 덮어씁니다.
d         = int(os.getenv("PGDPO_D", d))
k         = int(os.getenv("PGDPO_K", k))
epochs    = int(os.getenv("PGDPO_EPOCHS", epochs))
batch_size= int(os.getenv("PGDPO_BATCH_SIZE", batch_size))
lr        = float(os.getenv("PGDPO_LR", lr))
seed      = int(os.getenv("PGDPO_SEED", seed))
epsilon   = float(os.getenv("PGDPO_EPSILON", epsilon))

# ==============================================================================
# ===== (A) 사용자 정의 영역: 모델 차원, 파라미터, 하이퍼파라미터 =====
# ==============================================================================

# --------------------------- Model Dimensions ---------------------------
DIM_X = 1      # 자산 X는 부(wealth)를 의미하므로 1차원
DIM_Y = k      # 외생 변수 Y는 k개의 팩터
DIM_U = d      # 제어 u는 d개 자산에 대한 투자 비율

# --------------------------- Config ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------- Market & Utility Parameters ---------------------------
r     = 0.03
gamma = 2.0

# --------------------------- Simulation Parameters ---------------------------
T = 1.0
m = 40

# --------------------------- State & Control Ranges ---------------------------
X0_range = (0.1, 3.0)
u_cap    = 10.0
lb_X     = 1e-5

# --------------------------- Evaluation Parameters ---------------------------
N_eval_states = 100
CRN_SEED_EU   = 12345

# ====== Reasonable Market Generator (선택: 필요 시 교체 사용) ======

def _nearest_spd_correlation(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    C = 0.5 * (C + C.T)
    evals, evecs = torch.linalg.eigh(C)
    evals_clipped = torch.clamp(evals, min=eps)
    C_spd = (evecs @ torch.diag(evals_clipped) @ evecs.T)
    d = torch.diag(C_spd).clamp_min(eps).sqrt()
    Dinv = torch.diag(1.0 / d)
    C_corr = Dinv @ C_spd @ Dinv
    return 0.5 * (C_corr + C_corr.T)

def _build_block_correlation(d: int, k: int,
                             avg_corr_assets: float = 0.30,
                             avg_corr_factors: float = 0.20,
                             cross_std: float = 0.10,
                             jitter: float = 0.05,
                             seed: int | None = None,
                             device: torch.device | str = "cpu",
                             dtype: torch.dtype = torch.float32):
    rng = np.random.default_rng(seed)
    Psi = torch.full((d, d), float(avg_corr_assets), device=device, dtype=dtype)
    Psi.fill_diagonal_(1.0)
    noise_A = torch.from_numpy(rng.normal(0.0, jitter, size=(d, d))).to(device=device, dtype=dtype)
    Psi = _nearest_spd_correlation(Psi + 0.5 * (noise_A + noise_A.T))
    if k > 0:
        Phi_Y = torch.full((k, k), float(avg_corr_factors), device=device, dtype=dtype)
        Phi_Y.fill_diagonal_(1.0)
        noise_F = torch.from_numpy(rng.normal(0.0, jitter, size=(k, k))).to(device=device, dtype=dtype)
        Phi_Y = _nearest_spd_correlation(Phi_Y + 0.5 * (noise_F + noise_F.T))
        rho_Y = torch.from_numpy(rng.normal(0.0, cross_std, size=(d, k))).to(device=device, dtype=dtype)
    else:
        Phi_Y = torch.empty(0, 0, device=device, dtype=dtype)
        rho_Y = torch.empty(d, 0, device=device, dtype=dtype)
    block_corr = torch.zeros((d + k, d + k), device=device, dtype=dtype)
    block_corr[:d, :d] = Psi
    if k > 0:
        block_corr[d:, d:] = Phi_Y
        block_corr[:d, d:] = rho_Y
        block_corr[d:, :d] = rho_Y.T
    block_corr = _nearest_spd_correlation(block_corr)
    Psi = block_corr[:d, :d]
    if k > 0:
        Phi_Y = block_corr[d:, d:]
        rho_Y = block_corr[:d, d:]
    return Psi, Phi_Y, rho_Y, block_corr

def _calibrate_alpha(d: int, k: int,
                     sigma_vec: torch.Tensor,
                     kappa_Y: torch.Tensor,
                     theta_Y: torch.Tensor,
                     sigma_Y: torch.Tensor,
                     target_mu_excess_std: float = 0.05,
                     alpha_init_std: float = 0.15,
                     max_abs_alpha: float = 0.8,
                     nsamples: int = 1024,
                     seed: int | None = None,
                     device: torch.device | str = "cpu",
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if k == 0:
        return torch.empty(d, 0, device=device, dtype=dtype)
    rng = np.random.default_rng(seed)
    alpha = torch.from_numpy(rng.normal(0.0, alpha_init_std, size=(d, k))).to(device=device, dtype=dtype)
    kappa_diag = torch.diag(kappa_Y).clamp_min(1e-6)
    sigmaY_diag = torch.diag(sigma_Y)
    var_z = (sigmaY_diag ** 2) / (2.0 * kappa_diag)
    std_z = torch.sqrt(var_z).to(device=device, dtype=dtype)
    z_samples = torch.from_numpy(
        rng.normal(loc=theta_Y.cpu().numpy(), scale=std_z.cpu().numpy(), size=(nsamples, k))
    ).to(device=device, dtype=dtype)
    az = z_samples @ alpha.T
    mu_excess = az * sigma_vec
    std_per_asset = mu_excess.std(dim=0).clamp_min(1e-12)
    median_std = torch.median(std_per_asset)
    scale = float(target_mu_excess_std / median_std.item()) if median_std.item() > 0 else 1.0
    return torch.clamp(alpha * scale, min=-max_abs_alpha, max=max_abs_alpha)

def _sample_z_stationary(kappa_Y, theta_Y, sigma_Y, nsamples, seed, device, dtype):
    k = kappa_Y.shape[0]
    if k == 0:
        return torch.zeros(nsamples, 0, device=device, dtype=dtype)
    rng = np.random.default_rng(seed)
    kappa = torch.diag(kappa_Y).clamp_min(1e-8)
    sigY  = torch.diag(sigma_Y).clamp_min(1e-12)
    std   = sigY / torch.sqrt(2.0 * kappa)
    z = rng.normal(loc=theta_Y.cpu().numpy(),
                   scale=std.cpu().numpy(),
                   size=(nsamples, k))
    return torch.tensor(z, device=device, dtype=dtype)

@torch.no_grad()
def _calibrate_mu_and_gamma(alpha, sigma_vec, Sigma_reg, kappa_Y, theta_Y, sigma_Y,
                            S_target=0.7, L_cap=3.0, q_weights=0.95, cap_mu_over_var=4.0,
                            nsamples=2048, seed=777, device="cpu", dtype=torch.float32):
    d_ = sigma_vec.shape[0]
    z = _sample_z_stationary(kappa_Y, theta_Y, sigma_Y, nsamples, seed, device, dtype)
    mu0 = (z @ alpha.T) * sigma_vec
    Sigma_inv = torch.linalg.inv(Sigma_reg)
    quad = torch.einsum('bd,dd,bd->b', mu0, Sigma_inv, mu0).clamp_min(1e-18)
    S_med = torch.sqrt(torch.median(quad))
    s_mu  = float(S_target / S_med.item()) if S_med.item() > 0 else 1.0
    mu    = s_mu * mu0
    ratio = (mu.abs() / (sigma_vec**2).clamp_min(1e-12))
    r_q99 = torch.quantile(ratio.reshape(-1), 0.99).item()
    if r_q99 > cap_mu_over_var:
        s_mu *= (cap_mu_over_var / r_q99)
        mu    = s_mu * mu0
    w_tan = torch.linalg.solve(Sigma_reg, mu.T).T
    w_norm = torch.linalg.norm(w_tan, dim=1)
    w_q = torch.quantile(w_norm, q_weights).item()
    gamma_calib = max(w_q / L_cap, 1e-6)
    alpha_scaled = s_mu * alpha
    return alpha_scaled, gamma_calib, {
        "S_med_after": float(torch.sqrt(torch.median(
            torch.einsum('bd,dd,bd->b', mu, Sigma_inv, mu)
        )).item()),
        "mu_over_var_q99": r_q99 if r_q99 <= cap_mu_over_var else cap_mu_over_var,
        "w_norm_q": w_q, "scale_mu": s_mu, "gamma": gamma_calib
    }

def _generate_market_params_reasonable(d: int, k: int, r: float, dev, seed: int | None = None):
    device = torch.device(dev); dtype = torch.float32
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    if k > 0:
        kappa_Y = torch.diag(torch.empty(k, device=device, dtype=dtype).uniform_(0.7, 1.5))
        theta_Y = torch.empty(k, device=device, dtype=dtype).uniform_(-0.05, 0.05)
        sigma_Y = torch.diag(torch.empty(k, device=device, dtype=dtype).uniform_(0.10, 0.20))
    else:
        kappa_Y = torch.empty(0, 0, device=device, dtype=dtype)
        theta_Y = torch.empty(0, device=device, dtype=dtype)
        sigma_Y = torch.empty(0, 0, device=device, dtype=dtype)
    sigma = torch.empty(d, device=device, dtype=dtype).uniform_(0.18, 0.28)
    Psi, Phi_Y, rho_Y, block_corr = _build_block_correlation(d, k, seed=seed, device=device, dtype=dtype)
    Sigma = torch.diag(sigma) @ Psi @ torch.diag(sigma)
    lam = 1e-3 * float(torch.mean(torch.diag(Sigma)).item())
    Sigma_reg = Sigma + lam * torch.eye(d, device=device, dtype=dtype)
    Sigma_inv = torch.linalg.inv(Sigma_reg)
    alpha = _calibrate_alpha(d, k, sigma, kappa_Y, theta_Y, sigma_Y, device=device, dtype=dtype)
    cholesky_L = torch.linalg.cholesky(block_corr)
    alpha, gamma_adj, _ = _calibrate_mu_and_gamma(
        alpha=alpha, sigma_vec=sigma, Sigma_reg=Sigma_reg,
        kappa_Y=kappa_Y, theta_Y=theta_Y, sigma_Y=sigma_Y, device=device, dtype=dtype
    )
    return {
        'kappa_Y': kappa_Y, 'theta_Y': theta_Y, 'sigma_Y': sigma_Y,
        'sigma': sigma, 'alpha': alpha, 'Phi_Y': Phi_Y, 'Psi': Psi, 'rho_Y': rho_Y,
        'Sigma': Sigma_reg, 'Sigma_inv': Sigma_inv, 'block_corr': block_corr,
        'cholesky_L': cholesky_L
    }

# 파라미터 생성 실행
params = _generate_market_params_reasonable(d, k, r, device, seed)  # << 필요시 이 줄로 교체

# 자주 사용하는 파라미터를 전역 변수로 추출
alpha, sigma, kappa_Y, theta_Y, sigma_Y, rho_Y, Psi, Phi_Y = [
    params[key] for key in ['alpha', 'sigma', 'kappa_Y', 'theta_Y', 'sigma_Y', 'rho_Y', 'Psi', 'Phi_Y']
]
Sigma, Sigma_inv, block_corr, cholesky_L = params['Sigma'], params['Sigma_inv'], params['block_corr'], params['cholesky_L']

# Y의 평균 범위를 계산 (상태 샘플링에 사용)
if k > 0:
    Y_max_vec = theta_Y + 2. * torch.diag(sigma_Y)
    Y_min_vec = torch.zeros_like(Y_max_vec)
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
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)

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
    """다차원 SDE를 시뮬레이션합니다. (utility-scaled u^2 패널티 반영)"""
    m_eff = m_steps if m_steps is not None else m
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)          # dt: (B,1)
    else:
        states, dt = initial_states_dict, initial_states_dict['TmT'] / float(m_eff)

    logX = states['X'].clamp_min(lb_X).log()
    Y = states.get('Y')

    # 브라운 모션 샘플
    if random_draws is None:
        uncorrelated_Z = torch.randn(B, m_eff, d + k, device=device, generator=rng)
        Z = torch.einsum('bmd,dn->bmn', uncorrelated_Z, cholesky_L)
        ZX, ZY = Z[:, :, :d], Z[:, :, d:]
    else:
        ZX, ZY = random_draws

    # ---- u^2 적분 누적기 (utility-scaled): ∫ X^{1-γ} ||u||^2 dt  # [FIX]
    u_sq_int_weighted = torch.zeros(B, 1, device=device, dtype=logX.dtype)

    for i in range(m_eff):
        X_curr = logX.exp()
        current_states = {'X': X_curr, 'TmT': states['TmT'] - i * dt}
        if Y is not None:
            current_states['Y'] = Y

        u = policy(**current_states)                                # (B,d)

        # ---- u^2 적분 누적: ∫ X^{1-γ} ||u||^2 dt  # [FIX]
        u_sq = (u * u).sum(dim=1, keepdim=True)                     # (B,1)
        weight = X_curr.pow(1.0 - gamma)                            # (B,1)
        u_sq_int_weighted = u_sq_int_weighted + (u_sq * weight) * dt

        if Y is not None:
            risk_premium = alpha @ Y.unsqueeze(-1)                   # (d,k)@(B,k,1) -> (B,d,1)
            drift_term = r + (u * sigma * risk_premium.squeeze(-1)).sum(1, keepdim=True)
            var_term = torch.einsum('bi,bi->b', (u * sigma) @ Psi, (u * sigma)).view(-1, 1)

            Y_drift = (theta_Y - Y) @ kappa_Y.T
            dBY = (sigma_Y @ ZY[:, i, :].T).T
            Y = Y + Y_drift * dt + dBY * dt.sqrt()
        else:  # Merton k=0 (참고: alpha가 빈 텐서일 수 있음)
            drift_term = r + (u * alpha).sum(1, keepdim=True)
            var_term = torch.einsum('bi,bi->b', u @ Sigma, u).view(-1, 1)

        dBX = (u * sigma * ZX[:, i, :]).sum(1, keepdim=True)
        logX = logX + (drift_term - 0.5 * var_term) * dt + dBX * dt.sqrt()

    XT = logX.exp().clamp_min(lb_X)

    crra = (XT.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)      # (B,1)
    penalty = 0.5 * epsilon * u_sq_int_weighted             # (B,1)  # [FIX]
    
    return crra - penalty

class WrappedCFPolicy(nn.Module):
    """ClosedFormPolicy가 ver2의 표준 입력 `**states_dict`를 받도록 감싸는 래퍼 클래스"""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
    def forward(self, **states_dict):
        return self.policy.forward(**states_dict)

def build_closed_form_policy():
    """분석적 해를 계산하고, ver2 프레임워크와 호환되는 정책 모듈을 빌드합니다."""
    ode_solution = precompute_ABC(params, T, gamma, epsilon)           # [FIX] epsilon 전달
    cf_policy    = ClosedFormPolicy(ode_solution, params, T, gamma, epsilon).to(device)  # [FIX]
    return WrappedCFPolicy(cf_policy), ode_solution