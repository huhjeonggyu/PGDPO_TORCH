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
from closed_form_ref import precompute_ABC, ClosedFormPolicy

# --- 모델별 설정 및 환경변수 오버라이드 블록 ---

# 1. 모델 고유의 기본값을 설정합니다.
d = 5
k = 3
epochs = 200
batch_size = 1024
lr = 1e-4
seed = 7

epochs     = 200 
batch_size = 1024 
lr         = 1e-4

seed = 7 # pgdpo_base.py에서 이 변수를 사용해 실제 시드를 설정 <-- 상단 블록에서 제어

# 2. 환경변수가 존재하면 기본값을 덮어씁니다.
d = int(os.getenv("PGDPO_D", d))
k = int(os.getenv("PGDPO_K", k))
epochs = int(os.getenv("PGDPO_EPOCHS", epochs))
batch_size = int(os.getenv("PGDPO_BATCH_SIZE", batch_size))
lr = float(os.getenv("PGDPO_LR", lr))
seed = int(os.getenv("PGDPO_SEED", seed))

# --- 블록 끝 ---


# ==============================================================================
# ===== (A) 사용자 정의 영역: 모델 차원, 파라미터, 하이퍼파라미터 =====
# ==============================================================================

# --------------------------- Model Dimensions ---------------------------
# d: 자산 수, k: 팩터 수
# d = 5 <-- 상단 블록에서 제어
# k = 3 <-- 상단 블록에서 제어
DIM_X = 1      # 자산 X는 부(wealth)를 의미하므로 1차원
DIM_Y = k      # 외생 변수 Y는 k개의 팩터
DIM_U = d      # 제어 u는 d개 자산에 대한 투자 비율

# --------------------------- Config ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------- Market & Utility Parameters ---------------------------
r = 0.03
gamma = 2.0

# --------------------------- Simulation Parameters ---------------------------
T = 1.
m = 40

# --------------------------- State & Control Ranges ---------------------------
X0_range = (0.1, 3.0)
u_cap = 10.
lb_X  = 1e-5

# --------------------------- Evaluation Parameters ---------------------------
N_eval_states = 100
CRN_SEED_EU   = 12345

# --------------------------- Parameter Generation ---------------------------
def _generate_market_params(d, k, r, dev, seed=None):
    """다차원 시장 파라미터를 생성하고 torch 텐서로 반환합니다."""
    # pgdpo_base.py에서 이미 NumPy/PyTorch 시드를 모두 설정했지만,
    # 이 함수가 독립적으로도 결정적으로 작동하도록 시드를 한 번 더 명시합니다.
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 팩터 동역학 파라미터 (k x k, k) — linspace → Uniform 샘플링
    if k > 0:
        # kappa_Y: 각 대각 성분 ~ U[2.0, 2.0 + 0.5*(k-1)]
        kappa_min, kappa_max = 2.0, 2.0 + 0.5 * (k - 1)
        kappa_vals = torch.empty(k).uniform_(kappa_min, kappa_max)
        kappa_Y = torch.diag(kappa_vals)

        # theta_Y: 각 성분 ~ U[0.2, 0.4]
        theta_Y = torch.empty(k).uniform_(0.2, 0.4)

        # sigma_Y: 대각 각 성분 ~ U[0.3, 0.5]
        sigmaY_vals = torch.empty(k).uniform_(0.3, 0.5)
        sigma_Y = torch.diag(sigmaY_vals)
    else:
        kappa_Y = torch.empty(0, 0)
        theta_Y = torch.empty(0)
        sigma_Y = torch.empty(0, 0)

    # 자산별 파라미터 (d,) — linspace → Uniform 샘플링
    sigma = torch.empty(d).uniform_(0.2, 0.4)

    # α ~ Dirichlet (재현성 위해 random_state 사용)
    alpha_np = dirichlet.rvs([1.0] * k, size=d, random_state=seed) if k > 0 else np.zeros((d, 0))
    alpha = torch.tensor(alpha_np, dtype=torch.float32)

    # 상관관계 구조 (d x d, k x k, d x k)
    beta_corr = torch.empty(d).uniform_(-0.8, 0.8)
    Psi = torch.outer(beta_corr, beta_corr); Psi.fill_diagonal_(1.0)

    if k > 0:
        Z_Y = torch.randn(k, max(1, k)); corr_Y = Z_Y @ Z_Y.T
        d_inv_sqrt_Y = torch.diag(1.0 / torch.sqrt(torch.diag(corr_Y).clamp(min=1e-8)))
        Phi_Y = d_inv_sqrt_Y @ corr_Y @ d_inv_sqrt_Y; Phi_Y.fill_diagonal_(1.0)
        rho_Y = torch.empty(d, k).uniform_(-0.2, 0.2)
    else:
        Phi_Y = torch.empty(0, 0)
        rho_Y = torch.empty(d, 0)

    # 전체 상관 행렬 (Positive-Definite 보정 포함)
    block_corr = torch.zeros((d + k, d + k), dtype=torch.float32)
    block_corr[:d, :d], block_corr[d:, d:] = Psi, Phi_Y
    if k > 0:
        block_corr[:d, d:], block_corr[d:, :d] = rho_Y, rho_Y.T

    eigvals = torch.linalg.eigvalsh(block_corr)
    if eigvals.min() < 1e-6:
        block_corr += (abs(eigvals.min()) + 1e-4) * torch.eye(d + k)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(block_corr)))
        block_corr = D_inv_sqrt @ block_corr @ D_inv_sqrt
        Psi = block_corr[:d, :d]
        if k > 0:
            Phi_Y, rho_Y = block_corr[d:, d:], block_corr[:d, d:]

    params = {
        'kappa_Y': kappa_Y, 'theta_Y': theta_Y, 'sigma_Y': sigma_Y,
        'sigma': sigma, 'alpha': alpha, 'Phi_Y': Phi_Y, 'Psi': Psi, 'rho_Y': rho_Y
    }
    params['Sigma'] = torch.diag(params['sigma']) @ params['Psi'] @ torch.diag(params['sigma'])
    params['Sigma_inv'] = torch.linalg.inv(params['Sigma'])
    params['block_corr'] = block_corr
    params['cholesky_L'] = torch.linalg.cholesky(params['block_corr'])

    return {p_key: v.to(dev) for p_key, v in params.items()}

# --------------------------- Reasonable Market Parameter Generation ---------------------------
# 기존 _generate_market_params(d, k, r, dev, seed=None)를 "그대로" 교체해도 됩니다.
# torch, numpy 외 외부 의존성 없음. 반환 키들은 기존 코드와 동일합니다.
import math
import numpy as np
import torch

def _nearest_spd_correlation(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    대칭 행렬 C를 '가장 가까운'(eigenvalue floor) 상관행렬로 투영:
    1) 대칭화 -> 고유값 floor -> 재구성
    2) 대각 원소를 1로 재정규화 (D^{-1/2} C D^{-1/2})
    """
    C = 0.5 * (C + C.T)
    # 고유값 분해
    evals, evecs = torch.linalg.eigh(C)
    evals_clipped = torch.clamp(evals, min=eps)
    C_spd = (evecs @ torch.diag(evals_clipped) @ evecs.T)
    # 상관행렬로 정규화 (diag -> 1)
    d = torch.diag(C_spd).clamp_min(eps).sqrt()
    Dinv = torch.diag(1.0 / d)
    C_corr = Dinv @ C_spd @ Dinv
    # 수치 안정화
    C_corr = 0.5 * (C_corr + C_corr.T)
    return C_corr

def _build_block_correlation(d: int, k: int,
                             avg_corr_assets: float = 0.30,
                             avg_corr_factors: float = 0.20,
                             cross_std: float = 0.10,
                             jitter: float = 0.05,
                             seed: int | None = None,
                             device: torch.device | str = "cpu",
                             dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    자산/팩터/교차 상관을 합쳐 (d+k)x(d+k) 블록 상관행렬을 직접 구성하고 SPD로 투영.
    - Psi: (d,d) 자산 상관, 평균 ≈ avg_corr_assets
    - Phi_Y: (k,k) 팩터 상관, 평균 ≈ avg_corr_factors
    - rho_Y: (d,k) 자산-팩터 상관, 표준편차 ≈ cross_std
    반환: Psi, Phi_Y, rho_Y, block_corr
    """
    rng = np.random.default_rng(seed)
    # 자산 상관(기본: 상수 상관 + 작은 잡음)
    Psi = torch.full((d, d), float(avg_corr_assets), device=device, dtype=dtype)
    Psi.fill_diagonal_(1.0)
    noise_A = torch.from_numpy(rng.normal(0.0, jitter, size=(d, d))).to(device=device, dtype=dtype)
    Psi = Psi + 0.5 * (noise_A + noise_A.T)
    Psi = _nearest_spd_correlation(Psi)

    # 팩터 상관
    if k > 0:
        Phi_Y = torch.full((k, k), float(avg_corr_factors), device=device, dtype=dtype)
        Phi_Y.fill_diagonal_(1.0)
        noise_F = torch.from_numpy(rng.normal(0.0, jitter, size=(k, k))).to(device=device, dtype=dtype)
        Phi_Y = Phi_Y + 0.5 * (noise_F + noise_F.T)
        Phi_Y = _nearest_spd_correlation(Phi_Y)
        # 교차 상관
        rho_Y = torch.from_numpy(rng.normal(0.0, cross_std, size=(d, k))).to(device=device, dtype=dtype)
    else:
        Phi_Y = torch.empty(0, 0, device=device, dtype=dtype)
        rho_Y = torch.empty(d, 0, device=device, dtype=dtype)

    # 블록 상관행렬 구성 후, 한 번 더 SPD 투영
    block_corr = torch.zeros((d + k, d + k), device=device, dtype=dtype)
    block_corr[:d, :d] = Psi
    if k > 0:
        block_corr[d:, d:] = Phi_Y
        block_corr[:d, d:] = rho_Y
        block_corr[d:, :d] = rho_Y.T
    block_corr = _nearest_spd_correlation(block_corr)  # SPD + diag=1
    # 투영으로 인해 Psi/Phi_Y/rho_Y가 약간 달라질 수 있으므로 재추출
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
    """
    alpha ~ N(0, alpha_init_std^2)에서 시작하여, OU 정상분포 z ~ N(theta, Sigma_z) 샘플로
    mu_excess_i = sigma_i * sum_j(alpha_{ij} z_j)의 표준편차가 target_mu_excess_std에
    근접하도록 alpha 스케일을 자동 조정.
    """
    if k == 0:
        return torch.empty(d, 0, device=device, dtype=dtype)

    rng = np.random.default_rng(seed)
    # 초기 alpha
    alpha = torch.from_numpy(rng.normal(0.0, alpha_init_std, size=(d, k))).to(device=device, dtype=dtype)

    # OU 정상분포: Var(z_j) = sigmaY_j^2 / (2 * kappaY_j), 독립 가정(대각)으로 간단화
    kappa_diag = torch.diag(kappa_Y).clamp_min(1e-6)
    sigmaY_diag = torch.diag(sigma_Y)
    var_z = (sigmaY_diag ** 2) / (2.0 * kappa_diag)
    std_z = torch.sqrt(var_z).to(device=device, dtype=dtype)

    # z 샘플
    z_samples = torch.from_numpy(
        rng.normal(loc=theta_Y.cpu().numpy(), scale=std_z.cpu().numpy(), size=(nsamples, k))
    ).to(device=device, dtype=dtype)

    # 현재 alpha에서 mu_excess std 계산
    # mu_excess (nsamples, d): diag(sigma) @ (alpha @ z)
    alpha_T = alpha.transpose(0, 1)  # (k,d)
    az = z_samples @ alpha_T  # (nsamples, d)
    mu_excess = az * sigma_vec  # broadcasting, sigma_vec: (d,)
    std_per_asset = mu_excess.std(dim=0).clamp_min(1e-12)  # (d,)
    median_std = torch.median(std_per_asset)
    if median_std.item() == 0.0:
        scale = 1.0
    else:
        scale = float(target_mu_excess_std / median_std.item())
    alpha = torch.clamp(alpha * scale, min=-max_abs_alpha, max=max_abs_alpha)
    return alpha

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
                            S_target=0.7,         # 목표 탱전시 샤프(연율)
                            L_cap=3.0,            # 가중치 \ell2 한도 (95% 분위수 기준)
                            q_weights=0.95,       # 가중치 분위수
                            cap_mu_over_var=4.0,  # 상한: max_i |mu_i|/sigma_i^2 (99% 분위수)
                            nsamples=2048, seed=777, device="cpu", dtype=torch.float32):
    """
    alpha: (d,k), sigma_vec: (d,), Sigma_reg: (d,d)
    반환: alpha_scaled, gamma_calibrated, diag report dict
    """
    d = sigma_vec.shape[0]
    z = _sample_z_stationary(kappa_Y, theta_Y, sigma_Y, nsamples, seed, device, dtype)  # (B,k)
    # mu_excess(z) = diag(sigma) @ (alpha z)  -> (B,d)
    mu0 = (z @ alpha.T) * sigma_vec  # broadcasting

    # 1) 탱전시 샤프 보정: s * mu0가 S_target에 맞도록 전역 스케일 s 결정
    #    S(z; s)^2 = (s^2) * mu0^T Σ^{-1} mu0  → 중앙값 기준으로 맞춤
    Sigma_inv = torch.linalg.inv(Sigma_reg)  # 주의: 릿지 포함 역행렬
    quad = torch.einsum('bd,dd,bd->b', mu0, Sigma_inv, mu0).clamp_min(1e-18)
    S_med = torch.sqrt(torch.median(quad))
    s_mu  = float(S_target / S_med.item()) if S_med.item() > 0 else 1.0
    mu    = s_mu * mu0

    # 안전장치: per-asset |mu_i|/sigma_i^2 99% 분위수 상한
    ratio = (mu.abs() / (sigma_vec**2).clamp_min(1e-12))  # (B,d)
    r_q99 = torch.quantile(ratio.reshape(-1), 0.99).item()
    if r_q99 > cap_mu_over_var:
        s_mu *= (cap_mu_over_var / r_q99)
        mu    = s_mu * mu0  # 재스케일

    # 2) 가중치 규모 보정: w*=(1/γ) Σ^{-1} μ 의 \ell2-norm 95% 분위수가 L_cap 이하도록 γ 설정
    w_tan = torch.linalg.solve(Sigma_reg, mu.T).T  # (B,d), hedging 미포함 탱전시 방향
    w_norm = torch.linalg.norm(w_tan, dim=1)       # (B,)
    w_q = torch.quantile(w_norm, q_weights).item()
    gamma_calib = max(w_q / L_cap, 1e-6)  # 너무 작아지지 않게 바닥

    report = {
        "S_med_after": float(torch.sqrt(torch.median(
            torch.einsum('bd,dd,bd->b', mu, Sigma_inv, mu)
        )).item()),
        "mu_over_var_q99": r_q99 if r_q99 <= cap_mu_over_var else cap_mu_over_var,
        "w_norm_q": w_q,
        "scale_mu": s_mu,
        "gamma": gamma_calib
    }

    # alpha 자체를 스케일해서 반환(μ = diag(σ) (alpha z)이므로 alpha←s*alpha)
    alpha_scaled = s_mu * alpha
    return alpha_scaled, gamma_calib, report

def _generate_market_params(d: int, k: int, r: float, dev, seed: int | None = None):
    """
    다차원 '합리적' 시장 파라미터 생성.
    반환 키: 기존 코드와 동일
      - 'kappa_Y', 'theta_Y', 'sigma_Y', 'sigma', 'alpha', 'Phi_Y', 'Psi', 'rho_Y'
      - 'Sigma', 'Sigma_inv', 'block_corr', 'cholesky_L'
      - (참고) Y0_range는 호출부에서 필요시 계산
    """
    device = torch.device(dev)
    dtype = torch.float32
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # --------- 1) 팩터 OU 파라미터 (보수적 스케일) ---------
    if k > 0:
        # 평균회귀 속도: 0.7 ~ 1.5 (연 단위)
        kappa_vals = torch.empty(k, device=device, dtype=dtype).uniform_(0.7, 1.5)
        kappa_Y = torch.diag(kappa_vals)
        # 장기평균: -0.05 ~ 0.05 (거의 0 근방, drift 과대 방지)
        theta_Y = torch.empty(k, device=device, dtype=dtype).uniform_(-0.05, 0.05)
        # 팩터 변동성: 0.10 ~ 0.20
        sigmaY_vals = torch.empty(k, device=device, dtype=dtype).uniform_(0.10, 0.20)
        sigma_Y = torch.diag(sigmaY_vals)
    else:
        kappa_Y = torch.empty(0, 0, device=device, dtype=dtype)
        theta_Y = torch.empty(0, device=device, dtype=dtype)
        sigma_Y = torch.empty(0, 0, device=device, dtype=dtype)

    # --------- 2) 자산 변동성 벡터 σ_i (연 18~28%) ---------
    sigma = torch.empty(d, device=device, dtype=dtype).uniform_(0.18, 0.28)

    # --------- 3) 블록 상관행렬 (Psi, Phi_Y, rho_Y, block_corr) ---------
    Psi, Phi_Y, rho_Y, block_corr = _build_block_correlation(
        d=d, k=k,
        avg_corr_assets=0.30,     # 평균 자산 상관 ~ 0.3
        avg_corr_factors=0.20,    # 평균 팩터 상관 ~ 0.2
        cross_std=0.10,           # 자산-팩터 cross-corr 표준편차 ~ 0.1
        jitter=0.05,              # 약간의 구조적 잡음
        seed=seed,
        device=device, dtype=dtype
    )

    # --------- 4) 공분산 Σ = diag(σ) Ψ diag(σ), 릿지 안정화 ---------
    Sigma = torch.diag(sigma) @ Psi @ torch.diag(sigma)
    lam = 1e-3 * float(torch.mean(torch.diag(Sigma)).item())  # 평균 분산의 0.1% 수준
    Sigma_reg = Sigma + lam * torch.eye(d, device=device, dtype=dtype)

    # 하위호환(직접 역행렬) – 가능하면 solve 사용 권장
    Sigma_inv = torch.linalg.inv(Sigma_reg)

    # --------- 5) 알파(자산×팩터 로딩) 보정: 목표 초과수익 std≈5% ---------
    alpha = _calibrate_alpha(
        d=d, k=k, sigma_vec=sigma, kappa_Y=kappa_Y, theta_Y=theta_Y, sigma_Y=sigma_Y,
        target_mu_excess_std=0.05, alpha_init_std=0.15, max_abs_alpha=0.8,
        nsamples=1024, seed=seed, device=device, dtype=dtype
    )

    # --------- 6) Cholesky (브라운 모션 결합용) ---------
    # block_corr는 이미 SPD & diag=1
    cholesky_L = torch.linalg.cholesky(block_corr)

    alpha, gamma, diag = _calibrate_mu_and_gamma(
        alpha=alpha, sigma_vec=sigma, Sigma_reg=Sigma,  # 주의: 릿지 포함 Σ 사용
        kappa_Y=kappa_Y, theta_Y=theta_Y, sigma_Y=sigma_Y,
        S_target=0.7, L_cap=3.0, q_weights=0.95, cap_mu_over_var=4.0,
        nsamples=2048, seed=seed, device=device, dtype=dtype
    )

    params = {
        'kappa_Y': kappa_Y, 'theta_Y': theta_Y, 'sigma_Y': sigma_Y,
        'sigma': sigma, 'alpha': alpha, 'Phi_Y': Phi_Y, 'Psi': Psi, 'rho_Y': rho_Y,
        'Sigma': Sigma_reg, 'Sigma_inv': Sigma_inv,  # 주의: Sigma(릿지 포함) 기준
        'block_corr': block_corr, 'cholesky_L': cholesky_L
    }
    return params

# 파라미터 생성 실행
params = _generate_market_params(d, k, r, device, seed)

# 자주 사용하는 파라미터를 전역 변수로 추출
alpha, sigma, kappa_Y, theta_Y, sigma_Y, rho_Y, Psi, Phi_Y = [params[key] for key in ['alpha', 'sigma', 'kappa_Y', 'theta_Y', 'sigma_Y', 'rho_Y', 'Psi', 'Phi_Y']]
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
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, gain=0.8)

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
    """다차원 SDE를 시뮬레이션합니다."""
    m_eff = m_steps if m_steps is not None else m
    
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states, dt = initial_states_dict, initial_states_dict['TmT'] / float(m_eff)
        
    logX = states['X'].clamp_min(lb_X).log()
    Y = states.get('Y')
    
    if random_draws is None:
        uncorrelated_Z = torch.randn(B, m_eff, d + k, device=device, generator=rng)
        Z = torch.einsum('bmd,dn->bmn', uncorrelated_Z, cholesky_L)
        ZX, ZY = Z[:, :, :d], Z[:, :, d:]
    else:
        ZX, ZY = random_draws

    for i in range(m_eff):
        current_states = {'X': logX.exp(), 'TmT': states['TmT'] - i * dt}
        if Y is not None: current_states['Y'] = Y
        
        u = policy(**current_states)
        
        if Y is not None:
            risk_premium = alpha @ Y.unsqueeze(-1)
            drift_term = r + (u * sigma * risk_premium.squeeze(-1)).sum(1, keepdim=True)
            var_term = torch.einsum('bi,bi->b', (u * sigma) @ Psi, (u * sigma)).view(-1, 1)
            Y_drift = (theta_Y - Y) @ kappa_Y.T
            dBY = (sigma_Y @ ZY[:, i, :].T).T
            Y = Y + Y_drift * dt + dBY * dt.sqrt()
        else: # Merton case k=0
            drift_term = r + (u * alpha).sum(1, keepdim=True)
            var_term = torch.einsum('bi,bi->b', u @ Sigma, u).view(-1, 1)

        dBX = (u * sigma * ZX[:, i, :]).sum(1, keepdim=True)
        logX = logX + (drift_term - 0.5 * var_term) * dt + dBX * dt.sqrt()

    XT = logX.exp().clamp_min(lb_X)
    return (XT.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)

class WrappedCFPolicy(nn.Module):
    """ClosedFormPolicy가 ver2의 표준 입력 `**states_dict`를 받도록 감싸는 래퍼 클래스"""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
    def forward(self, **states_dict):
        return self.policy.forward(**states_dict)

def build_closed_form_policy():
    """분석적 해를 계산하고, ver2 프레임워크와 호환되는 정책 모듈을 빌드합니다."""

    ode_solution = precompute_ABC(params, T, gamma)
    cf_policy = ClosedFormPolicy(ode_solution, params, T, gamma, u_cap).to(device)
    return WrappedCFPolicy(cf_policy), ode_solution