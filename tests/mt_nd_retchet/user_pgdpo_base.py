# -*- coding: utf-8 -*-
# 파일: tests/mt_nd_retchet/user_pgdpo_base.py
# 목적: 학습 베이스 + (코어 호환) 무인자 build_closed_form_policy() 제공
#       - 학습 정책은 "소비 비율"을 예측 → C = max(Y, ĉ·X)
#       - 환경에서 랫칫팅을 한 번 더 강제 (집행 C = max(C_pred, Y))

from __future__ import annotations
import os, math, torch
import torch.nn as nn
import torch.nn.functional as F

# --- CF 모듈 임포트 (상대/절대 폴백) ---
try:
    from .closed_form_ref import build_cf_with_args
except Exception:
    from tests.mt_nd_retchet.closed_form_ref import build_cf_with_args

# --------------------------- 기본 설정 ---------------------------
d = int(os.getenv("PGDPO_D", 5)); k = 0
epochs = int(os.getenv("PGDPO_EPOCHS", 250))
batch_size = int(os.getenv("PGDPO_BS", 1024))
lr = float(os.getenv("PGDPO_LR", 1e-4))
seed = int(os.getenv("PGDPO_SEED", 24))

T = float(os.getenv("PGDPO_T", 1.5))
m = int(os.getenv("PGDPO_M", 40))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIM_X = 2; DIM_Y = k; DIM_U = d + 1
N_eval_states = int(os.getenv("PGDPO_EVAL_STATES", 100))
CRN_SEED_EU = int(os.getenv("PGDPO_CRN", 68))
USE_CLOSED_FORM = bool(int(os.getenv("PGDPO_CF", "1")))
RIM_STEPS = int(os.getenv("PGDPO_RIM_STEPS", str(max(200, m))))  # BW에선 미사용, 인터페이스 유지

# --------------------------- 경제 파라미터 ---------------------------
gamma = float(os.getenv("PGDPO_GAMMA", 3.0))
rho   = float(os.getenv("PGDPO_RHO",   0.04))
r     = float(os.getenv("PGDPO_RF",    0.01))
kappa = float(os.getenv("PGDPO_KAPPA", 1.0))

# 초기부의 분포
X0_range = (0.2, 1.2)
H0_RATIO = float(os.getenv("PGDPO_H0_RATIO", "0.02"))  # 논문 느낌: Y0 ≈ 0.02 * X0
lb_X = 1e-6

# 포트폴리오 총합(레버리지) 상한
L_cap = float(os.getenv("PGDPO_LCAP", 1.0))

# 학습 정책의 소비 비율 상한 (연율)
C_RATE_MAX = float(os.getenv("PGDPO_C_RATE_MAX", "2.0"))  # 예: 2.0/year

# --------------------------- 시장 파라미터 ---------------------------
@torch.no_grad()
def _nearest_spd_correlation(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    C = 0.5 * (C + C.T)
    evals, evecs = torch.linalg.eigh(C)
    evals = torch.clamp(evals, min=eps)
    C_spd = (evecs @ torch.diag(evals) @ evecs.T)
    d_ = torch.diag(C_spd).clamp_min(eps).sqrt()
    Dinv = torch.diag(1.0 / d_)
    return 0.5 * (Dinv @ C_spd @ Dinv + (Dinv @ C_spd @ Dinv).T)

@torch.no_grad()
def _generate_mu_sigma_balanced(
    d: int, dev, seed: int | None = None, *,
    vol_range=(0.16, 0.26), avg_corr=0.35, jitter=0.035, lam_factor=0.01,
    dirichlet_conc=10.0, target_leverage=None, alpha_mid=0.09,
    noise_std=0.01, hhi_factor=3.0
):
    if seed is not None:
        torch.manual_seed(seed)
    sigma = torch.empty(d, device=dev).uniform_(*vol_range)
    Psi = torch.full((d, d), float(avg_corr), device=dev); Psi.fill_diagonal_(1.0)
    if jitter > 0:
        N = torch.randn(d, d, device=dev) * jitter
        Psi = _nearest_spd_correlation(Psi + 0.5 * (N + N.T))
    Sigma = torch.diag(sigma) @ Psi @ torch.diag(sigma)
    Sigma = Sigma + lam_factor * Sigma.diag().mean() * torch.eye(d, device=dev)

    w_ref = torch.distributions.Dirichlet(torch.full((d,), float(dirichlet_conc), device=dev)).sample()
    hhi_target = hhi_factor / d
    if (w_ref**2).sum() > hhi_target:
        mix = 0.4; w_ref = (1 - mix) * w_ref + mix * (torch.ones(d, device=dev) / d)

    if target_leverage is not None: s = float(target_leverage)
    else: s = float(alpha_mid / (gamma * (Sigma @ w_ref).mean().clamp_min(1e-8)).item())
    alpha = gamma * s * (Sigma @ w_ref)
    if noise_std > 0:
        alpha += float(noise_std) * alpha.abs().mean() * torch.randn_like(alpha)
        alpha = alpha.clamp_min(1e-4)

    Sigma_inv = torch.linalg.inv(Sigma)
    return {"alpha": alpha, "Sigma": Sigma, "Sigma_inv": Sigma_inv}

params = _generate_mu_sigma_balanced(
    d, device, seed=seed,
    target_leverage=0.7 * L_cap,
    dirichlet_conc=10.0,
    hhi_factor=3.0
)
alpha, Sigma, Sigma_inv = params["alpha"], params["Sigma"], params["Sigma_inv"]
chol_S = torch.linalg.cholesky(Sigma)

# --------------------------- 학습 정책 (소비 "비율") ---------------------------
class DirectPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        state_dim = 3; hid = 256
        self.net = nn.Sequential(
            nn.Linear(state_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, d + 2)  # [u logits..., c_rate logit]
        )

    def forward(self, **states_dict):
        wealth, habit = states_dict['X'][:, 0:1], states_dict['X'][:, 1:2]
        TmT = states_dict['TmT']
        s_in = torch.cat([wealth, habit / wealth.clamp_min(1e-8), TmT], dim=1)

        raw = self.net(s_in)

        # (1) 포트폴리오: simplex(L_cap) 내부
        u = L_cap * F.softmax(raw[:, :d+1], dim=1)[:, :d]

        # (2) 소비: 비율 ĉ ∈ [0, C_RATE_MAX], 실제 C = max(habit, ĉ·wealth)
        c_rate = torch.sigmoid(raw[:, d+1:]) * C_RATE_MAX
        C = torch.maximum(habit, c_rate * wealth)

        return torch.cat([u, C], dim=1)

# --------------------------- 초기 상태/시뮬레이터 ---------------------------
def sample_initial_states(B, *, rng=None):
    wealth0 = torch.rand((B, 1), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
    habit0  = (wealth0 * H0_RATIO).clamp_min(1e-8)  # 랫칫 관찰 용이(작은 습관)
    X0 = torch.cat([wealth0, habit0], dim=1)
    TmT0 = torch.rand((B, 1), device=device, generator=rng) * T
    return {'X': X0, 'Y': None, 'TmT': TmT0}, TmT0 / float(m)

def simulate(policy, B, *, train=True, rng=None, initial_states_dict=None, random_draws=None, m_steps=None):
    m_eff = m_steps or m
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states = initial_states_dict
        dt = states['TmT'] / m_eff

    wealth = states['X'][:, 0:1].clamp_min(lb_X)
    habit  = states['X'][:, 1:2]
    Z = random_draws[0] if random_draws is not None else torch.randn(B, m_eff, d, device=device, generator=rng)
    total_utility = torch.zeros((B, 1), device=device)

    # (선택) 예산 집행 하드캡: 한 스텝에 전부 소비 못하게
    ENFORCE_BUDGET = bool(int(os.getenv("PGDPO_ENFORCE_BUDGET", "1")))
    CAP_MULT_STR = os.getenv("PGDPO_C_SOFTCAP", "None")
    CAP_MULT = None if CAP_MULT_STR == "None" else float(CAP_MULT_STR)

    for i in range(m_eff):
        tau = states['TmT'] - i * dt  # time-to-maturity

        # 1) 정책 출력
        out = policy(**{'X': torch.cat([wealth, habit], dim=1), 'TmT': tau})
        u, C_pred = out[:, :d], out[:, d:]

        # 2) 래칫 강제 (C는 비감소)
        C = torch.maximum(C_pred, habit)
        # (선택) 소비 소프트캡: C ≤ max(Y, cap_mult*X)
        if CAP_MULT is not None:
            cap = torch.maximum(habit, CAP_MULT * wealth)
            C = torch.minimum(C, cap)
        # (선택) 하드 예산: 한 스텝에 X보다 더 못쓴다 (cap 시나리오와 동일한 느낌)
        if ENFORCE_BUDGET:
            C = torch.minimum(C, wealth / dt.clamp_min(1e-12))

        # 3) 소비 선차감 → 투자 가능한 현금
        investable = torch.clamp(wealth - C * dt, min=lb_X)

        # 4) 투자 수익 반영 (log-Euler)
        quad  = (u.unsqueeze(1) @ Sigma @ u.unsqueeze(-1)).squeeze(-1)           # (B,1)
        drift = r + (u * alpha).sum(1, keepdim=True) - 0.5 * quad
        dBX   = (u @ chol_S * Z[:, i, :]).sum(1, keepdim=True)
        wealth = torch.exp(torch.log(investable) + drift * dt + dBX * dt.sqrt()).clamp_min(lb_X)

        # 5) 습관 갱신 (C는 이미 ≥ habit 이므로 바로 대입해도 동일)
        habit = C

        # 6) 효용 적분
        if gamma != 1.0:
            period_utility = (C.clamp_min(1e-12).pow(1.0 - gamma) - 1.0) / (1.0 - gamma)
        else:
            period_utility = torch.log(C.clamp_min(1e-12))
        t_i = T - tau
        total_utility += torch.exp(-rho * t_i) * period_utility * dt

    # 말기 효용
    if gamma != 1.0:
        terminal_utility = (wealth.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)
    else:
        terminal_utility = torch.log(wealth)
    total_utility += torch.exp(torch.tensor(-rho * T, device=device)) * kappa * terminal_utility
    return total_utility.view(-1)



# --------------------------- (코어 호환) 무인자 CF 빌더 ---------------------------
ALPHA_RAISE = float(os.getenv("PGDPO_ALPHA_RAISE", "0.9"))  # BW 기준정책의 부분 상승 정도
C_SOFTCAP   = os.getenv("PGDPO_C_SOFTCAP", "None")
C_SOFTCAP   = None if C_SOFTCAP == "None" else float(C_SOFTCAP)

def build_closed_form_policy():
    return build_cf_with_args(
        alpha=alpha, Sigma=Sigma, gamma=gamma, L_cap=L_cap,
        rho=rho, r=r, T=T, rim_steps=RIM_STEPS, device=device,
        alpha_raise=ALPHA_RAISE, c_softcap=C_SOFTCAP
    )

# (선택) 프리로드
CF_POLICY, CF_INFO = (None, {})
if USE_CLOSED_FORM:
    try:
        CF_POLICY, CF_INFO = build_closed_form_policy()
        print(f"[CF] loaded={CF_POLICY is not None} | {CF_INFO}")
    except Exception as e:
        print(f"[WARN] CF preload failed: {e}")

# --------------------------- 시각화 스키마 ---------------------------
def get_traj_schema():
    u_labels = [f"u_{i+1}" for i in range(d)] + ["Consumption"]
    x_labels = ["Wealth (X)", "Ratchet Y"]
    return {
        "roles": {"X": {"dim": DIM_X, "labels": x_labels},
                  "U": {"dim": DIM_U, "labels": u_labels}},
        "views": [
            {"name": "Consumption_Path", "block": "U", "mode": "indices", "indices": [d], "ylabel": "Consumption (C)"},
            {"name": "State_Variables",  "block": "X", "mode": "indices", "indices": [0, 1], "ylabel": "State Value"},
        ],
        "sampling": {"Bmax": 5}
    }
