# -*- coding: utf-8 -*-
# tests/mt_nd_retchet/user_pgdpo_base.py
# 목적:
#  - 심플/직관 버전: 신경망은 소비 '증분' dc만 예측하고 C = max(Y, Y+dc+margin)
#  - u는 토글로 무제약/제약(simplex) 선택 가능
#  - 시뮬레이터는 하드 랫쳇, 예산/소프트캡, GBM, CRRA 효용을 사용
#  - 코어 인터페이스: d,k,T,m,device,DIM_X,DIM_U,N_eval_states,CRN_SEED_EU,
#       sample_initial_states(), simulate(), build_closed_form_policy(), get_traj_schema()

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
d = int(os.getenv("PGDPO_D", 1))          # 위험자산 수
k = 0
T = float(os.getenv("PGDPO_T", 1.0))      # 만기
m = int(os.getenv("PGDPO_M", 20))         # 타임스텝
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIM_X = 2                                 # [Wealth X, Ratchet Y]
DIM_Y = k
DIM_U = d + 1                             # [u_1..u_d, C]
N_eval_states = int(os.getenv("PGDPO_EVAL_STATES", 120))
CRN_SEED_EU   = int(os.getenv("PGDPO_CRN", 71))

# (학습 하이퍼파라미터 — 코어가 참조)
epochs     = int(os.getenv("PGDPO_EPOCHS", 300))
batch_size = int(os.getenv("PGDPO_BS", 1024))
lr         = float(os.getenv("PGDPO_LR", 3e-4))
seed       = int(os.getenv("PGDPO_SEED", 24))

# --------------------------- 경제/실행 파라미터 ---------------------------
gamma = float(os.getenv("PGDPO_GAMMA", 3.0))
rho   = float(os.getenv("PGDPO_RHO",   0.04))
r     = float(os.getenv("PGDPO_RF",    0.01))
kappa = float(os.getenv("PGDPO_KAPPA", 1.0))

# 레버리지 캡(제약 모드에서 사용)
L_cap = float(os.getenv("PGDPO_LCAP", 2.0))

# 초기부/랫쳇
X0_range = (0.52, 0.56)
lb_X     = 1e-10
H0_MODE  = os.getenv("PGDPO_H0_MODE", "ratio").lower()  # ratio|random|zero|fixed
H0_RATIO = float(os.getenv("PGDPO_H0_RATIO", "0.02"))
H0_ABS   = float(os.getenv("PGDPO_H0_ABS",   "0.0"))

# ---- 소비 head: 'dc' 설계 파라미터 (간단/직관) ----
DC_GATE       = os.getenv("PGDPO_DC_GATE", "hard").lower()      # 'soft' | 'hard'
DC_MODE       = os.getenv("PGDPO_DC_MODE", "plain").lower()     # 'plain' | 'tau'
DC_FRAC       = float(os.getenv("PGDPO_DC_FRAC", "0.05"))       # dc의 기본 크기 비율(wealth 기준)
DC_TAU_BETA   = float(os.getenv("PGDPO_DC_TAU_BETA", "1.0"))    # DC_MODE='tau'에서 1/τ^beta
C_MARGIN_FRAC = float(os.getenv("PGDPO_C_JUMP_FRAC", "0.04"))   # 최소 점프= Y의 비율
C_MARGIN_ABS  = float(os.getenv("PGDPO_C_JUMP_ABS",  "0.0"))    # 절대 마진

# 실행 제약
ENFORCE_BUDGET = bool(int(os.getenv("PGDPO_ENFORCE_BUDGET", "0")))
C_SOFTCAP      = os.getenv("PGDPO_C_SOFTCAP", "None")
C_SOFTCAP      = None if C_SOFTCAP == "None" else float(C_SOFTCAP)

# u 모드 토글
UNCONSTRAINED_U = int(os.getenv("PGDPO_UNCONSTRAINED_U", "0"))  # 1=무제약
U_SCALE         = float(os.getenv("PGDPO_U_SCALE", "1.0"))      # 무제약일 때 스케일
U_BLEND_EPS     = float(os.getenv("PGDPO_U_BLEND_EPS", "0.15")) # 제약 모드에서 균등 바닥 혼합

# 확률 요동 강조
VOL_BOOST   = float(os.getenv("PGDPO_VOL_BOOST", "2.0"))

# --------------------------- 시장 파라미터 생성 ---------------------------
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
    vol_range=(0.18, 0.28), avg_corr=0.35, jitter=0.035, lam_factor=0.01,
    dirichlet_conc=8.0, target_leverage=None, alpha_mid=0.10,
    noise_std=0.01, hhi_factor=3.0
):
    if seed is not None:
        torch.manual_seed(seed)

    if d > 1:
        sigma = torch.empty(d, device=dev).uniform_(*vol_range)
        Psi = torch.full((d, d), float(avg_corr), device=dev); Psi.fill_diagonal_(1.0)
        if jitter > 0:
            N = torch.randn(d, d, device=dev) * jitter
            Psi = _nearest_spd_correlation(Psi + 0.5 * (N + N.T))
        Sigma = torch.diag(sigma) @ Psi @ torch.diag(sigma)
    else:
        s1 = float(torch.empty(1, device=dev).uniform_(*vol_range))
        Sigma = torch.tensor([[s1**2]], device=dev)

    Sigma = Sigma + lam_factor * Sigma.diag().mean() * torch.eye(d, device=dev)

    w_ref = torch.ones(d, device=dev) / d if d > 1 else torch.tensor([1.0], device=dev)
    if d > 1 and dirichlet_conc > 0:
        w_ref = torch.distributions.Dirichlet(torch.full((d,), float(dirichlet_conc), device=dev)).sample()
        hhi_target = hhi_factor / d
        if (w_ref**2).sum() > hhi_target:
            mix = 0.4; w_ref = (1 - mix) * w_ref + mix * (torch.ones(d, device=dev) / d)

    if target_leverage is not None:
        s = float(target_leverage)
    else:
        s = float(alpha_mid / (gamma * (Sigma @ w_ref).mean().clamp_min(1e-8)).item())

    alpha = gamma * s * (Sigma @ w_ref)
    if noise_std > 0:
        alpha += float(noise_std) * alpha.abs().mean() * torch.randn_like(alpha)
        alpha = alpha.clamp_min(1e-4)

    Sigma_inv = torch.linalg.inv(Sigma)
    return {"alpha": alpha, "Sigma": Sigma, "Sigma_inv": Sigma_inv}

params = _generate_mu_sigma_balanced(
    d, device, seed=seed,
    target_leverage=0.7 * L_cap,
    dirichlet_conc=8.0, hhi_factor=3.0
)
alpha, Sigma, Sigma_inv = params["alpha"], params["Sigma"], params["Sigma_inv"]
chol_S = torch.linalg.cholesky(Sigma) * VOL_BOOST

# --------------------------- 유틸 ---------------------------
def _annuity_factor(r: float, tau_t: torch.Tensor) -> torch.Tensor:
    if abs(r) < 1e-12:
        return tau_t.clamp_min(1e-12)
    return (1.0 - torch.exp(torch.tensor(-r, dtype=tau_t.dtype, device=tau_t.device) * tau_t)) / float(r)

# --------------------------- 정책 (dc 예측) ---------------------------
class DirectPolicy(nn.Module):
    """
    신경망은 '소비 증분 dc'를 예측:
      gate = σ(z) (soft) or 1{σ(z)>0.5} (hard)
      base = DC_FRAC * X         (DC_MODE='plain')
           = DC_FRAC * X / TmT^β (DC_MODE='tau')
      dc   = gate * clamp(base, 0, ∞)
      C    = max(Y + margin, Y + dc),  margin = C_MARGIN_ABS + C_MARGIN_FRAC * Y

    u는 두 모드:
      - 무제약(UNCONSTRAINED_U=1): u = U_SCALE * linear
      - 제약(기본): simplex(L_cap) 내부(softmax, d+1 logits → 앞 d개)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, d + 2)  # [u block, dc_logit] (제약모드: u logits d+1개, 무제약: u d개로 사용)
        )

    def forward(self, **states_dict):
        X   = states_dict["X"]
        W   = X[:, 0:1]           # wealth
        Y   = X[:, 1:2]           # ratchet level
        TmT = states_dict["TmT"]  # time-to-maturity

        s_in = torch.cat([W, Y, TmT], dim=1)
        raw  = self.net(s_in)

        # (1) 포트폴리오
        if UNCONSTRAINED_U:
            u_raw = raw[:, :d]
            u = U_SCALE * u_raw
        else:
            u_logits = raw[:, :d+1]
            u_smx    = F.softmax(u_logits, dim=1)[:, :d]
            if d > 0 and U_BLEND_EPS > 0:
                u_smx = (1.0 - U_BLEND_EPS) * u_smx + U_BLEND_EPS * (torch.ones_like(u_smx) / d)
            u = L_cap * u_smx

        # (2) dc 게이트 (soft/hard)
        dc_logit = raw[:, d+1:d+2] if not UNCONSTRAINED_U else raw[:, d:d+1]
        p_jump   = torch.sigmoid(dc_logit)
        gate     = (p_jump > 0.5).float() if DC_GATE == "hard" else p_jump

        # (3) 증분 base 크기
        if DC_MODE == "tau":
            base = DC_FRAC * W / TmT.clamp_min(1e-3).pow(DC_TAU_BETA)
        else:
            base = DC_FRAC * W

        dc = gate * torch.clamp(base, min=0.0)
        margin = C_MARGIN_ABS + C_MARGIN_FRAC * Y
        C = torch.maximum(Y + margin, Y + dc)

        return torch.cat([u, C], dim=1)

# --------------------------- 초기 상태/시뮬레이터 ---------------------------
def sample_initial_states(B, *, rng=None):
    wealth0 = torch.rand((B, 1), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]

    mode = H0_MODE
    if mode == "ratio":
        habit0 = H0_RATIO * wealth0
    elif mode == "zero":
        habit0 = torch.zeros_like(wealth0)
    elif mode == "fixed":
        habit0 = torch.full_like(wealth0, H0_ABS)
    else:  # "random"
        habit0 = torch.rand((B, 1), device=device, generator=rng) * wealth0

    habit0 = habit0.clamp_min(1e-12)

    X0   = torch.cat([wealth0, habit0], dim=1)
    TmT0 = torch.rand((B, 1), device=device, generator=rng) * T
    return {'X': X0, 'Y': None, 'TmT': TmT0}, TmT0 / float(m)

def simulate(policy, B, *, train=True, rng=None, initial_states_dict=None, random_draws=None, m_steps=None):
    """
    심플 시뮬레이터: 하드 랫쳇(habit = max(habit, C)), 예산/소프트캡, GBM, CRRA 효용.
    """
    import torch

    m_eff = m_steps or m

    # 초기 상태 & dt
    if initial_states_dict is None:
        states, dt_val = sample_initial_states(B, rng=rng)
        dt = (dt_val[0].item() if torch.is_tensor(dt_val) and dt_val.numel() > 0 else float(T) / float(m_eff))
    else:
        states = initial_states_dict
        dt = (states['TmT'][0] / float(m_eff)).item() if torch.is_tensor(states['TmT']) else float(T) / float(m_eff)

    wealth = states['X'][:, 0:1].clamp_min(lb_X).to(device)
    habit  = states['X'][:, 1:2].to(device)

    # 공통 난수
    if random_draws is not None:
        Z = random_draws[0].to(device)
    else:
        Z = torch.randn(B, m_eff, d, device=device, generator=rng) if d > 0 else torch.zeros(B, m_eff, 0, device=device)

    total_utility = torch.zeros((B, 1), device=device)

    for i in range(m_eff):
        tau = states['TmT'] - i * dt  # (B,1)

        # 정책 호출
        out = policy(**{'X': torch.cat([wealth, habit], dim=1), 'TmT': tau})
        u, C = out[:, :d], out[:, d:]

        # 실행 제약
        if C_SOFTCAP is not None:
            cap = torch.maximum(habit, C_SOFTCAP * wealth)
            C = torch.minimum(C, cap)
        if ENFORCE_BUDGET:
            C = torch.minimum(C, wealth / max(dt, 1e-12))

        # GBM 갱신 (소비 차감 후)
        investable = torch.clamp(wealth - C * dt, min=lb_X)

        if d > 0:
            quad  = (u.unsqueeze(1) @ Sigma @ u.unsqueeze(-1)).squeeze(-1)
            drift = r + (u * alpha).sum(1, keepdim=True) - 0.5 * quad
            dBX   = (u @ chol_S * Z[:, i, :]).sum(1, keepdim=True) * math.sqrt(dt)
        else:
            drift = torch.full_like(wealth, r)
            dBX   = torch.zeros_like(wealth)

        wealth = torch.exp(torch.log(investable) + drift * dt + dBX).clamp_min(lb_X)

        # 랫쳇(하드)
        habit = torch.maximum(habit, C)

        # 효용 적분
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

# --------------------------- (코어 호환) CF 빌더 ---------------------------
ALPHA_RAISE = float(os.getenv("PGDPO_ALPHA_RAISE", "1.0"))
RIM_STEPS   = int(os.getenv("PGDPO_RIM_STEPS", str(max(200, m))))
DW_FLOOR    = float(os.getenv("PGDPO_DW_FLOOR", "0.0"))

def build_closed_form_policy():
    return build_cf_with_args(
        alpha=alpha, Sigma=Sigma, gamma=gamma, L_cap=L_cap,
        rho=rho, r=r, T=T, rim_steps=RIM_STEPS, device=device,
        alpha_raise=ALPHA_RAISE, c_softcap=C_SOFTCAP, dw_floor=DW_FLOOR
    )

# (선택) 프리로드
CF_POLICY, CF_INFO = (None, {})
R_INFO = {}
try:
    CF_POLICY, CF_INFO = build_closed_form_policy()
    R_INFO = dict(CF_INFO) if isinstance(CF_INFO, dict) else {}
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
            {"name": "Consumption_Stairs",     "block": "U", "mode": "stairs",      "indices": [d], "ylabel": "C and C̄"},
            {"name": "Consumption_Amount",     "block": "U", "mode": "c_amount",    "indices": [d], "ylabel": "Consumption amount (C·dt)"},
            {"name": "Consumption_Cumulative", "block": "U", "mode": "c_cum",       "indices": [d], "ylabel": "Cumulative consumption (∑ C·dt)"},
            {"name": "Wealth_with_thresholds", "block": "X", "mode": "jedc_wealth", "indices": [0], "ylabel": "Wealth / thresholds"},
            {"name": "Risky_Share",            "block": "U", "mode": "risky_share", "indices": [0], "ylabel": "π/X"},
            {"name": "Discretionary_Wealth",   "block": "X", "mode": "dw",          "indices": [0], "ylabel": "DW = X - b(t,C̄)"},
            {"name": "X_log_returns",          "block": "X", "mode": "logret",      "indices": [0], "ylabel": "Δlog X"},
            {"name": "X_first_components",     "block": "X", "mode": "indices",     "indices": [0], "ylabel": "X[0] Component"},
            {"name": "U_first_components",     "block": "U", "mode": "indices",     "indices": [0], "ylabel": "U[0] Component"},
        ],
        "sampling": {"Bmax": 5},
    }