# tests/vpp/user_pgdpo_base.py
# VPP LQ Tracking (Paper Sec. 5.1) — user definitions for PG-DPO framework
# - State (per battery): x_i (SoC)
# - Control: u_i (charge + / discharge -)
# - SDE: dx_i = -u_i dt + sigma_i dW_i
# - Reward: P(t) * sum(u) - 0.5 u^T R u - 0.5 (x - x_tar)^T Q (x - x_tar)

import os
import math
import numpy as np
import torch
import torch.nn as nn

# ================== (A) Dimensions & global config ==================
# defaults (overridable via env)
d_default      = 10
epochs_default = 500
batch_default  = 1024
lr_default     = 1e-4
seed_default   = 42

# --- 모델별 설정 및 환경변수 오버라이드 블록 ---
d        = int(os.getenv("PGDPO_D", d_default))
k        = 0
epochs   = int(os.getenv("PGDPO_EPOCHS", epochs_default))
batch_size = int(os.getenv("PGDPO_BATCH_SIZE", batch_default))
lr       = float(os.getenv("PGDPO_LR", lr_default))
seed     = int(os.getenv("PGDPO_SEED", seed_default))
# --- 블록 끝 ---

DIM_X, DIM_Y, DIM_U = d, k, d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 공통 시드
CRN_SEED_EU   = 777            # eval common random numbers (코어와 호환)
N_eval_states = 2048           # 코어 평가 루틴이 기대하는 샘플 수

torch.manual_seed(seed); np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ================== (B) Problem parameters ==================
T       = 1.0
m       = 40
dt_const = T / m

# (1) Price model P(t)
def price_fn(t_tensor: torch.Tensor) -> torch.Tensor:
    # t in [0, T]
    # multi-harmonic day-ahead–like profile; clamp to strictly positive
    price = 25 * torch.sin(2 * torch.pi * t_tensor / T) \
          + 15 * torch.sin(4 * torch.pi * t_tensor / T) + 30
    return price.clamp_min(0.1)  # avoid degenerate division

# (2) Heterogeneous quadratic control costs R = diag(R_i)  (SPD)
R_diag = torch.empty(d, device=device).uniform_(0.2, 1.0)
R_matrix = torch.diag(R_diag)

# (3) Heterogeneous state-tracking penalties Q = diag(Q_i)  (SPD)
Q_diag = torch.empty(d, device=device).uniform_(0.5, 1.5)
Q_matrix = torch.diag(Q_diag)

# (4) SoC target (vector form; can be heterogeneous if desired)
x_target_scalar = 0.5
x_target_vec = torch.full((1, d), x_target_scalar, device=device)

# (5) Noise scales (control-independent diffusion)
vols_W = torch.empty(d, device=device).uniform_(0.3, 0.7)

# ================== (C) Helpers ==================
def _draw_dW_from_ZX(ZX_slice: torch.Tensor, dt_val: float) -> torch.Tensor:
    # ZX_slice: (B, d) ~ N(0,1)
    return ZX_slice * vols_W * math.sqrt(dt_val)

def draw_dW(B: int, dt_val: float, *, rng: torch.Generator | None = None) -> torch.Tensor:
    # fallback Gaussian
    Z = torch.randn(B, d, device=device, generator=rng)
    return _draw_dW_from_ZX(Z, dt_val)

# ================== (D) Policy (direct actor) ==================
class DirectPolicy(nn.Module):
    """Feed-forward policy u = pi(X, T - t)."""
    def __init__(self):
        super().__init__()
        state_dim = DIM_X + 1  # [X, TmT]
        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, DIM_U)
        )
        # start near-zero to ease warm-up stability
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, **states_dict) -> torch.Tensor:
        X   = states_dict["X"]   # (B,d)
        TmT = states_dict["TmT"] # (B,1) = T - t
        x_in = torch.cat([X, TmT], dim=1)
        u = self.net(x_in)
        return u

# ================== (E) Simulator ==================
def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    # Start SoC in [0.2, 0.8]; initial remaining time TmT ~ U[0,T]
    X0   = torch.rand((B, DIM_X), device=device, generator=rng) * 0.6 + 0.2
    TmT0 = torch.rand((B, 1), device=device, generator=rng) * T
    dt_vec = torch.full_like(TmT0, dt_const)
    return {'X': X0, 'TmT': TmT0}, dt_vec

@torch.enable_grad()
def simulate(
    policy: nn.Module,
    B: int,
    *,
    train: bool = True,
    rng: torch.Generator | None = None,
    initial_states_dict: dict | None = None,
    random_draws: tuple[torch.Tensor, torch.Tensor] | None = None,  # (ZX, ZY); VPP uses ZX only
    m_steps: int | None = None
):
    """
    Returns reward to be maximized:
        profit = ∫ [ P(t) * sum(u)
                     - 0.5 * u^T R u
                     - 0.5 * (x - x_tar)^T Q (x - x_tar) ] dt
    Compatible with core: if random_draws passed, uses ZX[:, step, :] as dW source.
    """
    m_eff = m_steps if m_steps is not None else m

    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states, dt = initial_states_dict, torch.full_like(initial_states_dict['TmT'], T / float(m_eff))

    X = states['X']           # (B,d)
    TmT0 = states['TmT']      # (B,1) = T - t0
    total_profit = torch.zeros(B, 1, device=device)

    # unpack ZX if provided
    ZX = None
    if random_draws is not None and isinstance(random_draws, tuple) and len(random_draws) >= 1:
        ZX = random_draws[0]  # (B, m_eff, d)

    for k_step in range(m_eff):
        # current time and remaining time
        t_k  = T - (TmT0 - k_step * dt)   # (B,1)
        TmTk = TmT0 - k_step * dt         # (B,1)

        # policy eval
        cur_states = {'X': X, 'TmT': TmTk}
        u = policy(**cur_states) if train else policy(**cur_states).detach()

        # reward components
        price  = price_fn(t_k)                               # (B,1)
        rev    = price * torch.sum(u, dim=1, keepdim=True)   # P(t)*1^T u
        quad_u = 0.5 * torch.einsum('bi,ij,bj->b', u, R_matrix, u).unsqueeze(1)
        x_err  = X - x_target_vec
        quad_x = 0.5 * torch.einsum('bi,ij,bj->b', x_err, Q_matrix, x_err).unsqueeze(1)

        profit_step = rev - quad_u - quad_x
        total_profit += profit_step * dt

        # SDE step: dx = -u dt + sigma dW
        if ZX is not None:
            dW = _draw_dW_from_ZX(ZX[:, k_step, :], float(dt_const))
        else:
            dW = draw_dW(B, float(dt_const), rng=rng)
        X = X - u * dt + dW
        # (optional) clamp SoC
        # X = X.clamp_(0.0, 1.0)

    return total_profit

# ================== (F) Closed-form reference (S, ψ) ==================
# 논문 5.1의 PMP/LQ 폐형해 참조정책 — RMSE 비교를 위해 사용
def build_closed_form_policy():
    from tests.vpp.closed_form_ref import build_closed_form_policy as _build_cf
    print("✅ Closed-form PMP policy (S, ψ 기반) loaded.")
    # 코어의 print_policy_rmse_and_samples_* 시그니처와 호환 (두 번째 리턴은 None)
    return _build_cf(), None

# ================== (G) Viz / misc ==================
def ref_signals_fn(t_np: np.ndarray) -> dict:
    # for overlaying price curve on plots
    price_t = price_fn(torch.from_numpy(t_np).float().to(device))
    return {"Price": price_t.cpu().numpy()}

R_INFO = {"R": R_matrix, "R_diag": R_diag, "Q": Q_matrix, "Q_diag": Q_diag}

def get_traj_schema():
    """
    Visualization schema consumed by core viz helpers.
    """
    return {
        "roles": {
            "X": {"dim": int(DIM_X), "labels": [f"SoC_{i+1}" for i in range(int(DIM_X))]},
            "U": {"dim": int(DIM_U), "labels": [f"u_{i+1}"   for i in range(int(DIM_U))]},
        },
        "views": [
            {"name": "Arbitrage_Strategy", "mode": "tracking_vpp", "ylabel": "Price / Power"},
            {"name": "U_Rnorm", "mode": "u_rnorm", "block": "U", "ylabel": "||u||_R"},
        ],
        "sampling": {"Bmax": 5}
    }