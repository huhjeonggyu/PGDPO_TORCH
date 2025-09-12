# tests/vpp/user_pgdpo_base.py
# VPP LQ Tracking (Paper Sec. 5.1) — user definitions for PG-DPO framework
# - State (per battery): x_i (SoC)
# - Control: u_i (charge + / discharge -)
# - SDE: dx_i = -u_i dt + sigma_i dW_i
# - Reward: P(t) * sum(u) - 0.5 u^T R u  <- ✨ 상태 추적 페널티 제거됨

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
lr_default     = 1e-3
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

CRN_SEED_EU   = 777
N_eval_states = 2048

torch.manual_seed(seed); np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ================== (B) Problem parameters ==================
T       = 1.0
m       = 40
dt_const = T / m

def price_fn(t_tensor: torch.Tensor) -> torch.Tensor:
    price = 25 * torch.sin(2 * torch.pi * t_tensor / T) \
          + 15 * torch.sin(4 * torch.pi * t_tensor / T) + 30
    return price.clamp_min(0.1)

R_diag = torch.empty(d, device=device).uniform_(0.2, 1.0)
R_matrix = torch.diag(R_diag)

# ✨✨✨ 수정된 부분: Q와 x_target을 0으로 설정하여 페널티 제거 ✨✨✨
Q_diag = torch.zeros(d, device=device) # 0으로 변경
Q_matrix = torch.diag(Q_diag)
x_target_vec = torch.zeros((1, d), device=device) # 0으로 변경
# ✨✨✨ 여기까지 수정 ✨✨✨

vols_W = torch.empty(d, device=device).uniform_(0.3, 0.7)

# ================== (C) Helpers (변경 없음) ==================
def _draw_dW_from_ZX(ZX_slice: torch.Tensor, dt_val: float) -> torch.Tensor:
    return ZX_slice * vols_W * math.sqrt(dt_val)
def draw_dW(B: int, dt_val: float, *, rng: torch.Generator | None = None) -> torch.Tensor:
    Z = torch.randn(B, d, device=device, generator=rng)
    return _draw_dW_from_ZX(Z, dt_val)

# ================== (D) Policy (변경 없음) ==================
class DirectPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        state_dim = DIM_X + 1
        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, DIM_U)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, **states_dict) -> torch.Tensor:
        X   = states_dict["X"]
        TmT = states_dict["TmT"]
        x_in = torch.cat([X, TmT], dim=1)
        u = self.net(x_in)
        return u

# ================== (E) Simulator (변경 없음) ==================
def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    X0   = torch.rand((B, DIM_X), device=device, generator=rng) * 0.6 + 0.2
    TmT0 = torch.rand((B, 1),      device=device, generator=rng) * T
    dt_vec = TmT0 / float(m)
    return {'X': X0, 'TmT': TmT0}, dt_vec

@torch.enable_grad()
def simulate(
    policy: nn.Module,
    B: int,
    *,
    train: bool = True,
    rng: torch.Generator | None = None,
    initial_states_dict: dict | None = None,
    random_draws: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
    m_steps: int | None = None
):
    m_eff = int(m_steps) if m_steps is not None else int(m)
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
        dt = dt * (m / float(m_eff))
    else:
        states, dt = initial_states_dict, initial_states_dict['TmT'] / float(m_eff)
    X    = states['X']
    TmT0 = states['TmT']
    total_profit = torch.zeros(B, 1, device=device)
    ZX = None
    if random_draws is not None:
        ZX = random_draws[0] if isinstance(random_draws, tuple) else random_draws
    for k_step in range(m_eff):
        t_k  = T - (TmT0 - k_step * dt)
        TmTk = TmT0 - k_step * dt
        cur_states = {'X': X, 'TmT': TmTk}
        u = policy(**cur_states) if train else policy(**cur_states).detach()

        # ✨✨✨ 수정된 부분: quad_x 계산 제거 ✨✨✨
        price  = price_fn(t_k)
        rev    = price * torch.sum(u, dim=1, keepdim=True)
        quad_u = 0.5 * torch.einsum('bi,ij,bj->b', u, R_matrix, u).unsqueeze(1)
        # x_err, quad_x 항을 계산에서 제외
        total_profit += (rev - quad_u) * dt
        # ✨✨✨ 여기까지 수정 ✨✨✨
        
        if ZX is not None:
            dW = ZX[:, k_step, :] * vols_W.view(1, -1) * torch.sqrt(dt)
        else:
            Z  = torch.randn(B, DIM_X, device=device, generator=rng)
            dW = Z * vols_W.view(1, -1) * torch.sqrt(dt)
        X = X - u * dt + dW
    return total_profit

# ================== (F) Closed-form reference (변경 없음) ==================
def build_closed_form_policy():
    from tests.vpp.closed_form_ref import build_closed_form_policy as _build_cf
    print("✅ Closed-form PMP policy (S, ψ 기반) loaded.")
    return _build_cf(), None

# ================== (G) Viz / misc (변경 없음) ==================
def ref_signals_fn(t_np: np.ndarray) -> dict:
    price_t = price_fn(torch.from_numpy(t_np).float().to(device))
    return {"Price": price_t.cpu().numpy()}
R_INFO = {"R": R_matrix, "R_diag": R_diag, "Q": Q_matrix, "Q_diag": Q_diag}
def get_traj_schema():
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