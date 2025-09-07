# tests/vpp/user_pgdpo_base.py
# VPP benchmark (based on the stable single-file script), split into base/policy/sim
# Exposes all symbols expected by core/pgdpo_base.py

import torch
import torch.nn as nn
import numpy as np

# ================== (A) Dimensions & global config ==================
d = 10                              # number of batteries (state dim)
k = 0                                   # no exogenous Y
DIM_X, DIM_Y, DIM_U = d, k, d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Required training hyperparams (exported for core) ----
epochs      = 500
batch_size  = 1024
lr          = 1e-4
seed        = 42          # used by core for seeding
CRN_SEED_EU = 777         # common random number seed for eval (exported)
N_eval_states = 2048      # eval batch size (exported)

# Apply module-level seed immediately
torch.manual_seed(seed); np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ================== (B) Problem parameters ==================
T = 1.0
m = 20                                 # N steps (dt = T/m)
dt_const = T / m

alpha_val = 0.3                         # R = alpha * I
R_matrix = torch.eye(d, device=device) * alpha_val

# Myopic analytical control tracks this net load
def N_agg(t_tensor: torch.Tensor) -> torch.Tensor:
    # shape: (B,1) -> (B,1)
    return 2.5 * torch.sin(2 * torch.pi * t_tensor / T)

# Control bounds (optional; used by our policy if soft-clip on)
u_min, u_max = -1.0, 1.0
use_soft_clip  = False
soft_k = 3.0

def soft_clip(u, lo=-1.0, hi=1.0, k=3.0):
    mid = 0.5 * (hi + lo); half = 0.5 * (hi - lo)
    return mid + half * torch.tanh(k * (u - mid))

def clip_u(u):
    return soft_clip(u, u_min, u_max, soft_k) if use_soft_clip else torch.clamp(u, u_min, u_max)

# ================== (C) One-factor correlated noise ==================
# Corr ≈ beta beta^T + diag(1 - beta^2); vols ~ U[0.3, 0.7]
beta_corr = torch.empty(d, device=device).uniform_(-0.8, 0.8)
vols_W    = torch.empty(d, device=device).uniform_(0.3, 0.7)
sqrt_idio = torch.sqrt(torch.clamp(1.0 - beta_corr**2, min=1e-8))

def draw_correlated_dW(B: int, dt_val: float, dtype=torch.float16):
    scale = torch.sqrt(torch.tensor(dt_val, device=device, dtype=dtype))
    Z0 = torch.randn(B, 1, device=device, dtype=dtype)   # common factor
    Zi = torch.randn(B, d, device=device, dtype=dtype)   # idiosyncratic
    dW = (beta_corr.to(dtype) * Z0 + sqrt_idio.to(dtype) * Zi) * (vols_W.to(dtype) * scale)
    return dW.to(torch.float32)

class DirectPolicy(nn.Module):
    """
    Pure neural network policy:
      input  : concat([X, TmT]) with shape (B, d+1)
      output : u in R^d with shape (B, d)
    No myopic baseline, no residual split, no clamp.
    """
    def __init__(self):
        super().__init__()
        state_dim = DIM_X + DIM_Y + 1   # = d + 1
        hidden = 64                     # 가볍고 d=1e4에도 무난한 폭

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, DIM_U)      # -> (B, d)
        )

        # 안정적인 시작을 위해 마지막 레이어는 0 근처에서 시작
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, **states_dict) -> torch.Tensor:
        X   = states_dict["X"]    # (B, d)
        TmT = states_dict["TmT"]  # (B, 1)
        x_in = torch.cat([X, TmT], dim=1)  # (B, d+1)
        u = self.net(x_in)                 # (B, d)
        return u

# ================== (E) Simulator (PG-DPO core calls this) ==================
def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    # X0 ~ U(0.2, 0.8), TmT0 ~ U(0, T)
    X0 = torch.rand((B, DIM_X), device=device, generator=rng) * 0.6 + 0.2
    TmT0 = torch.rand((B, 1), device=device, generator=rng) * T
    # core expects (states_dict, dt_vec)
    dt_vec = torch.full_like(TmT0, dt_const)  # constant dt = T/m
    return {'X': X0, 'TmT': TmT0}, dt_vec

def simulate(policy, B, *, train=True, rng=None, initial_states_dict=None, random_draws=None, m_steps=None):
    """
    Returns reward = - (running cost + terminal cost).
    Running cost per step: (N_agg(t) - sum u)^2 + alpha * ||u||^2.
    """
    m_eff = m_steps if m_steps is not None else m
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states, dt = initial_states_dict, torch.full_like(initial_states_dict['TmT'], T / float(m_eff))

    X = states['X']                               # (B,d)
    running_cost = torch.zeros(B, 1, device=device)

    for i in range(m_eff):
        # time bookkeeping consistent with prior tests: t = T - (TmT - i*dt)
        t_current = T - (states['TmT'] - i * dt)  # (B,1)
        current_states = {'X': X, 'TmT': states['TmT'] - i * dt}
        u = policy(**current_states)              # (B,d)

        Nagg = N_agg(t_current)                   # (B,1)
        agg_u = torch.sum(u, dim=1, keepdim=True) # (B,1)
        cost_step = (Nagg - agg_u).pow(2) + alpha_val * torch.sum(u**2, dim=1, keepdim=True)
        running_cost += cost_step * dt

        # SDE step (one-factor correlated noise)
        dW = draw_correlated_dW(B, float(dt_const))
        X = X - u * dt + dW                       # (B,d)
        # (필요 시 상태 clamp를 켤 수 있지만, 기본은 무제약)
        # X = torch.clamp(X, 0.0, 1.0)

    terminal_cost = torch.zeros_like(running_cost)  # no terminal penalty in this baseline
    total_cost = running_cost + terminal_cost
    return -total_cost

# ================== (F) Closed-form (myopic) policy for RMSE reference ==================
class AnalyticalMyopicPolicy(nn.Module):
    """u_analytical(t) = N_agg(t)/(alpha + d) * 1 (optionally clipped)."""
    def __init__(self):
        super().__init__()
    def forward(self, **states_dict):
        t = T - states_dict['TmT']
        u = (N_agg(t) / (alpha_val + d)).repeat(1, d)
        return clip_u(u)

def build_closed_form_policy():
    print("✅ Analytical VPP myopic policy loaded (based on stable script).")
    return AnalyticalMyopicPolicy().to(device), None

# ==========================================================================================

# (A) N_agg(t) 레퍼런스 신호 제공
def ref_signals_fn(t_np: np.ndarray) -> dict:
    # t_np: (steps,) numpy array
    # 여러분의 aggregate_net_load(t)를 그대로 사용
    import numpy as np
    # torch 없이 순수 numpy 버전 하나 둡니다 (또는 torch->numpy 변환)
    return {"Nagg": 2.5 * np.sin(2 * np.pi * t_np / float(T))}

# (B) R-메트릭 정보 제공 (R = alpha * I 가 기본)
R_INFO = {"alpha": float(alpha_val)}   # 일반 R을 쓰면 {"R_diag": diag_list} 또는 {"R": R_matrix}

def get_traj_schema():
    return {
        "roles": {
            "X": {"dim": int(DIM_X), "labels": [f"SoC_{i+1}" for i in range(int(DIM_X))]},
            "U": {"dim": int(DIM_U), "labels": [f"u_{i+1}"   for i in range(int(DIM_U))]},
        },
        "views": [
            {"name": "Tracking",    "mode": "tracking_vpp", "ylabel": "Power"},      # Nagg vs sum u
            {"name": "U_Rnorm",     "mode": "u_rnorm",      "ylabel": "||u||_R"},    # R-노름
        ],
        "sampling": {"Bmax": 5}
    }