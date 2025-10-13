# user_pgdpo_base.py — Dollar-control pre-commitment MV (constant opportunity set)
# --------------------------------------------------------------------------------
# States: X (wealth), TmT (time-to-maturity τ = T - t). No factors (k=0).
# Control: π ∈ R^d  (DOLLAR exposure per asset).
# Dynamics:
#   dX_t = ( r X_t + π_t^T α ) dt + π_t^T Σ^{1/2} dW_t,   α = μ - r·1
# Objective (pre-commitment):
#   J_pc = E[X_T] - (γ/2) Var(X_T).
# Mini-batch unbiased estimator:
#   U_i = X_T_i - (γ/2) X_T_i^2 + (γ/2) (mean_batch X_T)^2
#
# Interfaces expected by core:
#   - sample_initial_states(B) -> (states_dict, dt_vector)
#   - simulate(policy, B, ...) -> per-sample U  (B,1)
#   - DirectPolicy (nn.Module): forward(X, TmT, t) -> π  (B,d)
#   - build_closed_form_policy() -> (policy_module, meta | None)
#
# Notes:
#   • This file uses X (not log X) and DOLLAR control to match PMP form
#       π*(t,x) = (Σ^{-1} α) · ( φ(t) - x ),  φ(t) = (e^{ρT}/γ + x0 e^{rT}) e^{-r(T - t)}.
#   • Set PGDPO_TAU_FIXED=1 (default) to train/eval on full horizon [0,T] only,
#     which matches the "student" recipe and stabilizes costate estimation.
#   • Expose CURRENT_POLICY and POLICY_VERSION so the projector can estimate φ once
#     per policy update without touching the core.
# --------------------------------------------------------------------------------

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

# --------------------------------- Dimensions ---------------------------------
d = int(os.getenv("PGDPO_D", 1))  # number of risky assets
k = 0                             # no factors

DIM_X = 1
DIM_Y = k
DIM_U = d

# -------------------------------- Device & Seeds -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed   = int(os.getenv("PGDPO_SEED", 777))
CRN_SEED_EU = int(os.getenv("PGDPO_CRN_SEED_EU", 202))

# ------------------------------ Market parameters ------------------------------
r     = float(os.getenv("PGDPO_R", 0.02))
gamma = float(os.getenv("PGDPO_GAMMA", 2.0))
x0    = float(os.getenv("PGDPO_X0", 1.0))   # anchor wealth at t=0 in φ(t)

# μ (not excess). Default: evenly spaced around r+0.06 ± 0.02
_mu_env = os.getenv("PGDPO_MU", "")
if _mu_env:
    mu_vals = [float(x.strip()) for x in _mu_env.split(",") if x.strip()]
    assert len(mu_vals) == d, "PGDPO_MU must provide d entries"
    mu = torch.tensor(mu_vals, device=device, dtype=torch.float32)
else:
    base_mu = r + 0.06
    offs = torch.linspace(-0.02, 0.02, d, device=device)
    mu = torch.full((d,), base_mu, device=device) + offs

# volatilities σ and correlation Ψ (diag by default)
_sig_env = os.getenv("PGDPO_SIGMA", "")
if _sig_env:
    sig_vals = [float(x.strip()) for x in _sig_env.split(",") if x.strip()]
    assert len(sig_vals) == d, "PGDPO_SIGMA must provide d entries"
    sigma_vec = torch.tensor(sig_vals, device=device, dtype=torch.float32)
else:
    sigma_vec = torch.full((d,), 0.20, device=device, dtype=torch.float32)
Psi = torch.eye(d, device=device, dtype=torch.float32)

Sigma = torch.diag(sigma_vec) @ Psi @ torch.diag(sigma_vec)    # (d,d) covariance
Sigma_chol = torch.linalg.cholesky(Sigma)                      # (d,d)
alpha = mu - r * torch.ones(d, device=device)                  # (d,)

# -------------------------------- Time grid -----------------------------------
T = float(os.getenv("PGDPO_T", 1.0))
# Lighter default for speed; override via env if needed.
m = int(os.getenv("PGDPO_M", 256))

# ------------------------------- Training knobs --------------------------------
epochs     = int(os.getenv("PGDPO_EPOCHS", 300))
batch_size = int(os.getenv("PGDPO_BATCH_SIZE", 1024))
lr         = float(os.getenv("PGDPO_LR", 3e-4))
# Smaller eval sample by default to avoid OOM backoff; core can override via env
N_eval_states = int(os.getenv("PGDPO_N_EVAL", 2000))

# ------------------------------ Ranges & clamps --------------------------------
X0_range = (x0, x0)                               # default anchor wealth
pi_cap   = float(os.getenv("PGDPO_PICAP", 2.0))   # clamp absolute dollars per asset

# ----------------------- Expose current policy to projector --------------------
CURRENT_POLICY = None
POLICY_VERSION = 0

# =============================================================================
# A. Sampler & Simulator (Euler–Maruyama on X, *dollar control*)
# =============================================================================

def sample_initial_states(B: int, *, rng: Optional[torch.Generator] = None) -> Tuple[dict, torch.Tensor]:
    # Anchor current wealth X uniformly in X0_range (default: x0)
    X0  = torch.rand((B, 1), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
    # Use full horizon by default (student recipe). Set PGDPO_TAU_FIXED=0 to randomize τ.
    if os.getenv("PGDPO_TAU_FIXED", "1") == "1":
        TmT = torch.full((B, 1), T, device=device)
    else:
        TmT = torch.rand((B, 1), device=device, generator=rng) * T
    dt_vec = TmT / float(m)
    states = {"X": X0, "TmT": TmT}
    return states, dt_vec

def simulate(policy: nn.Module, B: int, *, train: bool = True,
             rng: Optional[torch.Generator] = None,
             initial_states_dict: Optional[dict] = None,
             random_draws: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
             m_steps: Optional[int] = None) -> torch.Tensor:
    """
    Roll forward wealth with dollar-control policy and return per-sample score U_i.
    dX_t = ( r X_t + π_t^T α ) dt + (π_t^T Σ^{1/2}) dW_t
    """
    global CURRENT_POLICY, POLICY_VERSION
    # keep the most recent policy for projector's student estimator
    CURRENT_POLICY = policy
    POLICY_VERSION += 1

    m_eff = int(m_steps if m_steps is not None else m)
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states = initial_states_dict
        dt = states["TmT"] / float(m_eff)

    X = states["X"].clone()                                  # (B,1)

    for i in range(m_eff):
        tleft = states["TmT"] - i * dt                       # (B,1)
        # normalized time t ∈ [0,1] from 0→T
        t_norm = (T - tleft).clamp_min(0.0) / T              # (B,1)
        pi = policy(X=X, TmT=tleft, t=t_norm)                # (B,d) dollars
        drift = r * X + (pi * alpha).sum(dim=1, keepdim=True)
        noise_vec = (pi @ Sigma_chol)                        # (B,d)
        dW = torch.randn(X.shape[0], d, device=device, generator=rng)  # (B,d)
        dX = drift * dt + (noise_vec * dW).sum(dim=1, keepdim=True) * dt.sqrt()
        X = X + dX

    XT = X                                                   # (B,1), dollar control allows negatives
    mean_X = XT.mean()
    U = XT - 0.5 * gamma * (XT**2) + 0.5 * gamma * (mean_X**2)
    return U

# =============================================================================
# B. Policy Network (DirectPolicy)  — π(t, x) in dollars
# =============================================================================

class DirectPolicy(nn.Module):
    """Small MLP: inputs (t_norm, X) → output π ∈ R^d, clamped to ±pi_cap per asset."""
    def __init__(self):
        super().__init__()
        hidden = int(os.getenv("PGDPO_HIDDEN", 128))
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, d), nn.Tanh()
        )
        self.register_buffer("pi_cap", torch.tensor(float(pi_cap)))

    def forward(self, **states_dict):
        X   = states_dict["X"]
        t   = states_dict.get("t", (T - states_dict["TmT"]).clamp_min(0.0) / float(T))  # normalized time
        z   = torch.cat([t, X], dim=1)                                    # (B,2)
        raw = self.net(z)                                                 # (B,d) in [-1,1]
        return self.pi_cap * raw

# =============================================================================
# C. Closed-form Reference (Dollar PMP) — optional
# =============================================================================

def build_closed_form_policy():
    try:
        from closed_form_ref import build_closed_form_policy as _build
        params = {
            "d": d, "r": r, "mu": mu.detach().cpu().tolist(),
            "Sigma": (torch.diag(sigma_vec) @ Psi @ torch.diag(sigma_vec)).detach().cpu().numpy(),
            "pi_cap": pi_cap, "x0": x0
        }
        cf, meta = _build(params, T=T, gamma=gamma)
        return cf, meta
    except Exception as e:
        print(f"[WARN] closed_form_ref build failed: {e}")
        return None, {"note": "no closed-form"}

print(f"✅ mv_pc_const (dollar-control) loaded: d={{d}}, k={{k}}, T={{T}}, m={{m}}, r={{r:.3f}}, gamma={{gamma:.3f}}, x0={{x0:.3f}}")
