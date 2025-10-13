# user_pgdpo_base.py — ND, constant opportunity set, dollar-control (pre-commit MV)
# ---------------------------------------------------------------------------------
# * d-asset market, time-invariant (μ, Σ)
# * Dollar control π ∈ R^d on X-scale: dX = (rX + π^T α) dt + (π^T Σ^{1/2}) dW
# * Pre-commitment objective: J = E[X_T] - (γ/2) Var(X_T)
# * Mini-batch unbiased per-sample score: U_i = X_Ti - (γ/2) X_Ti^2 + (γ/2) (mean X_T)^2
# * Exposes CURRENT_POLICY / POLICY_VERSION so projector can estimate φ once per policy.
# * Exposes CRN_SEED_EU for the core (common random numbers in Euler).
#
# Market generator (no factor model; constant correlation):
#   - σ_i: either PGDPO_SIGMA list, or uniform in [VOL_MIN, VOL_MAX]
#   - Corr: equicorrelation with parameter ρ (PGDPO_RHO, default 0.30), SPD if ρ∈(-1/(d-1),1)
#   - Sharpe per-asset: SR_MODE=equal|normal, default equal with SR_MEAN=0.40 (SR_STD for normal)
#   - μ = r + α, α_i = SR_i * σ_i
#
# Key env toggles:
#   PGDPO_D                : number of risky assets (default 1)
#   PGDPO_RHO              : equicorrelation ρ (default 0.30)
#   PGDPO_VOL_MIN/MAX      : σ range if not provided (defaults 0.15 ~ 0.25)
#   PGDPO_SIGMA            : comma list of σ_i (overrides range)
#   PGDPO_SR_MODE          : 'equal' | 'normal' (default 'equal')
#   PGDPO_SR_MEAN, SR_STD  : Sharpe mean/std (default 0.40, 0.10)
#   PGDPO_TAU_FIXED        : 1 → always full horizon (default 1; student recipe)
#   PGDPO_PICAP            : per-asset dollar clamp (default 2.0)
#   PGDPO_PI_L2_CAP        : optional L2 clamp for π (per-sample) (default 0 → off)
# ---------------------------------------------------------------------------------

from __future__ import annotations
import os, math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# --------------------------------- Dimensions ---------------------------------
d = int(os.getenv("PGDPO_D", 1))
k = 0  # no extra factors in the state

DIM_X = 1
DIM_Y = k
DIM_U = d

# -------------------------------- Device & Seeds -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
torch.set_default_dtype(DTYPE)

# Core requests this for CRN in Euler:
CRN_SEED_EU   = int(os.getenv("PGDPO_CRN_SEED_EU", 202))
CRN_SEED_EVAL = int(os.getenv("PGDPO_CRN_SEED_EVAL", 303))

seed   = int(os.getenv("PGDPO_SEED", 777))
torch.manual_seed(seed)

# ------------------------------ Core parameters --------------------------------
r     = float(os.getenv("PGDPO_R", 0.02))
gamma = float(os.getenv("PGDPO_GAMMA", 2.0))
x0    = float(os.getenv("PGDPO_X0", 1.0))

T = float(os.getenv("PGDPO_T", 1.0))
m = int(os.getenv("PGDPO_M", 256))  # time steps

# Training / eval knobs
epochs        = int(os.getenv("PGDPO_EPOCHS", 300))
batch_size    = int(os.getenv("PGDPO_BATCH_SIZE", 2048))
lr            = float(os.getenv("PGDPO_LR", 3e-4))
N_eval_states = int(os.getenv("PGDPO_N_EVAL", 8192))

# Clamps
pi_cap    = float(os.getenv("PGDPO_PICAP", 2.0))      # per-asset clamp
pi_l2_cap = float(os.getenv("PGDPO_PI_L2_CAP", 0.0))  # 0 → off

X0_range = (x0, x0)  # default: fixed anchor

# ----------------------- Expose current policy to projector --------------------
CURRENT_POLICY = None
POLICY_VERSION = 0

# ============================= Market construction =============================

def _equicorr(d: int, rho: float) -> torch.Tensor:
    """Equicorrelation matrix with off-diagonal rho, diag=1. SPD if rho∈(-1/(d-1), 1)."""
    I = torch.eye(d, device=device, dtype=DTYPE)
    J = torch.ones((d, d), device=device, dtype=DTYPE)
    C = (1.0 - rho) * I + rho * J
    # numerical symmetrize
    return 0.5 * (C + C.T)

def _cholesky_with_ridge(S: torch.Tensor, max_tries: int = 8):
    lam = 0.0
    I = torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(S + lam * I)
            return L, lam
        except RuntimeError:
            lam = max(lam * 2.0, 1e-6) if lam > 0 else 1e-6
    L = torch.linalg.cholesky(S + (lam + 1e-3) * I)
    return L, lam + 1e-3

def _parse_list_env(name: str, n: int):
    s = os.getenv(name, "").strip()
    if not s:
        return None
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    assert len(vals) == n, f"{name} must have {n} entries"
    return torch.tensor(vals, device=device, dtype=DTYPE)

def _make_market():
    # σ
    sigma_env = _parse_list_env("PGDPO_SIGMA", d)
    if sigma_env is not None:
        sigma_vec = sigma_env.clamp_min(1e-6)
    else:
        vol_min = float(os.getenv("PGDPO_VOL_MIN", 0.15))
        vol_max = float(os.getenv("PGDPO_VOL_MAX", 0.25))
        sigma_vec = (vol_min + (vol_max - vol_min) * torch.rand(d, device=device, dtype=DTYPE)).clamp_min(1e-6)

    # Corr (equicorrelation)
    rho = float(os.getenv("PGDPO_RHO", 0.30))
    # SPD range: (-1/(d-1), 1); clip safely
    rho_min = -1.0 / (d - 1) + 1e-6 if d > 1 else -0.99
    rho = max(min(rho, 0.999), rho_min)
    C = _equicorr(d, rho)

    # Sharpe→α
    sr_mode = os.getenv("PGDPO_SR_MODE", "equal").lower()
    sr_mean = float(os.getenv("PGDPO_SR_MEAN", 0.40))
    sr_std  = float(os.getenv("PGDPO_SR_STD", 0.10))
    if sr_mode.startswith("equal"):
        sharpe = torch.full((d,), sr_mean, device=device, dtype=DTYPE)
    else:
        z = torch.randn(d, device=device, dtype=DTYPE)
        z = (z - z.mean()) / (z.std() + 1e-12)
        sharpe = sr_mean + sr_std * z

    alpha = sharpe * sigma_vec
    mu = r + alpha

    # Σ = Dσ C Dσ
    Sigma = torch.diag(sigma_vec) @ C @ torch.diag(sigma_vec)
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma_chol, ridge = _cholesky_with_ridge(Sigma)

    meta = {"rho": rho, "ridge": ridge, "sr_mode": sr_mode}
    return mu, alpha, Sigma, Sigma_chol, meta

# Build market once
mu, alpha, Sigma, Sigma_chol, _MARKET_META = _make_market()

# =============================================================================
# Sampler & Simulator (Euler–Maruyama on X, *dollar control*)
# =============================================================================

def sample_initial_states(B: int, *, rng: Optional[torch.Generator] = None) -> Tuple[dict, torch.Tensor]:
    X0  = torch.rand((B, 1), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
    if os.getenv("PGDPO_TAU_FIXED", "1") == "1":
        TmT = torch.full((B, 1), T, device=device)
    else:
        TmT = torch.rand((B, 1), device=device, generator=rng) * T
    dt_vec = TmT / float(m)
    states = {"X": X0, "TmT": TmT}
    return states, dt_vec

def _apply_l2_cap(pi: torch.Tensor, cap: float) -> torch.Tensor:
    if cap is None or cap <= 0: return pi
    norm = pi.norm(dim=1, keepdim=True) + 1e-12
    scale = torch.clamp(cap / norm, max=1.0)
    return pi * scale

def simulate(policy: nn.Module, B: int, *, train: bool = True,
             rng: Optional[torch.Generator] = None,
             initial_states_dict: Optional[dict] = None,
             random_draws: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
             m_steps: Optional[int] = None) -> torch.Tensor:
    """
    dX_t = ( r X_t + π_t^T α ) dt + (π_t^T Σ^{1/2}) dW_t
    Returns per-sample score U (B,1).
    """
    global CURRENT_POLICY, POLICY_VERSION
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
        t_norm = (T - tleft).clamp_min(0.0) / T              # (B,1)
        pi = policy(X=X, TmT=tleft, t=t_norm)                # (B,d) dollars
        # elementwise clamp + optional L2 cap
        if pi_cap > 0:
            pi = torch.clamp(pi, -pi_cap, pi_cap)
        if pi_l2_cap > 0:
            pi = _apply_l2_cap(pi, pi_l2_cap)

        drift = r * X + (pi * alpha).sum(dim=1, keepdim=True)   # (B,1)
        noise_vec = (pi @ Sigma_chol)                           # (B,d)
        dW = torch.randn(X.shape[0], d, device=device, generator=rng)
        dX = drift * dt + (noise_vec * dW).sum(dim=1, keepdim=True) * torch.sqrt(dt)
        X = X + dX

    XT = X
    mean_X = XT.mean()
    U = XT - 0.5 * gamma * (XT**2) + 0.5 * gamma * (mean_X**2)
    return U

# =============================================================================
# Policy Network — π(t, x) in dollars (ND)
# =============================================================================

class DirectPolicy(nn.Module):
    """MLP: inputs (t_norm, X) → π ∈ R^d, clamped to ±pi_cap elementwise, optional L2 cap."""
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
        t   = states_dict.get("t", (T - states_dict["TmT"]).clamp_min(0.0) / float(T))
        z   = torch.cat([t, X], dim=1)                                    # (B,2)
        raw = self.net(z)                                                 # (B,d) in [-1,1]
        out = self.pi_cap * raw
        if pi_l2_cap > 0:
            out = _apply_l2_cap(out, pi_l2_cap)
        return out

# =============================================================================
# Closed-form Reference (Dollar PMP) — optional
# =============================================================================

def _build_closed_form_local():
    """Fallback closed-form if module import fails."""
    import numpy as np
    class _CF(nn.Module):
        def __init__(self, r, mu_np, Sigma_np, gamma, T, x0, pi_cap):
            super().__init__()
            b = mu_np - r
            H = np.linalg.inv(Sigma_np)
            Gd = H @ b.reshape(-1,1)
            rho = float(b @ H @ b)
            self.register_buffer("Gd_row", torch.from_numpy(Gd.T.astype('float32')), persistent=False)
            self.r = float(r); self.gamma=float(gamma); self.T=float(T)
            self.rho = float(rho); self.x0 = float(x0); self.pi_cap=float(pi_cap)
        @torch.no_grad()
        def forward(self, **states_dict):
            X = states_dict["X"]; TmT = states_dict["TmT"]
            phi0 = (math.exp(self.rho * self.T) / self.gamma) + (self.x0 * math.exp(self.r * self.T))
            phi  = torch.as_tensor(phi0, dtype=X.dtype, device=X.device).view(1,1) * torch.exp(- self.r * TmT)
            pi   = (phi - X) @ self.Gd_row.to(X.device)
            if self.pi_cap and self.pi_cap > 0:
                pi = torch.clamp(pi, -self.pi_cap, self.pi_cap)
            return pi
    import numpy as np
    cf = _CF(r, mu.detach().cpu().numpy(), Sigma.detach().cpu().numpy(), gamma, T, x0, pi_cap).to(device)
    return cf, {"note":"fallback"}

def build_closed_form_policy():
    try:
        from closed_form_ref import build_closed_form_policy as _build
        params = {
            "d": d, "r": r, "mu": mu.detach().cpu().tolist(),
            "Sigma": Sigma.detach().cpu().numpy(),
            "pi_cap": pi_cap, "x0": x0
        }
        cf, meta = _build(params, T=T, gamma=gamma)
        return cf, meta
    except Exception as e:
        print(f"[WARN] closed_form_ref build failed: {e} — using fallback.")
        return _build_closed_form_local()

print(f"✅ mv_pc_const (ND, dollar-control) loaded: d={d}, T={T}, m={m}, r={r:.3f}, γ={gamma:.3f}, x0={x0:.3f}")
print(f"   Market: equicorr ρ={_MARKET_META['rho']:.3f}, ridge={_MARKET_META['ridge']:.1e}, SR={_MARKET_META['sr_mode']}")
