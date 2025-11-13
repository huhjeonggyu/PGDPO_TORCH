
# user_pgdpo_base.py — Minimal λ-curvature P-PGDPO (pre-commitment, CIS, k=0) [FIXED λ, CORE-COMPAT]
# ---------------------------------------------------------------------------------------------------
# Exports the full interface expected by core/pgdpo_base.py:
#   device, T, m, d, k, DIM_X, DIM_Y, DIM_U,
#   epochs, batch_size, lr, CRN_SEED_EU, seed,
#   sample_initial_states, simulate, DirectPolicy, N_eval_states
#
# J_λ = E[X_T] - (γ/2)Var(X_T) - (λ_curv/2)E[X_T^2]
# Denominator (for projector): ∂_x Y_λ(t,t) = -λ_curv * Φ(t)^2, Φ(t)=exp(r*(T - t)).

from __future__ import annotations
import os, math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

# ---------------------------
# Device / dtype
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() and os.getenv("PGDPO_USE_CUDA","1")!="0" else "cpu")
DTYPE = torch.float32

# ---------------------------
# Globals (time/risk/hparam)
# ---------------------------
T: float = float(os.getenv("PGDPO_T", "1.0"))
m: int   = int(os.getenv("PGDPO_M", "256"))   # Euler steps
d: int   = int(os.getenv("PGDPO_D", "10"))    # risky assets
k: int   = 0                                  # factor dim (CIS)

DIM_X: int = 1
DIM_Y: int = k
DIM_U: int = d

epochs: int      = int(os.getenv("PGDPO_EPOCHS", "300"))
batch_size: int  = int(os.getenv("PGDPO_BATCH_SIZE", "2048"))
lr: float        = float(os.getenv("PGDPO_LR", "3e-4"))

# objective / params
gamma: float = float(os.getenv("PGDPO_GAMMA", "10.0"))
r: float     = float(os.getenv("PGDPO_R", "0.03"))
x0: float    = float(os.getenv("PGDPO_X0", "1.0"))

# fixed λ-curvature
lambda_curv: float = float(os.getenv("PGDPO_LAMBDA", "0.3"))

# caps
pi_cap: float     = float(os.getenv("PGDPO_PICAP", "2.0"))
pi_l2_cap: float  = float(os.getenv("PGDPO_PI_L2_CAP", "0.0"))

# seeds
seed: int         = int(os.getenv("PGDPO_SEED", "777"))
CRN_SEED_EU: int  = int(os.getenv("PGDPO_CRN_SEED_EU", "42"))
torch.manual_seed(seed)

# eval batch for viz
N_eval_states: int = int(os.getenv("PGDPO_N_EVAL", "8192"))

# ---------------------------
# Market (CIS, equicorr Σ)
# ---------------------------
def _parse_list_env(name: str, d: int) -> Optional[torch.Tensor]:
    s = os.getenv(name, "").strip()
    if not s:
        return None
    vals = [float(x) for x in s.replace(",", " ").split() if x]
    t = torch.tensor(vals, device=device, dtype=DTYPE)
    if t.numel() == 1: t = t.repeat(d)
    assert t.numel() == d, f"{name} length mismatch: expected {d}, got {t.numel()}"
    return t

def _equicorr(d: int, rho: float) -> torch.Tensor:
    I = torch.eye(d, device=device, dtype=DTYPE)
    J = torch.ones((d,d), device=device, dtype=DTYPE)
    C = (1.0 - rho) * I + rho * J
    return 0.5*(C+C.T)

def _cholesky_with_ridge(S: torch.Tensor, max_tries: int = 8):
    lam = 0.0
    I = torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(S + lam * I)
            return L, lam
        except RuntimeError:
            lam = 1e-8 if lam == 0.0 else lam * 10.0
    raise RuntimeError("Cholesky failed even with ridge.")

def _make_market() -> Dict[str, torch.Tensor]:
    # vols
    sigma_env = _parse_list_env("PGDPO_SIGMA", d)
    if sigma_env is None:
        vols = torch.empty(d, device=device, dtype=DTYPE).uniform_(0.15, 0.35)
    else:
        vols = sigma_env
    # equi-corr SPD
    rho = float(os.getenv("PGDPO_RHO", "0.30"))
    rho = max(min(rho, 0.999), -1.0/(d-1) + 1e-6)
    Psi = _equicorr(d, rho)
    D = torch.diag(vols)
    Sigma = D @ Psi @ D
    # Sharpe
    sr_mode = os.getenv("PGDPO_SR_MODE", "equal").strip().lower()
    sr_mean = float(os.getenv("PGDPO_SR_MEAN", "0.40"))
    sr_std  = float(os.getenv("PGDPO_SR_STD",  "0.00"))
    if sr_mode == "equal":
        SR = torch.full((d,), sr_mean, device=device, dtype=DTYPE)
    else:
        SR = torch.normal(sr_mean, sr_std, size=(d,), device=device, dtype=DTYPE)
    alpha = SR * vols
    mu = r + alpha
    return dict(Sigma=Sigma, mu=mu, alpha=alpha, Psi=Psi, vols=vols)

mkt = _make_market()
Sigma: torch.Tensor = mkt["Sigma"]
alpha: torch.Tensor = mkt["alpha"]
Sigma_chol, _ = _cholesky_with_ridge(Sigma)
# G = Σ^{-1} α
G: torch.Tensor = torch.cholesky_solve(alpha.view(-1,1), Sigma_chol).view(-1)
Gd: torch.Tensor = G.view(1,-1)  # (1,d)

# ---------------------------
# Helpers
# ---------------------------
def Phi_of_tau(tau: torch.Tensor) -> torch.Tensor:
    return torch.exp(r * tau)

X0_range = (x0, x0)  # fixed X0; can widen if desired

def sample_initial_states(B: int, *, rng: Optional[torch.Generator] = None) -> Tuple[dict, torch.Tensor]:
    # states: {"X": X0, "TmT": τ}
    X0 = torch.rand((B,1), device=device, generator=rng) * (X0_range[1] - X0_range[0]) + X0_range[0]
    if os.getenv("PGDPO_TAU_FIXED","1") == "1":
        TmT = torch.full((B,1), T, device=device)
    else:
        TmT = torch.rand((B,1), device=device, generator=rng) * T
    dt_vec = TmT / float(m)
    states = {"X": X0, "TmT": TmT}
    return states, dt_vec

def _apply_l2_cap(pi: torch.Tensor, cap: float) -> torch.Tensor:
    if cap is None or cap <= 0: return pi
    norm = pi.norm(dim=1, keepdim=True) + 1e-12
    scale = torch.clamp(cap / norm, max=1.0)
    return pi * scale

# ---------------------------
# Baseline simulate (for --base / viz compatibility)
# ---------------------------
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

    # --- init states (+ t-only fallback) ---
    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
        X   = states["X"].clone()
        TmT = states["TmT"].clone()
    else:
        s = initial_states_dict
        X = s["X"].clone()
        if "TmT" in s:
            TmT = s["TmT"].clone()
        else:
            # t가 정규화(t_norm = (T - TmT)/T)되어 들어옴
            t_norm = s.get("t", torch.zeros_like(X))
            TmT = (T - t_norm * T).to(device=X.device, dtype=X.dtype)
        dt = TmT / float(m_eff)

    # --- rollout ---
    for i in range(m_eff):
        t_norm = (T - TmT).clamp_min(0.0) / float(T)
        pi = policy(t=t_norm, X=X)                          # (B,d)
        if pi_cap and pi_cap > 0:
            pi = torch.clamp(pi, -pi_cap, pi_cap)
        if pi_l2_cap and pi_l2_cap > 0:
            pi = _apply_l2_cap(pi, pi_l2_cap)

        drift = r * X + (pi * alpha.view(1,-1)).sum(dim=1, keepdim=True)
        noise_vec = (pi @ Sigma_chol)                       # (B,d)

        if random_draws is None:
            dW = torch.randn(B, d, device=device, generator=rng)     # (B,d)
        else:
            ZX, _ = random_draws                                      # ZX: (B,m,d)
            dW = ZX[:, i, :]                                          # (B,d)

        X   = X   + drift * dt + (noise_vec * dW).sum(dim=1, keepdim=True) * torch.sqrt(dt)
        TmT = TmT - dt

    XT = X
    M  = XT.mean()
    # 기존:
    # U  = XT - 0.5 * gamma * (XT**2) + 0.5 * gamma * (M**2)
    # 수정(λ 포함):
    U  = XT - 0.5 * (gamma + lambda_curv) * (XT**2) + 0.5 * gamma * (M**2)
    return U


# ---------------------------
# Baseline policy (for --base compatibility)
# ---------------------------
class DirectPolicy(nn.Module):
    """MLP: inputs (t_norm, X) → π ∈ R^d; with elementwise cap and optional L2 cap."""
    def __init__(self):
        super().__init__()
        hidden = int(os.getenv("PGDPO_HIDDEN", "128"))
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, d), nn.Tanh()
        )
        self.register_buffer("pi_cap", torch.tensor(float(pi_cap)))

    def forward(self, **states_dict):
        X = states_dict["X"]
        if "t" in states_dict:
            t = states_dict["t"]
        else:
            t = (T - states_dict["TmT"]).clamp_min(0.0) / float(T)
        z = torch.cat([t, X], dim=1)
        pi = self.net(z)
        if self.pi_cap and self.pi_cap > 0: pi = torch.clamp(pi, -self.pi_cap, self.pi_cap)
        if pi_l2_cap and pi_l2_cap > 0: pi = _apply_l2_cap(pi, pi_l2_cap)
        return pi

# ---------------------------
# Minimal CURRENT_POLICY exposure (for projector costates)
# ---------------------------
class _RhoNet(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for m_ in self.net.modules():
            if isinstance(m_, nn.Linear):
                nn.init.xavier_uniform_(m_.weight, gain=0.5)
                nn.init.zeros_(m_.bias)
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t,x], dim=1))

class _CurrentPolicy(nn.Module):
    def __init__(self): super().__init__(); self.rho = _RhoNet()
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.rho(t,x) @ Gd.to(x.device, x.dtype)

CURRENT_POLICY: nn.Module = _CurrentPolicy().to(device)
POLICY_VERSION: int = 0

def build_closed_form_policy():
    import importlib, numpy as np, torch
    # 같은 폴더의 closed_form_ref 모듈을 불러와 래핑
    cf = importlib.import_module("closed_form_ref")
    params = {
        "d": int(d),
        "r": float(r),
        "mu": (r + alpha).detach().cpu().numpy(),               # μ = r·1 + α
        "Sigma": Sigma.detach().cpu().numpy(),
        "pi_cap": float(pi_cap),
        "x0": float(x0),
    }
    policy, meta = cf.build_closed_form_policy(params, float(T), float(gamma))
    return policy, meta 

__all__ = [
    "device", "DTYPE",
    "T", "m", "d", "k",
    "DIM_X", "DIM_Y", "DIM_U",
    "epochs", "batch_size", "lr",
    "CRN_SEED_EU", "seed",
    "sample_initial_states", "simulate",
    "DirectPolicy",
    "N_eval_states",
    # extras used by projector/closed-form
    "gamma", "r", "x0", "lambda_curv",
    "Sigma", "Sigma_chol", "alpha", "G", "Gd", "Phi_of_tau",
    "pi_cap", "pi_l2_cap",
    "CURRENT_POLICY", "POLICY_VERSION",
]
