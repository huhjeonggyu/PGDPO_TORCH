# tests/harvest/user_pgdpo_base.py
# Harvesting (§5.4) — PG-DPO baseline (rng 호환 simulate, myopic을 참조정책으로 노출)

import math
import torch
import torch.nn as nn

# --------------------------- Device / seed ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 2025
torch.manual_seed(seed)

# --------------------------- Horizon / steps -------------------------
T = 2.0
m = 50  # steps
epochs = 200
lr = 3e-4
batch_size = 256
CRN_SEED_EU = 24680  # common random numbers for eval

# >>> 코어 RMSE 헬퍼가 기대하는 평가 샘플 수
N_eval_states = 1000

# --------------------------- Dimensions ------------------------------
N_species = 5
d = N_species         # state dim: X (biomass per species)
du = N_species        # control dim: harvesting rate per species

# --------------------------- Problem params --------------------------
# Prices (per unit biomass)
price = torch.empty(du, device=device).uniform_(0.8, 1.2)

# Quadratic trade cost R (diag SPD) and its inverse
R_diag = torch.empty(du, device=device).uniform_(0.1, 1.0)
R_mat  = torch.diag(R_diag)
R_inv  = torch.diag(1.0 / R_diag.clamp_min(1e-8))
R_inv_diag = 1.0 / R_diag.clamp_min(1e-8)  # handy

# State tracking (optional)
Qx_diag = torch.empty(d, device=device).uniform_(0.05, 0.2)
Qx = torch.diag(Qx_diag)

# Logistic growth and carrying capacity
r_g = torch.empty(d, device=device).uniform_(0.05, 0.15)   # growth rates
K   = torch.empty(d, device=device).uniform_(1.0, 2.0)     # capacities

# Interactions (symmetric, small)
Alpha = torch.randn(d, d, device=device) * 0.01
Alpha = 0.5 * (Alpha + Alpha.T)
Alpha.fill_diagonal_(0.0)

# Multiplicative noise coefficient per species
sigma_x = torch.empty(d, device=device).uniform_(0.03, 0.08)

# Bounds
u_cap = 2.0
lb_X  = 0.0
X_ref = 0.5 * K  # keep population near half capacity (optional term)

# --------------------------- RNG helpers -----------------------------
def make_generator(seed_val: int | None):
    g = torch.Generator(device=device)
    if seed_val is not None:
        g.manual_seed(int(seed_val))
    return g

def sample_TmT(B: int, *, rng: torch.Generator | None = None):
    return torch.rand((B,1), device=device, generator=rng).clamp_min(1e-6) * T

def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    # start near K/2 with some variability, strictly positive
    X0 = (0.5 * K + 0.2 * K * torch.randn(B, d, device=device, generator=rng)).clamp_min(1e-3)
    TmT0 = sample_TmT(B, rng=rng)
    return {"X": X0, "TmT": TmT0}, None

# --------------------------- Policy (residual) -----------------------
class DirectPolicy(nn.Module):
    """
    Residual on top of myopic: u_my = R^{-1} diag(X) price.
    Network learns bounded residual; final u is clamped to [0, u_cap].
    Input: [X, TmT] -> residual
    """
    def __init__(self):
        super().__init__()
        h = 256
        self.net = nn.Sequential(
            nn.Linear(d + 1, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, du),
        )
        # small init around 0 to not distort myopic too much initially
        for m_ in self.net.modules():
            if isinstance(m_, nn.Linear):
                nn.init.kaiming_uniform_(m_.weight, a=math.sqrt(5))
                if m_.bias is not None:
                    nn.init.zeros_(m_.bias)

    def forward(self, **states_dict):
        X   = states_dict["X"]
        TmT = states_dict["TmT"]
        z   = torch.cat([X, TmT], dim=1)
        residual = self.net(z)

        # myopic baseline: R^{-1} diag(X) price  =  (X * price) ⊙ R_inv_diag
        u_my = (X * price.view(1, -1)) * R_inv_diag.view(1, -1)
        u    = u_my + residual
        return torch.clamp(u, 0.0, u_cap)

# --------------------------- Myopic policy (참조정책) ----------------
class MyopicPolicy(nn.Module):
    """u_my = R^{-1} diag(X) price, clamped to [0, u_cap]."""
    def __init__(self):
        super().__init__()
    def forward(self, **states_dict):
        X = states_dict["X"]
        u_my = (X * price.view(1, -1)) * R_inv_diag.view(1, -1)
        return torch.clamp(u_my, 0.0, u_cap)

# --------------------------- Simulator --------------------------------
def simulate(policy: nn.Module,
             B: int,
             *,
             train: bool = True,
             rng: torch.Generator | None = None,
             initial_states_dict: dict | None = None,
             random_draws = None,
             m_steps: int | None = None):
    """
    Returns utility per path, shape (B,1): sum_t [ harvest - 0.5 u^T R u - 0.5 (X-Xref)^T Q (X-Xref) ] * dt
    Compatible with core: accepts rng=Generator and typical kwargs.
    """
    # turn grad on if training OR external X requires grad (for costates)
    need_grad = bool(train)
    if initial_states_dict is not None and "X" in initial_states_dict:
        need_grad = need_grad or bool(getattr(initial_states_dict["X"], "requires_grad", False))

    with torch.set_grad_enabled(need_grad):
        steps = int(m_steps or m)

        if initial_states_dict is None:
            g0 = rng if isinstance(rng, torch.Generator) else make_generator(seed if train else CRN_SEED_EU)
            states, _ = sample_initial_states(B, rng=g0)
            X   = states["X"]
            TmT = states["TmT"]
        else:
            # Use as-is to keep autograd path intact
            X   = initial_states_dict["X"]
            TmT = initial_states_dict["TmT"]
            if X.device != device or TmT.device != device:
                raise RuntimeError(f"initial_states_dict must be on {device}, got X:{X.device}, TmT:{TmT.device}")

        # per-sample dt via time-to-go (keeps per-sample horizons)
        dt = (TmT / float(steps)).view(-1, 1)

        # noise draws
        if random_draws is not None:
            ZX = random_draws[0] if isinstance(random_draws, tuple) else random_draws
            ZX = ZX.to(device)
        else:
            g = rng if isinstance(rng, torch.Generator) else make_generator(seed if train else CRN_SEED_EU)
            ZX = torch.randn(B, steps, d, device=device, generator=g)

        U_acc = torch.zeros((B,1), device=device)

        for i in range(steps):
            tau = TmT - i * dt
            u = policy(X=X, TmT=tau)  # [B,du], already clamped [0,u_cap]

            # reward pieces
            harvest = ((u * X) * price.view(1, -1)).sum(dim=1, keepdim=True)
            quad_u  = 0.5 * torch.einsum('bi,ij,bj->b', u, R_mat, u).unsqueeze(1)
            x_err   = X - X_ref.view(1, -1)
            quad_x  = 0.5 * torch.einsum('bi,ij,bj->b', x_err, Qx, x_err).unsqueeze(1)
            reward  = harvest - quad_u - quad_x
            U_acc   = U_acc + reward * dt

            # dynamics: logistic + interactions - harvest, plus multiplicative noise
            growth = r_g.view(1, -1) * X * (1.0 - X / K.view(1, -1))
            inter  = (X @ Alpha.T) * X     # Σ_j α_{ij} x_i x_j
            drift  = growth + inter - u * X

            dW = ZX[:, i, :] * torch.sqrt(dt)  # [B,d], broadcast sqrt(dt)
            X  = (X + drift * dt + sigma_x.view(1, -1) * X * dW).clamp(lb_X, None)

        return U_acc  # (B,1)

# --------------------------- "Closed-form" 대용 ----------------------
def build_closed_form_policy():
    """
    하베스팅에는 진짜 closed-form이 없으므로,
    코어의 비교 함수와 호환되도록 myopic 정책을 '참조 정책'으로 반환.
    """
    cf = MyopicPolicy()
    meta = {"note": "no true closed-form; using myopic as reference"}
    return cf, meta

# dummy vars for core compatibility (일부 러너가 import함)
gamma = 1.0
r = 0.0