# closed_form_ref.py — ND dollar closed-form (pre-commitment MV, constant opportunity set)
# ----------------------------------------------------------------------------------------
# π*(t,x) = (Σ^{-1} α) · ( φ(t) - x ),    α = μ - r·1,
# φ(t)   = ( e^{ρT}/γ + x0 e^{rT} ) e^{-r (T - t)},   ρ = α^T Σ^{-1} α.
#
# API: build_closed_form_policy(params, T, gamma) -> (nn.Module, None)
#       params keys: d, r, mu, Sigma (or sigma,Psi), pi_cap(optional), x0(optional)
# ----------------------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

__all__ = ["ClosedFormPolicy", "build_closed_form_policy"]

def _np(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)

def _build_Sigma(params: dict) -> np.ndarray:
    if "Sigma" in params and params["Sigma"] is not None:
        return _np(params["Sigma"])
    sig = _np(params.get("sigma", None)).reshape(-1)
    d = sig.shape[0]
    Psi = _np(params.get("Psi", np.eye(d, dtype=np.float32))).reshape(d, d)
    return np.diag(sig) @ Psi @ np.diag(sig)

class ClosedFormPolicy(nn.Module):
    """Return dollar control π(t,x) as (B,d)."""
    def __init__(self, *, r: float, mu: np.ndarray, Sigma: np.ndarray,
                 gamma: float, T: float, x0: float = 1.0, pi_cap: float = 0.0):
        super().__init__()
        mu = _np(mu).reshape(-1)
        Sigma = _np(Sigma).reshape(len(mu), len(mu))
        b = mu - float(r) * np.ones_like(mu)
        H = np.linalg.inv(Sigma)
        Gd = H @ b.reshape(-1, 1)        # (d,1)
        rho = float(b @ H @ b)           # α^T Σ^{-1} α

        self.register_buffer("Gd_row", torch.from_numpy(Gd.T.astype(np.float32)), persistent=False)
        self.r = float(r); self.gamma = float(gamma); self.T = float(T); self.rho = float(rho)
        self.x0 = float(x0)
        self.pi_cap = float(pi_cap)

    @torch.no_grad()
    def forward(self, **states_dict):
        X = states_dict["X"]                          # (B,1)
        TmT = states_dict["TmT"]                      # (B,1)  τ = T - t
        dev, dt = X.device, X.dtype
        # φ(t) = ( e^{ρT}/γ + x0 e^{rT} ) e^{-r τ}
        phi0 = (np.exp(self.rho * self.T) / self.gamma) + (self.x0 * np.exp(self.r * self.T))
        phi  = torch.as_tensor(phi0, dtype=dt, device=dev).view(1,1) * torch.exp(- self.r * TmT)
        pi   = (phi - X) @ self.Gd_row.to(dev)        # (B,1)@(1,d)->(B,d)
        if self.pi_cap and self.pi_cap > 0:
            pi = torch.clamp(pi, -self.pi_cap, self.pi_cap)
        return pi

def build_closed_form_policy(params: dict | None = None,
                             T: float = 1.0,
                             gamma: float = 2.0):
    if params is None:
        d = 1; r = 0.02
        mu = np.array([0.08], dtype=np.float32)
        sig = np.array([0.20], dtype=np.float32)
        Sigma = np.diag(sig) @ np.diag(sig)
        pi_cap = 0.0; x0 = 1.0
    else:
        d = int(params.get("d", 1))
        r = float(params.get("r", 0.02))
        mu = _np(params.get("mu", np.ones(d, dtype=np.float32) * (r + 0.06)))
        Sigma = _build_Sigma(params)
        pi_cap = float(params.get("pi_cap", 0.0))
        x0 = float(params.get("x0", 1.0))

    pol = ClosedFormPolicy(r=r, mu=mu, Sigma=Sigma, gamma=gamma, T=T, x0=x0, pi_cap=pi_cap)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return pol.to(device), None
