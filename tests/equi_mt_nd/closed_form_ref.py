# closed_form_ref.py — ND dollar closed-form (time-consistent equilibrium MV, constant opportunity set)
# ----------------------------------------------------------------------------------------
# π_eq^*(t) = ρ_eq(t) · G,   where  G = Σ^{-1} α,  α = μ - r·1,
# and  ρ_eq(t) = (1/γ) · e^{-∫_t^T r du}  (CIS → r constant ⇒ ρ_eq(t) = (1/γ) e^{-r (T - t)}).
# Wealth-independent; no intertemporal hedging term in k=0 (no factor) case.
#
# API: build_closed_form_policy(params, T, gamma) -> (nn.Module, None)
#      params keys: d, r, mu, Sigma (or sigma,Psi), pi_cap(optional), x0(optional)
# ----------------------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

__all__ = ["ClosedFormPolicy", "build_closed_form_policy"]

DTYPE = torch.float32

def _np(x):  # ensure numpy array (float32)
    a = np.asarray(x, dtype=np.float32)
    return a

def _build_Sigma(params):
    # Either a full Sigma or constructed from sigma (vols) and Psi (corr)
    if "Sigma" in params and params["Sigma"] is not None:
        S = _np(params["Sigma"])
        return S
    sig = _np(params.get("sigma", None))
    if sig is None:
        raise ValueError("Need Sigma or (sigma,Psi) in params")
    Psi = _np(params.get("Psi", np.eye(len(sig), dtype=np.float32)))
    D = np.diag(sig)
    return D @ Psi @ D

class ClosedFormPolicy(nn.Module):
    """Time-consistent equilibrium policy for CIS MV in dollar control."""
    def __init__(self, r: float, mu: np.ndarray, Sigma: np.ndarray, gamma: float, T: float, pi_cap: float = 0.0):
        super().__init__()
        mu = np.asarray(mu, dtype=np.float32)
        d = mu.shape[0]
        G = np.linalg.solve(Sigma.astype(np.float32), (mu - r*np.ones(d, dtype=np.float32)))
        self.register_buffer("G", torch.from_numpy(G))
        self.register_buffer("r", torch.tensor(float(r), dtype=torch.float32))
        self.register_buffer("gamma", torch.tensor(float(gamma), dtype=torch.float32))
        self.register_buffer("T", torch.tensor(float(T), dtype=torch.float32))
        self.register_buffer("pi_cap", torch.tensor(float(pi_cap), dtype=torch.float32))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # allow (t, x) or keywords {t_abs|t|TmT, X, ...}
        if len(args) == 2 and not kwargs:
            t_abs, _x = args
        else:
            t_abs = kwargs.get("t_abs", None)
            if t_abs is None:
                t_abs = kwargs.get("t", None)
            if t_abs is None and ("TmT" in kwargs):
                t_abs = self.T - kwargs["TmT"]
            if t_abs is None:
                any_tensor = None
                for v in kwargs.values():
                    if torch.is_tensor(v):
                        any_tensor = v
                        break
                B = any_tensor.size(0) if (any_tensor is not None and any_tensor.dim() > 0) else 1
                t_abs = torch.zeros((B, 1), device=self.G.device, dtype=self.G.dtype)

        if t_abs.dim() == 1:
            t_abs = t_abs.view(-1, 1)

        rho = (1.0/self.gamma) * torch.exp(- self.r * (self.T - t_abs))
        pi = rho * self.G.view(1, -1)
        if self.pi_cap.item() > 0.0:
            pi = torch.clamp(pi, -self.pi_cap, self.pi_cap)
        return pi

def build_closed_form_policy(params: dict, T: float, gamma: float):
    if params is None: params = {}
    if not isinstance(params, dict): raise TypeError("params must be dict")

    d = int(params.get("d", 1))
    r = float(params.get("r", 0.02))
    mu = _np(params.get("mu", np.ones(d, dtype=np.float32) * (r + 0.06)))
    Sigma = _build_Sigma(params)
    pi_cap = float(params.get("pi_cap", 0.0))

    pol = ClosedFormPolicy(r=r, mu=mu, Sigma=Sigma, gamma=gamma, T=T, pi_cap=pi_cap)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return pol.to(device), None