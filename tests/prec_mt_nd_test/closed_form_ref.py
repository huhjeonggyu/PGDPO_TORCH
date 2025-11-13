
# closed_form_ref.py — Pre-commitment MV closed-form (CIS, k=0), core-compatible
# ------------------------------------------------------------------------------
# π*(t,x) = G ( φ*(t) - x ),  φ*(t) = ( e^{ρT}/γ + x0 e^{rT} ) e^{-r τ}, τ=TmT,  ρ = α^T Σ^{-1} α.

from __future__ import annotations
import math
import torch
import torch.nn as nn

from user_pgdpo_base import device, DTYPE, T, r, x0, Sigma, Gd, alpha

def _rho_scalar() -> torch.Tensor:
    return (alpha * torch.cholesky_solve(alpha.view(-1,1), torch.linalg.cholesky(Sigma)).view(-1)).sum()

class ClosedFormPolicy(nn.Module):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("Gd_row", Gd.to(device, DTYPE), persistent=False)

    @torch.no_grad()
    def forward(self, **states_dict):
        X = states_dict["X"]                       # (B,1)
    
        # 1) 'TmT'가 있으면 그대로 사용
        if "TmT" in states_dict:
            TmT = states_dict["TmT"]              # (B,1)
        else:
            # 2) 't'만 넘어오는 경우(traj 경로): t는 정규화된 시간 t_norm = (T - TmT)/T
            if "t" not in states_dict:
                raise KeyError("closed_form_ref.forward: need 'TmT' or 't'")
            t_norm = states_dict["t"]
            # TmT = T - t_norm * T
            from user_pgdpo_base import T as _T
            TmT = (_T - t_norm * _T).to(device=X.device, dtype=X.dtype)
    
        # φ(t) = ( e^{ρT}/γ + x0 e^{rT} ) e^{-r·TmT}
        import math, torch
        rho = float((alpha @ torch.linalg.solve(Sigma, alpha)).item()) if Sigma.numel()>1 else float((alpha*alpha/Sigma).sum().item())
        phi0 = (math.exp(rho * T) / self.gamma) + (x0 * math.exp(r * T))
        phi  = torch.as_tensor(phi0, dtype=X.dtype, device=X.device).view(1,1) * torch.exp(- r * TmT)
    
        pi = (phi - X) @ self.Gd_row.to(X.device, X.dtype)   # (B,1)@(1,d)->(B,d)
        return pi

def build_closed_form_policy(params: dict, T_in: float, gamma: float):
    return ClosedFormPolicy(gamma), None
