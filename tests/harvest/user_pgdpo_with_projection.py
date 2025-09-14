# user_pgdpo_with_projection.py for Harvesting
import torch
from user_pgdpo_base import R_inv, u_cap, price

PP_NEEDS = ("JX",)

def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    JX = costates["JX"]
    X  = states["X"]
    target = X * (price.view(1,-1) - JX)
    u = (R_inv @ target.unsqueeze(-1)).squeeze(-1)
    return torch.clamp(u, 0.0, u_cap)
