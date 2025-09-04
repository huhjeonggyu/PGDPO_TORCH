# user_pgdpo_with_projection.py for SIR
import torch
from pgdpo_base import u_cap, B_cost

def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    JX = costates["JX"]
    X  = states["X"]
    regions = B_cost.shape[0]
    S = X[:,0:regions]; R = X[:,2*regions:3*regions]
    pS = JX[:,0:regions]; pR = JX[:,2*regions:3*regions]
    u = (S*(pS - pR))/B_cost.view(1,-1)
    return torch.clamp(u,0.0,u_cap)
