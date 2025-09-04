# user_pgdpo_residual.py for Harvesting
import torch, torch.nn as nn
from pgdpo_base import R_inv, u_cap, price

class MyopicPolicy(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, **states_dict):
        X = states_dict["X"]
        u = (R_inv @ (X * price.view(1,-1)).unsqueeze(-1)).squeeze(-1)
        return torch.clamp(u, 0.0, u_cap)
