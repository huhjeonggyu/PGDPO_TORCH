# user_pgdpo_residual.py for SIR
import torch, torch.nn as nn
from user_pgdpo_base import u_cap, B_cost

ResCap = 1.0

class MyopicPolicy(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, **states_dict):
        X=states_dict["X"]; regions=B_cost.shape[0]
        S=X[:,0:regions]
        u = S/B_cost.view(1,-1)*0.1
        return torch.clamp(u,0.0,u_cap)
