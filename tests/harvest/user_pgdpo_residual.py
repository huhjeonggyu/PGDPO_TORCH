# tests/harvest/user_pgdpo_residual.py for Harvesting
import torch, torch.nn as nn
from user_pgdpo_base import R_inv, u_cap, price

# ✨ FIX: 잔차 네트워크의 출력 크기를 조절하는 ResCap 변수 추가
ResCap = 1.0

class MyopicPolicy(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, **states_dict):
        X = states_dict["X"]
        u = (R_inv @ (X * price.view(1,-1)).unsqueeze(-1)).squeeze(-1)
        return torch.clamp(u, 0.0, u_cap)