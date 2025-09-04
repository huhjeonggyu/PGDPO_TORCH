# tests/vpp/user_pgdpo_residual.py
# Myopic-only policy module (used as baseline and for residual learning base)

import torch
import torch.nn as nn
from user_pgdpo_base import d, N_agg, T, device, alpha_val, u_min, u_max, use_soft_clip, soft_k

def soft_clip(u, lo=-1.0, hi=1.0, k=3.0):
    mid = 0.5*(hi+lo); half = 0.5*(hi-lo)
    return mid + half*torch.tanh(k*(u-mid))

def _clip_u(u):
    return soft_clip(u, u_min, u_max, soft_k) if use_soft_clip else torch.clamp(u, u_min, u_max)

class MyopicPolicy(nn.Module):
    """
    u_myopic(t) = N_agg(t)/(alpha + d) * 1
    (Matches analytical policy from the stable single-file code.)
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("ones_d_T", torch.ones(1, d, device=device))
    def forward(self, **states_dict) -> torch.Tensor:
        t = T - states_dict['TmT']          # (B,1)
        u = (N_agg(t) / (alpha_val + d)) * self.ones_d_T
        return _clip_u(u)
