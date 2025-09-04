# tests/vpp/user_pgdpo_with_projection.py
# PMP projection for R = alpha I using Sherman–Morrison (same as stable script)

import torch
from user_pgdpo_base import alpha_val, d, N_agg, T, device, u_min, u_max, use_soft_clip, soft_k

def soft_clip(u, lo=-1.0, hi=1.0, k=3.0):
    mid = 0.5*(hi+lo); half = 0.5*(hi-lo)
    return mid + half*torch.tanh(k*(u-mid))

def _clip_u(u):
    return soft_clip(u, u_min, u_max, soft_k) if use_soft_clip else torch.clamp(u, u_min, u_max)

def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    u* = (alpha I + 11^T)^{-1} ( N(t)*1 + 0.5 * JX )
    Assumption: core provides JX with the same sign as costate p (empirically true in the stable script).
    Using Sherman–Morrison: (alpha I + 11^T)^{-1} = (1/alpha)(I - 11^T/(alpha + d)).
    """
    JX = costates['JX']      # (B,d), assumed ≈ p
    TmT = states['TmT']      # (B,1)
    t = T - TmT              # (B,1)

    w = N_agg(t).repeat(1, d) + 0.5 * JX            # (B,d)
    w_sum = w.sum(dim=1, keepdim=True)              # (B,1)

    alpha = alpha_val
    # (alpha I + 11^T)^{-1} w = (1/alpha)w - (1/alpha)*1 * (1^T w)/(alpha + d)
    u = (w / alpha) - (w_sum / (alpha * (alpha + d)))
    return _clip_u(u)
