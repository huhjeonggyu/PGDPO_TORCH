# user_pgdpo_residual.py
# 역할: 잔차 학습(Residual Learning)의 베이스라인 Myopic 정책 (클램프 없음)
#  u(Y) = (1/γ) · solve(Σ, (μ(Y) - r)),  (μ - r) = F.linear(Y, α) ⊙ σ
#  - Σ^{-1} 사용 금지. user_pgdpo_with_projection의 _sigma_solve 재사용.

import torch
import torch.nn as nn
import torch.nn.functional as F

# 모델 파라미터는 user_pgdpo_base(정본)에서 로드
from user_pgdpo_base import (
    alpha,   # (d, k)
    sigma,   # (d,)
    gamma,   # scalar (tensor 또는 float)
)

# 동일 디렉터리의 선형계 풀이 유틸을 재사용
try:
    from user_pgdpo_with_projection import _sigma_solve as sigma_solve
except Exception as e:
    raise ImportError(
        "user_pgdpo_residual: user_pgdpo_with_projection._sigma_solve 를 찾을 수 없습니다. "
        "같은 디렉터리에 두거나 PYTHONPATH를 조정하세요."
    ) from e

ResCap = 1.

class MyopicPolicy(nn.Module):
    """
    베이스라인 마이오픽 정책:
      u(Y) = (1/γ) · argmin_u ||Σ^{1/2}u - (μ - r)||_2  =  (1/γ)·solve(Σ, μ - r)
    - 학습 파라미터/클램프 없음.
    - solve는 user_pgdpo_with_projection._sigma_solve의 캐시/폴백 로직을 사용.
    """
    def __init__(self, lam: float = 1e-6):
        super().__init__()
        self.lam = float(lam)  # _sigma_solve에 전달될 릿지 강도
        inv = (1.0 / gamma) if isinstance(gamma, torch.Tensor) else torch.tensor(1.0 / float(gamma))
        self.register_buffer("_inv_gamma", inv)

    def forward(self, **states_dict) -> torch.Tensor:
        Y = states_dict.get("Y")
        if Y is None:
            raise ValueError("MyopicPolicy requires states['Y'] (Y must exist).")

        device, dtype = Y.device, Y.dtype
        inv_gamma = self._inv_gamma.to(device=device, dtype=dtype)

        # (μ - r) = F.linear(Y, α) ⊙ σ  ;  F.linear(Y, α) = Y @ α^T → (B, d)
        mu_minus_r = F.linear(Y, alpha.to(device=device, dtype=dtype)) \
                     * sigma.to(device=device, dtype=dtype)            # (B, d)

        # Σ ū = (μ - r)  →  ū = solve(Σ, μ - r)  (with_projection 캐시/폴백 경로 공유)
        u_bar = sigma_solve(mu_minus_r, lam=self.lam)                  # (B, d)

        # 최종: u = (1/γ) · ū
        return inv_gamma * u_bar
