# tests/mt_nd_short/user_pgdpo_residual.py
# ------------------------------------------------------------
# Residual 모드용 "고정 마이오픽 정책":
#   u_myopic = (1/gamma) * Sigma_inv @ alpha  (머튼 공식)
#   -> 음수는 0으로 클램핑 (공매도 금지)
# 코어는 이 정책을 고정하고 ResidualPolicy(잔차 Δu)만 학습합니다.
# ------------------------------------------------------------

import torch
import torch.nn as nn

from user_pgdpo_base import (
    alpha as ALPHA_CONST,     # (d,)
    Sigma_inv as SIGMAi_CONST,  # (d,d)
    gamma as GAMMA_CONST,     # float or 0-dim tensor
)

# (선택) 잔차 크기 제한을 코어가 읽을 수 있도록 노출 (없어도 동작)
#   - 코어가 존재하면 사용하고, 없으면 기본값을 쓸 수 있게 되어 있음.
ResCap = 1.0  # 필요 없으면 조정/삭제 가능


class MyopicPolicy(nn.Module):
    """
    (상수) 마이오픽 정책 네트: 배치 입력에 대해 동일한 벡터 u를 반환.
    u* = (1/gamma) * Sigma_inv @ alpha,  이후 u* = clamp(u*, min=0).
    """
    def __init__(self):
        super().__init__()
        # 상수 파라미터로부터 미리 u* 계산
        # (장치/dtype을 ALPHA_CONST 기준으로 맞춤)
        device = ALPHA_CONST.device
        dtype  = ALPHA_CONST.dtype

        gamma_val = float(GAMMA_CONST) if torch.is_tensor(GAMMA_CONST) else float(GAMMA_CONST)
        u_star = (1.0 / gamma_val) * (SIGMAi_CONST.to(device=device, dtype=dtype) @ ALPHA_CONST.to(device=device, dtype=dtype))
        u_star = torch.clamp(u_star, min=0.0)  # 공매도 금지

        # 학습/그래프와 무관한 상수이므로 buffer에 보관
        self.register_buffer("u_star", u_star.view(-1))  # (d,)

    def forward(self, **states_dict):
        """
        states_dict: {"X": (B,1) 또는 (B,), "TmT": (B,1) ...}
        반환: (B, d) — 배치 크기에 맞춰 u_star를 복제한 텐서
        """
        # 배치 크기는 X로 파악 (없으면 u_star만 1행으로 반환)
        X = states_dict.get("X", None)
        if X is None:
            return self.u_star.unsqueeze(0)  # (1, d)

        B = X.shape[0]
        # (선택) forward에서 상한을 걸고 싶다면 아래 줄에서 max 인자만 쓰세요.
        # 예: torch.clamp(self.u_star, min=0.0, max=u_cap)
        return self.u_star.unsqueeze(0).expand(B, -1)
