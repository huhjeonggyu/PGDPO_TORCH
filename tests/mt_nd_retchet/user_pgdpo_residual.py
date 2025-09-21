# 파일: tests/mt_nd_ratchet/user_pgdpo_residual.py
# 모델: 소비 랫칭 모델의 잔차 학습을 위한 기준 정책 (최종 완성본)

import torch
import torch.nn as nn
import torch.nn.functional as F

# user_pgdpo_base.py에서 필요한 파라미터를 가져옵니다.
try:
    from user_pgdpo_base import (
        alpha as ALPHA_CONST,
        Sigma_inv as SIGMAi_CONST,
        gamma as GAMMA_CONST,
        L_cap as L_CAP_CONST,
        d,
    )
except Exception as e:
    raise RuntimeError("[residual] user_pgdpo_base에서 필요한 상수를 찾지 못했습니다.") from e

# --- (핵심 수정) ResCap 변수 추가 ---
# 코어 프레임워크가 필요로 하는 잔차 크기 조절 변수
ResCap = 1.0

# 기준 소비를 위한 소비 성향 (상수)
C_FRAC_MYOPIC = 0.10

class MyopicPolicy(nn.Module):
    """
    잔차 학습의 베이스라인이 될 Myopic 정책.
    - 투자(u): Softmax 매핑으로 제약 조건을 만족하는 상수 머튼 포트폴리오.
    - 소비(C): 비례 소비 규칙을 따르되, 랫칭 제약을 만족.
    """
    def __init__(self):
        super().__init__()
        u_unc = (1.0 / float(GAMMA_CONST)) * (SIGMAi_CONST @ ALPHA_CONST)
        u_logits = u_unc.clamp(min=-1e8) 
        weights = F.softmax(u_logits, dim=0)
        u_star = L_CAP_CONST * weights
        
        self.register_buffer("u_star", u_star.view(-1))

    @torch.no_grad()
    def forward(self, **states_dict):
        wealth = states_dict['X'][:, 0:1]
        habit  = states_dict['X'][:, 1:2]
        B = wealth.size(0)

        u = self.u_star.unsqueeze(0).expand(B, -1)
        c_proportional = C_FRAC_MYOPIC * wealth
        C = torch.max(c_proportional, habit)
        C = torch.min(C, wealth)

        return torch.cat([u, C], dim=1)