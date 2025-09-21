# 파일: tests/mt_nd_max_c/user_pgdpo_with_projection.py
# 모델: 소비 상한 모델 (PMP가 투자와 소비를 각각 최적화) - 최종 수정 버전

from __future__ import annotations
import torch

# user_pgdpo_base.py에서 필요한 심볼 가져오기
try:
    from user_pgdpo_base import (
        alpha as ALPHA_CONST,
        Sigma_inv as SIGMAi_CONST,
        gamma as GAMMA_CONST,
        C_abs_cap as C_MAX_CONST # 고정 소비 상한을 가져옴
    )
except Exception as e:
    raise RuntimeError("[with_projection] user_pgdpo_base에서 필요한 심볼을 찾지 못했습니다.") from e

# 코어가 필요로 하는 코스테이트 키
PP_NEEDS = ("JX", "JXX")

# 최적화 하이퍼파라미터
PP_OPTS = {
    "C_min": 1e-5,       # 소비 하한
    "eps_bar": 1e-7,     # 장벽 세기
    "max_newton": 15,    # 뉴턴 최대 반복
    "tol_grad": 1e-8,    # 수렴 허용오차
    "tau": 0.95,         # ✨ 수정: 누락되었던 'tau' 파라미터 추가
}

@torch.no_grad()
def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    PMP에 따라 최적 포트폴리오(u)와 최적 소비(C)를 각각 계산합니다.
    """
    # --- 상태 및 파라미터 준비 ---
    X = states["X"]
    JX = costates["JX"]
    JXX = costates["JXX"]
    B = X.shape[0]
    device = X.device
    
    # --- 1. 최적 포트폴리오 u* 계산 (제약 없음) ---
    denominator = (X * JXX).clamp(max=-1e-12)
    s = (-JX / denominator)
    alpha = ALPHA_CONST.view(1, -1)
    Sigma_inv = SIGMAi_CONST
    portfolio_direction = (Sigma_inv @ alpha.T).T
    u_optimal = s * portfolio_direction
    
    # --- 2. 최적 소비 C* 계산 (제약 있음, 뉴턴법 사용) ---
    C_optimal = torch.empty(B, 1, device=device, dtype=X.dtype)
    gamma_val = float(GAMMA_CONST)
    eps_bar = float(PP_OPTS["eps_bar"])
    C_max = float(C_MAX_CONST)
    C_min = float(PP_OPTS["C_min"])
    tau = float(PP_OPTS["tau"]) # ✨ 수정: tau 값을 변수로 사용

    # 각 배치 샘플에 대해 독립적으로 1차원 뉴턴법 수행
    for b in range(B):
        jx_b = JX[b].item()
        c_b = (jx_b + 1e-9)**(-1.0 / gamma_val)
        c_b = min(max(c_b, C_min), C_max)
        if c_b <= C_min: c_b = C_min * 1.01
        if c_b >= C_max: c_b = C_max * 0.99

        for _ in range(PP_OPTS["max_newton"]):
            c_minus_min = c_b - C_min
            c_max_minus_c = C_max - c_b
            
            f_c = c_b**(-gamma_val) - jx_b + eps_bar / c_minus_min - eps_bar / c_max_minus_c
            
            if abs(f_c) < PP_OPTS["tol_grad"]:
                break
            
            df_c = -gamma_val * c_b**(-gamma_val - 1) - eps_bar / (c_minus_min**2) - eps_bar / (c_max_minus_c**2)
            
            step = -f_c / (df_c - 1e-12)
            
            # ✨ 수정: tau를 사용하여 라인 서치 수행
            alpha_step = 1.0
            if step < 0:
                alpha_step = min(alpha_step, tau * (-c_minus_min / step))
            if step > 0:
                alpha_step = min(alpha_step, tau * (c_max_minus_c / step))

            c_b = c_b + alpha_step * step
        
        C_optimal[b] = c_b

    # --- 3. u*와 C*를 결합하여 반환 ---
    return torch.cat([u_optimal, C_optimal], dim=1)