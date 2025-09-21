# 파일: tests/mt_nd_max_c/user_pgdpo_with_projection.py
# 모델: 소비 상한 모델 (PMP가 투자와 소비를 각각 최적화) - 벡터화 & 최종 수정 버전

from __future__ import annotations
import os
import torch

# user_pgdpo_base.py에서 필요한 심볼 가져오기
try:
    from user_pgdpo_base import (
        alpha as ALPHA_CONST,       # (d,)
        Sigma_inv as SIGMAi_CONST,  # (d,d)
        gamma as GAMMA_CONST,       # CRRA 계수 (스칼라)
        C_abs_cap as C_MAX_CONST    # 소비 상한 (스칼라)
    )
except Exception as e:
    raise RuntimeError("[with_projection] user_pgdpo_base에서 필요한 심볼(alpha, Sigma_inv, gamma, C_abs_cap)을 찾지 못했습니다.") from e

# 코어가 필요로 하는 코스테이트 키
PP_NEEDS = ("JX", "JXX")

# 옵션 (소비는 기본적으로 닫힌형+클램프 사용; 필요시 배리어-뉴턴으로 전환 가능)
PP_OPTS = {
    "C_min": 1e-5,      # 소비 하한
    "eps_bar": 1e-7,    # 배리어 세기 (배리어 모드에서만 사용)
    "max_newton": 15,   # 뉴턴 최대 반복 (배리어 모드)
    "tol_grad": 1e-8,   # 수렴 허용오차 (배리어 모드)
    "tau": 0.95,        # fraction-to-the-boundary (배리어 모드)
    "use_closed_form_C": True,  # True면 닫힌형+클램프, False면 배리어-뉴턴
}

@torch.no_grad()
def _solve_C_barrier_batched(
    JX: torch.Tensor, *,
    Cmin: float, Cmax: float,
    eps_bar: float, gamma: float,
    max_newton: int = 15,
    tau: float = 0.95,
    tol: float = 1e-8
) -> torch.Tensor:
    """
    소비 1차 방정식을 배치 뉴턴으로 푸는 버전(로그-배리어).
    JX: (B,1)
    반환: C (B,1)
    """
    dev, dt = JX.device, JX.dtype
    jx = JX.clamp_min(1e-12)

    # 내부해 초기값 후 박스 클램프
    c = jx.pow(-1.0 / gamma).clamp(min=Cmin, max=Cmax)

    for _ in range(int(max_newton)):
        c_minus_min   = (c - Cmin).clamp_min(1e-12)
        c_max_minus_c = (Cmax - c).clamp_min(1e-12)

        # f(c) = c^{-γ} - JX + ε/(c-Cmin) - ε/(Cmax-c)
        f  = c.pow(-gamma) - jx + eps_bar / c_minus_min - eps_bar / c_max_minus_c
        # f'(c) = -γ c^{-γ-1} - ε/(c-Cmin)^2 - ε/(Cmax-c)^2
        df = -(gamma) * c.pow(-gamma - 1.0) - eps_bar / (c_minus_min**2) - eps_bar / (c_max_minus_c**2)
        df = torch.minimum(df, torch.full_like(df, -1e-20))  # 안전 클램프(음수 유지)

        delta = -f / df

        # fraction-to-the-boundary (배치)
        alpha = torch.ones_like(delta)
        neg = delta < 0
        pos = delta > 0
        alpha = torch.minimum(alpha, torch.where(neg, tau * (c - Cmin) / (-delta + 1e-24), alpha))
        alpha = torch.minimum(alpha, torch.where(pos, tau * (Cmax - c) / ( delta + 1e-24), alpha))

        c_next = c + alpha * delta

        done = f.abs() < tol
        if done.all():
            c = c_next
            break

        c = torch.where(done, c, c_next)

    return c.clamp(min=Cmin, max=Cmax)


@torch.no_grad()
def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    PMP에 따라 최적 포트폴리오(u)와 최적 소비(C)를 배치로 계산.
    반환 텐서 shape: (B, d+1)  [u_1,...,u_d, C]
    """
    # --- 상태, 코스테이트 정리 ---
    X   = states["X"].view(-1, 1)        # (B,1)
    JX  = costates["JX"].view(-1, 1)     # (B,1)
    JXX = costates["JXX"].view(-1, 1)    # (B,1)

    dev, dt = X.device, X.dtype
    B = X.shape[0]

    # --- 1) 포트폴리오 u* (무제약 머튼형) : u* = s * (Σ^{-1} α),  s = -JX / (X JXX)
    #     수치 안전형: negJXX = (-JXX)^+  →  s = JX / (X * negJXX)
    alpha     = ALPHA_CONST.to(dev, dt).view(1, -1)      # (1,d)
    Sigma_inv = SIGMAi_CONST.to(dev, dt)                 # (d,d)
    dir_vec   = (Sigma_inv @ alpha.T).T                  # (1,d)

    negJXX = (-JXX).clamp_min(1e-12)                     # (B,1) > 0
    s      = (JX / (X * negJXX)).view(B, 1)              # (B,1)
    u_opt  = s * dir_vec                                 # (B,d)  (브로드캐스트)

    # --- 2) 소비 C*
    C_min = float(PP_OPTS["C_min"])
    C_max = float(C_MAX_CONST)
    gamma = float(GAMMA_CONST)

    # 환경변수로도 토글 가능: PGDPO_USE_CLOSED_FORM_C=0 → 배리어-뉴턴
    use_closed_form_env = os.getenv("PGDPO_USE_CLOSED_FORM_C", "")
    use_closed_form = PP_OPTS["use_closed_form_C"] if use_closed_form_env == "" else (use_closed_form_env != "0")

    if use_closed_form:
        # 닫힌형 + 박스 클램프: C = (JX)^(-1/γ) ∈ [Cmin, Cmax]
        jx = JX.clamp_min(1e-12)                           # (B,1)
        C_opt = jx.pow(-1.0 / gamma).clamp(min=C_min, max=C_max)
    else:
        # 로그-배리어 배치 뉴턴
        C_opt = _solve_C_barrier_batched(
            JX, Cmin=C_min, Cmax=C_max,
            eps_bar=float(PP_OPTS["eps_bar"]),
            gamma=gamma,
            max_newton=int(PP_OPTS["max_newton"]),
            tau=float(PP_OPTS["tau"]),
            tol=float(PP_OPTS["tol_grad"])
        )

    # --- 3) 결합 반환 ---
    return torch.cat([u_opt, C_opt], dim=1)