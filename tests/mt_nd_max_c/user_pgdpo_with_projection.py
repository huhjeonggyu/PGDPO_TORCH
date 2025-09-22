# 파일: tests/mt_nd_max_c/user_pgdpo_with_projection.py
# 모델: 소비 상한 모델 (상대 상한; u는 PMP 투영으로 학습, C는 배리어-뉴턴) - 벡터화 & 최종

from __future__ import annotations
import os
import torch

# user_pgdpo_base.py에서 필요한 심볼 가져오기
try:
    from user_pgdpo_base import (
        alpha as ALPHA_CONST,        # (d,)
        Sigma_inv as SIGMAi_CONST,   # (d,d)
        gamma as GAMMA_CONST,        # CRRA 계수 (스칼라)
        rho as RHO_CONST,            # 할인율
        T as T_CONST,                # 만기
        c_frac as CFRAC_CONST,       # 상대 소비 상한 비율: C_max(X)=c_frac*X
    )
except Exception as e:
    raise RuntimeError("[with_projection] user_pgdpo_base에서 필요한 심볼(alpha, Sigma_inv, gamma, rho, T, c_frac)을 찾지 못했습니다.") from e

# 코어가 필요로 하는 코스테이트 키
PP_NEEDS = ("JX", "JXX")

# 옵션
PP_OPTS = {
    # u 관련
    "force_const_merton": bool(int(os.getenv("PGDPO_FORCE_CONST_U", "0"))),  # 1이면 u=(1/γ)Σ^{-1}α
    "min_curv": 1e-12,    # -JXX 하한(수치 안정)

    # C(배리어-뉴턴) 관련
    "C_min": 1e-6,
    "eps_bar": 1e-5,
    "max_newton": 25,
    "tau": 0.95,          # fraction-to-the-boundary
    "tol_grad": 1e-8,

    # 반환 형태: 기본 (u,C) 동시 반환. 0으로 두면 u만 반환(ppgdpo_u_direct 호환)
    "return_both_default": bool(int(os.getenv("PGDPO_RETURN_BOTH", "1"))),
}

# ------------------------- 소비 배리어-뉴턴 -------------------------
@torch.no_grad()
def _solve_C_barrier_batched(
    JX: torch.Tensor, X: torch.Tensor, t: torch.Tensor,
    *, rho: float, gamma: float, c_frac: float,
    Cmin: float, eps_bar: float, max_newton: int, tau: float, tol: float
) -> torch.Tensor:
    """
    상대 상한 C ∈ (Cmin, Cmax),  Cmax = c_frac * X
    FOC: e^{-ρt} C^{-γ} - JX + ε/(C-Cmin) - ε/(Cmax-C) = 0
    """
    dev, dt = JX.device, JX.dtype
    disc = torch.exp(-rho * t)                          # (B,1)
    Cmax = (c_frac * X).clamp_min(Cmin + 1e-12)         # (B,1)
    jx   = JX.clamp_min(1e-12)

    # 무배리어 내부해 초기화: e^{-ρt} C^{-γ} = JX ⇒ C = (disc/JX)^{1/γ}
    c = (disc / jx).pow(1.0 / gamma)
    # ⚠️ clamp 혼합인자 금지: 둘 다 텐서로 맞추거나 minimum/maximum 사용
    c_min_t = torch.full_like(c, Cmin + 1e-8)
    c_max_t = (Cmax - 1e-8)
    c = torch.maximum(c, c_min_t)
    c = torch.minimum(c, c_max_t)

    for _ in range(int(max_newton)):
        c_minus_min   = torch.maximum(c - Cmin, torch.full_like(c, 1e-12))
        c_max_minus_c = torch.maximum(Cmax - c, torch.full_like(c, 1e-12))

        # f(c), f'(c)
        f  = disc * c.pow(-gamma) - jx + eps_bar / c_minus_min - eps_bar / c_max_minus_c
        df = disc * (-gamma) * c.pow(-gamma - 1.0) \
             - eps_bar / (c_minus_min**2) - eps_bar / (c_max_minus_c**2)
        df = torch.minimum(df, torch.full_like(df, -1e-20))  # 안정성(음수 유지)

        delta = -f / df  # 뉴턴 스텝

        # fraction-to-the-boundary
        step = torch.ones_like(delta)
        neg = delta < 0
        pos = delta > 0
        if neg.any():
            step = torch.minimum(step, torch.where(neg, tau * (c - Cmin) / (-delta + 1e-24), step))
        if pos.any():
            step = torch.minimum(step, torch.where(pos, tau * (Cmax - c) / ( delta + 1e-24), step))

        c_next = c + step * delta

        # 수렴 체크
        if f.abs().max() < tol:
            c = c_next
            break
        c = c_next

    # ⚠️ 최종 클램프도 두 텐서 조합으로
    c = torch.maximum(c, torch.full_like(c, Cmin))
    c = torch.minimum(c, Cmax)
    return c  # (B,1)

@torch.no_grad()
def project_pmp_C(costates: dict, states: dict) -> torch.Tensor:
    """
    소비 C를 배리어-뉴턴으로 계산해 반환. shape: (B,1)
    """
    X   = states["X"].view(-1, 1)
    tau = states.get("TmT", None)
    if tau is None:
        # t 정보가 없으면 보수적으로 t=0 처리
        t = torch.zeros_like(X)
    else:
        t = (T_CONST - tau).view(-1, 1)

    JX  = costates["JX"].view(-1, 1)

    C = _solve_C_barrier_batched(
        JX, X, t,
        rho=float(RHO_CONST), gamma=float(GAMMA_CONST), c_frac=float(CFRAC_CONST),
        Cmin=float(PP_OPTS["C_min"]), eps_bar=float(PP_OPTS["eps_bar"]),
        max_newton=int(PP_OPTS["max_newton"]), tau=float(PP_OPTS["tau"]), tol=float(PP_OPTS["tol_grad"])
    )
    return C  # (B,1)

# ------------------------- u (PMP 투영: 학습) -------------------------
@torch.no_grad()
def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    PMP에 따라 최적 포트폴리오 u (및 옵션에 따라 C)를 계산.
    기본은 (u,C) 동시 반환. PGDPO_RETURN_BOTH=0이면 u만 반환.
    """
    X   = states["X"].view(-1, 1)        # (B,1)
    JX  = costates["JX"].view(-1, 1)     # (B,1)
    JXX = costates["JXX"].view(-1, 1)    # (B,1)

    dev, dt = X.device, X.dtype
    B, d = X.shape[0], ALPHA_CONST.numel()

    alpha     = ALPHA_CONST.to(dev, dt).view(1, -1)   # (1,d)
    Sigma_inv = SIGMAi_CONST.to(dev, dt)              # (d,d)
    dir_vec   = (Sigma_inv @ alpha.T).T               # (1,d) = (Σ^{-1}α)^T

    # u: 코스테이트 기반 스케일 s = -JX/(X JXX)  (JXX<0 ⇒ s>0)
    if PP_OPTS["force_const_merton"]:
        s = torch.full((B, 1), 1.0 / float(GAMMA_CONST), device=dev, dtype=dt)
    else:
        negJXX = (-JXX).clamp_min(float(PP_OPTS["min_curv"]))  # (B,1) > 0
        s = (JX / (X * negJXX)).to(dt)                         # (B,1)

    u_opt = s * dir_vec                                        # (B,d), 브로드캐스트

    # 소비 C: 배리어-뉴턴 (상대 상한)
    env_return_both = os.getenv("PGDPO_RETURN_BOTH")
    return_both = PP_OPTS["return_both_default"] if env_return_both is None else (env_return_both != "0")
    if return_both:
        C_opt = project_pmp_C(costates, states)                # (B,1)
        return torch.cat([u_opt, C_opt], dim=1)                # (B, d+1)
    else:
        return u_opt                                           # (B, d)
