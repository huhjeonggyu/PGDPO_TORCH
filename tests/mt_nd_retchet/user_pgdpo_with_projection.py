# -*- coding: utf-8 -*-
# 파일: tests/mt_nd_retchet/user_pgdpo_with_projection.py
# 목적: PMP 프로젝션( u & C 동시 ) — consumption cap 구현과 동일한 장벽-뉴턴 스타일
# 제약:
#   u_i > 0,  slack = L_cap - sum(u) > 0
#   C - Y > 0              (래칫팅 하한)
#   (선택) C_max - C > 0   (상한 쓰려면 cap_mult*X에 대해 장벽 추가)
#
# 반환: (B, d+1)  = [u_proj (B,d), C_proj (B,1)]

from __future__ import annotations
import os
import torch

# ---- base 모듈에서 상수 import (상대/절대 폴백) ----
try:
    from tests.mt_nd_retchet.user_pgdpo_base import (  # type: ignore
        alpha as ALPHA_CONST,
        Sigma as SIGMA_CONST,
        Sigma_inv as SIGMAi_CONST,
        gamma as GAMMA_CONST,
        L_cap as L_CAP_CONST,
        rho as RHO_CONST,
        T   as T_CONST,
    )
except Exception:
    from user_pgdpo_base import (  # type: ignore
        alpha as ALPHA_CONST,
        Sigma as SIGMA_CONST,
        Sigma_inv as SIGMAi_CONST,
        gamma as GAMMA_CONST,
        L_cap as L_CAP_CONST,
        rho as RHO_CONST,
        T   as T_CONST,
    )

PP_NEEDS = ("JX", "JXX")  # 진단용

# 장벽/뉴턴 하이퍼 (consumption cap 시나리오와 동일한 스타일)
PP_OPTS = {
    "L_cap": float(L_CAP_CONST),
    "eps_bar": 1e-3,          # 장벽 세기
    "ridge":   1e-10,         # 헤시안 리지
    "tau":     0.95,          # fraction-to-the-boundary
    "armijo":  1e-4,          # Armijo 상수
    "backtrack": 0.5,         # 라인서치 축소비
    "max_newton": 30,         # 뉴턴 최대 반복
    "tol_grad":  1e-8,        # ∥g∥∞ 정지
    "interior_eps": 1e-8,     # 내부성 보정
}

# (선택) 소비 상한 — cap_mult * X. 사용 안 하려면 환경변수 PGDPO_C_SOFTCAP=None
CAP_MULT_STR = os.getenv("PGDPO_C_SOFTCAP", "None")
CAP_MULT = None if CAP_MULT_STR == "None" else float(CAP_MULT_STR)


def _first_col_1d(T: torch.Tensor) -> torch.Tensor:
    """wealth 방향 성분을 (B,1)로 안전 추출. 허용 shape: (B,), (B,1), (B,k), (B,k,k)."""
    if T is None:
        raise RuntimeError("Required costate tensor is None")
    if T.dim() == 1:
        return T.view(-1, 1)
    if T.dim() == 2:
        return T[:, 0:1]
    if T.dim() == 3:
        return T[:, 0:1, 0:1].view(-1, 1)
    raise RuntimeError(f"Unexpected costate tensor shape: {tuple(T.shape)}")


@torch.no_grad()
def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    입력:
      costates: {"JX": ..., "JXX": ...} (shape 가변)
      states  : {"X": (B,2)=[wealth,habit], "TmT": (B,1) [옵션]}
    반환:
      U_pp    : (B, d+1) = [u_proj (B,d), C_proj (B,1)]
    """
    alpha = ALPHA_CONST.view(-1)      # (d,)
    Sigma = SIGMA_CONST
    Sigma_inv = SIGMAi_CONST
    gamma = float(GAMMA_CONST)
    d = int(alpha.numel())
    device = alpha.device
    dtype = alpha.dtype

    X_block = states["X"]                 # (B,2): [wealth, habit]
    wealth = X_block[:, 0:1]              # (B,1)
    habit  = X_block[:, 1:2]              # (B,1)
    tau    = states.get("TmT", None)      # (B,1) or None
    if tau is None:
        tau = torch.zeros_like(wealth)    # 없으면 t=0 가정

    # costates: λ=∂J/∂X,  ∂^2J/∂X^2
    lam    = _first_col_1d(costates.get("JX",  None)).clamp_min(1e-12)   # (B,1)
    JXX_ww = _first_col_1d(costates.get("JXX", None)).clamp_min(1e-12)   # (B,1)

    # s ≈ -(∂J/∂X)/(X ∂^2J/∂X^2)
    s_vec = (-lam / (wealth * JXX_ww)).view(-1).clamp(1e-4, 1e4)         # (B,)

    # 장벽/뉴턴 준비
    ones_d = torch.ones(d, device=device, dtype=dtype)
    L_cap  = PP_OPTS["L_cap"]
    eps_bar= PP_OPTS["eps_bar"]
    ridge0 = PP_OPTS["ridge"]
    tau_fb = PP_OPTS["tau"]
    c1     = PP_OPTS["armijo"]
    back   = PP_OPTS["backtrack"]
    max_n  = PP_OPTS["max_newton"]
    tol_g  = PP_OPTS["tol_grad"]
    eps_in = PP_OPTS["interior_eps"]

    # 초기점 (consumption cap 코드와 동일한 규칙)
    # u0: unconstrained Merton on simplex 내부
    u_unc = (1.0 / gamma) * (Sigma_inv @ alpha)           # (d,)
    u0 = u_unc.clamp_min(eps_in)
    s0 = float(u0.sum().item())
    if s0 >= L_cap:
        u0 = (L_cap - eps_in) * u0 / (s0 + 1e-12)

    # C0: 래칫 하한에서 시작 (너무 커서 경계 밖이면 한 스텝 안으로)
    C0 = (habit + 1e-6).clone()

    # 배치 결과
    B = wealth.size(0)
    U_pp = torch.empty(B, d + 1, device=device, dtype=dtype)

    def Hbar(u_vec: torch.Tensor, c_val: torch.Tensor, s: float, lam_b: float, t_abs: float,
             y_val: float, x_val: float) -> torch.Tensor:
        """
        장벽 포함 스테이지 목적:
        s(α^T u - 1/2 γ u^TΣu) + e^{-ρt}U(c) - λ c + ε [∑log u + log slack + log(c - y) + (opt)log(Cmax-c)]
        """
        slack = L_cap - u_vec.sum()
        if slack <= 0 or (u_vec <= 0).any():  # 내부성
            return torch.tensor(float("-inf"), device=device, dtype=dtype)

        val = s * (alpha @ u_vec - 0.5 * gamma * (u_vec @ (Sigma @ u_vec)))
        # 효용
        if abs(GAMMA_CONST - 1.0) > 1e-12:
            Uc = (c_val.clamp_min(1e-12) ** (1.0 - GAMMA_CONST) - 1.0) / (1.0 - GAMMA_CONST)
        else:
            Uc = torch.log(c_val.clamp_min(1e-12))
        val = val + torch.exp(torch.tensor(-RHO_CONST * t_abs, device=device, dtype=dtype)) * Uc - lam_b * c_val
        # 장벽
        val = val + eps_bar * (torch.log(u_vec).sum() + torch.log(slack))
        val = val + eps_bar * torch.log(c_val - y_val + 1e-12)  # 래칫 하한
        if CAP_MULT is not None:
            cmax = CAP_MULT * x_val
            val = val + eps_bar * torch.log(cmax - c_val + 1e-12)

        return val

    def grad_blocks(u_vec: torch.Tensor, c_val: torch.Tensor, s: float, lam_b: float, t_abs: float,
                    y_val: float, x_val: float):
        """ ∇u Hbar, ∂Hbar/∂c 및 헤시안 블록 (교차 0) """
        # ∂/∂u
        slack = L_cap - u_vec.sum()
        g_u = s * (alpha - gamma * (Sigma @ u_vec)) + eps_bar * (1.0 / u_vec) - (eps_bar / slack) * ones_d
        H_uu = (-s * gamma) * Sigma - eps_bar * torch.diag(1.0 / (u_vec ** 2)) \
               - (eps_bar / (slack ** 2)) * (ones_d.view(-1,1) @ ones_d.view(1,-1))
        # ∂/∂c
        if abs(GAMMA_CONST - 1.0) > 1e-12:
            U1 = (c_val.clamp_min(1e-12) ** (-GAMMA_CONST))
            U2 = -GAMMA_CONST * (c_val.clamp_min(1e-12) ** (-GAMMA_CONST - 1.0))
        else:
            U1 = 1.0 / c_val.clamp_min(1e-12)
            U2 = -1.0 / (c_val.clamp_min(1e-12) ** 2)

        disc = torch.exp(torch.tensor(-RHO_CONST * t_abs, device=c_val.device, dtype=c_val.dtype))
        g_c = disc * U1 - lam_b + eps_bar / (c_val - y_val + 1e-12)
        H_cc = disc * U2 - eps_bar / ((c_val - y_val + 1e-12) ** 2)
        if CAP_MULT is not None:
            cmax = CAP_MULT * x_val
            g_c = g_c - eps_bar / (cmax - c_val + 1e-12)
            H_cc = H_cc - eps_bar / ((cmax - c_val + 1e-12) ** 2)

        return g_u, H_uu, g_c, H_cc

    def frac_to_boundary(u_vec: torch.Tensor, du: torch.Tensor, c_val: torch.Tensor, dc: torch.Tensor,
                         y_val: float, x_val: float, tau_frac: float) -> float:
        """ fraction-to-the-boundary: 모든 장벽 제약을 유지하는 최대 step """
        a_max = 1.0
        # u_i > 0
        neg = (du < 0)
        if neg.any():
            a_max = min(a_max, float((-tau_frac * u_vec[neg] / du[neg]).min().item()))
        # slack = L_cap - sum(u) > 0
        sum_dir = float(du.sum().item())
        slack = float((L_cap - float(u_vec.sum().item())))
        if sum_dir > 0:
            a_max = min(a_max, float(tau_frac * slack / sum_dir))
        # c - y > 0
        if float(dc.item()) < 0:
            a_max = min(a_max, float(tau_frac * (float(c_val.item()) - y_val) / (-float(dc.item()))))
        # (선택) cmax - c > 0
        if CAP_MULT is not None:
            cmax = CAP_MULT * x_val
            if float(dc.item()) > 0:
                a_max = min(a_max, float(tau_frac * (cmax - float(c_val.item())) / float(dc.item())))
        return max(1e-12, a_max)

    B = wealth.size(0)
    uC_out = torch.empty(B, d + 1, device=device, dtype=dtype)

    # per-sample Newton (consumption cap 코드와 동일한 패턴)
    for b in range(B):
        x_b   = float(wealth[b, 0].item())
        y_b   = float(habit [b, 0].item())
        t_abs = float(T_CONST - float(tau[b, 0].item()))
        lam_b = float(lam  [b, 0].item())
        s_b   = float(s_vec[b].item())

        u_b = (ALPHA_CONST * 0.0).to(device=device, dtype=dtype)
        # 초기 u: u0 복제
        u_b.copy_((1.0 / gamma) * (SIGMAi_CONST @ ALPHA_CONST))
        u_b.clamp_(min=eps_in)
        su = float(u_b.sum().item())
        if su >= L_cap:
            u_b.mul_((L_cap - eps_in) / (su + 1e-12))

        c_b = max(y_b + 1e-6, 1e-6)  # 초기 C는 랫칫 하한 살짝 위

        # 초기 목적
        f0 = Hbar(u_b, torch.tensor(c_b, device=device, dtype=dtype), s_b, lam_b, t_abs, y_b, x_b)

        for _ in range(max_n):
            g_u, H_uu, g_c, H_cc = grad_blocks(u_b, torch.tensor(c_b, device=device, dtype=dtype),
                                               s_b, lam_b, t_abs, y_b, x_b)
            # 정지 판정
            if float(torch.max(torch.abs(g_u)).item()) < tol_g and abs(float(g_c.item())) < tol_g:
                break

            # 뉴턴 방향 (블록 대각선 — 교차=0)
            try:
                du = torch.linalg.solve(-H_uu + ridge0 * torch.eye(d, device=device, dtype=dtype), g_u)
            except RuntimeError:
                du = torch.linalg.pinv(-H_uu + ridge0 * torch.eye(d, device=device, dtype=dtype)) @ g_u
            dc = float(g_c.item() / max(-float(H_cc.item()) + ridge0, 1e-12))

            # fraction-to-the-boundary
            a_max = frac_to_boundary(u_b, du, torch.tensor(c_b, device=device, dtype=dtype),
                                     torch.tensor(dc, device=device, dtype=dtype), y_b, x_b, tau_fb)

            # Armijo 라인서치
            a = a_max; accepted = False
            gTd = float(g_u.dot(du).item() + g_c.item() * dc)
            for _ in range(25):
                u_try = u_b + a * du
                c_try = c_b + a * dc
                if (u_try > 0).all() and (float(u_try.sum().item()) < L_cap) and (c_try > y_b):
                    if CAP_MULT is None or (c_try < CAP_MULT * x_b):
                        f1 = Hbar(u_try, torch.tensor(c_try, device=device, dtype=dtype),
                                  s_b, lam_b, t_abs, y_b, x_b)
                        if torch.isfinite(f1) and float(f1.item()) >= float(f0.item()) + PP_OPTS["armijo"] * a * gTd:
                            u_b, c_b, f0, accepted = u_try, c_try, f1, True
                            break
                a *= back
            if not accepted:
                break

        # 최종 보정(내부 유지)
        u_b.clamp_(min=eps_in)
        su = float(u_b.sum().item())
        if su >= L_cap:
            u_b.mul_((L_cap - eps_in) / (su + 1e-12))
        c_b = max(c_b, y_b + 1e-9)
        if CAP_MULT is not None:
            c_b = min(c_b, CAP_MULT * x_b - 1e-9)

        uC_out[b, :d]   = u_b
        uC_out[b, d:d+1] = torch.tensor([c_b], device=device, dtype=dtype)

    return uC_out
