# 파일: tests/mt_nd_ratchet/user_pgdpo_with_projection.py
# 모델: 소비 랫칭 모델의 PMP 프로젝션 (최종 버그 수정)

from __future__ import annotations
import torch

try:
    from user_pgdpo_base import (
        alpha as ALPHA_CONST,
        Sigma as SIGMA_CONST,
        Sigma_inv as SIGMAi_CONST,
        gamma as GAMMA_CONST,
        L_cap as L_CAP_CONST
    )
except Exception as e:
    raise RuntimeError("[with_projection] user_pgdpo_base에서 필요한 상수를 찾지 못했습니다.") from e

PP_NEEDS = ("JX", "JXX")

PP_OPTS = {
    "L_cap": float(L_CAP_CONST),
    "eps_bar": 1e-3, "ridge": 1e-12, "tau": 0.95, "armijo": 1e-4,
    "backtrack": 0.5, "max_newton": 30, "tol_grad": 1e-8, "interior_shrink": 1e-6,
}

@torch.no_grad()
def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    alpha = ALPHA_CONST.view(-1)
    Sigma = SIGMA_CONST
    Sigma_inv = SIGMAi_CONST
    gamma = float(GAMMA_CONST)
    d = alpha.numel()
    device = alpha.device

    wealth = states["X"][:, 0:1]
    
    # --- (핵심 수정) Co-state 텐서 인덱싱 방식 변경 ---
    # JX는 (B, 2), JXX는 (B, 2) 형태
    JX_w = costates["JX"][:, 0:1]
    
    # JXX가 2차원 텐서이므로, 2차원 인덱싱으로 첫 번째 열을 추출
    JXX_ww = costates["JXX"][:, 0:1]

    B = wealth.size(0)
    L_cap = float(PP_OPTS["L_cap"])
    eps_bar = float(PP_OPTS["eps_bar"])
    ridge0 = float(PP_OPTS["ridge"])
    tau = float(PP_OPTS["tau"])
    c1 = float(PP_OPTS["armijo"])
    back = float(PP_OPTS["backtrack"])
    max_newton = int(PP_OPTS["max_newton"])
    tol_g = float(PP_OPTS["tol_grad"])
    interior_eps = float(PP_OPTS["interior_shrink"])
    ones = torch.ones(d, device=device, dtype=alpha.dtype)

    denom = (wealth * JXX_ww).clamp_min(1e-12)
    s_unclamped = (-JX_w / denom).view(-1)
    s = s_unclamped.clamp(1e-4, 1e4)

    u_unc = (1.0 / gamma) * (Sigma_inv @ alpha)
    u0 = u_unc.clamp_min(0.0)
    if u0.sum().item() > L_cap:
        u0 = u0 * L_cap / u0.sum()
    u0 = u0.clamp_min(interior_eps)
    if u0.sum().item() >= L_cap:
        u0 = (L_cap - interior_eps) * u0 / (u0.sum() + 1e-12)
    
    u = torch.empty(B, d, device=device, dtype=alpha.dtype)

    def Hbar(u_vec, sb):
        slack = L_cap - u_vec.sum()
        if slack <= 0 or (u_vec <= 0).any(): return torch.tensor(float("-inf"), device=device)
        return sb * (alpha @ u_vec - 0.5 * gamma * (u_vec @ (Sigma @ u_vec))) + \
               eps_bar * (torch.log(u_vec).sum() + torch.log(slack))

    def grad_Hbar(u_vec, sb):
        slack = L_cap - u_vec.sum()
        return sb * (alpha - gamma * (Sigma @ u_vec)) + eps_bar * (1.0 / u_vec) - (eps_bar / slack) * ones

    def hess_Hbar(u_vec, sb):
        slack = L_cap - u_vec.sum()
        return (-sb * gamma) * Sigma - eps_bar * torch.diag(1.0 / (u_vec ** 2)) - \
               (eps_bar / (slack ** 2)) * (ones.view(-1,1) @ ones.view(1,-1))

    for b in range(B):
        sb = float(s[b].item())
        ub = u0.clone()
        val = Hbar(ub, sb)
        
        for it in range(max_newton):
            g = grad_Hbar(ub, sb)
            if torch.isfinite(g).all() and float(g.abs().max().item()) < tol_g: break
            
            H = hess_Hbar(ub, sb)
            A = -H + ridge0 * torch.eye(d, device=device, dtype=H.dtype)
            try:
                dlt = torch.linalg.solve(A, g)
            except RuntimeError:
                dlt = torch.linalg.pinv(A) @ g

            a_max = 1.0
            if (dlt < 0).any(): a_max = min(a_max, float((-tau * ub[dlt < 0] / dlt[dlt < 0]).min().item()))
            if -dlt.sum() > 0: a_max = min(a_max, float(tau * (L_cap - ub.sum()) / (-dlt.sum())))
            
            a, f0, gTd = a_max, float(val.item()), float(g.dot(dlt).item())
            accepted = False
            for _ls in range(20):
                uc = ub + a * dlt
                if (uc > 0).all() and (uc.sum().item() < L_cap):
                    f1t = Hbar(uc, sb)
                    if torch.isfinite(f1t) and (f1t.item() >= f0 + c1 * a * gTd):
                        ub, val, accepted = uc, f1t, True
                        break
                a *= back
            if not accepted: break
        u[b] = ub

    return u