# tests/mt_nd_short/user_pgdpo_with_projection.py
import torch
from user_pgdpo_base import alpha as ALPHA_CONST, Sigma as SIGMA_CONST, gamma as GAMMA_CONST

PP_NEEDS = ("JX", "JXX")  # 코어 추정 키

PP_OPTS  = {
    "L_cap": 1.0,        # 무차입: sum(u) <= 1
    "eps_bar": 1e-3,     # 장벽 세기 (상황 따라 1e-4 ~ 1e-2)
    "ridge": 1e-10,      # 뉴턴 선형계 안정화 기본값 (필요시 아래에서 자동 증강)
    "tau": 0.95,         # fraction-to-the-boundary
    "armijo": 1e-4,      # Armijo (maximize)
    "backtrack": 0.5,    # 역추적 배율
    "max_newton": 30,    # 뉴턴 반복
    "tol_grad": 1e-8,    # ∥grad∥_∞ 허용오차
}

@torch.no_grad()
def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    X   = states["X"].view(-1, 1)         # (B,1)
    JX  = costates["JX"].view(-1, 1)      # (B,1)
    JXX = costates["JXX"].view(-1, 1)     # (B,1)
    device = X.device
    B = X.shape[0]

    # alpha / Sigma 배치화
    a_in = states.get("alpha", None)
    if a_in is None:
        alpha = ALPHA_CONST.to(device).view(1, -1).expand(B, -1)    # (B,d)
    else:
        alpha = (a_in if a_in.dim()==2 else a_in.view(1,-1).expand(B,-1)).to(device)
    d = alpha.shape[1]

    S_in = states.get("Sigma", None)
    if S_in is None:
        Sigma = SIGMA_CONST.to(device)                              # (d,d)
    else:
        Sigma = (S_in if S_in.dim()==2 else S_in[0]).to(device)     # 공통 Σ

    one = torch.ones(d, device=device)
    eps_bar = float(PP_OPTS["eps_bar"]); L_cap = float(PP_OPTS["L_cap"])
    tau = float(PP_OPTS["tau"]); armijo = float(PP_OPTS["armijo"])
    back = float(PP_OPTS["backtrack"]); ridge_base = float(PP_OPTS["ridge"])
    max_newton = int(PP_OPTS["max_newton"]); tol_grad = float(PP_OPTS["tol_grad"])

    # PMP 계수
    negJXX = (-JXX).clamp_min(1e-12)            # > 0
    a = (JX * X) * alpha                         # (B,d)
    scale = (X * X * negJXX).view(-1)            # (B,)

    # 무제약 초기값 (Q + ridge I) u = a
    u0 = torch.empty(B, d, device=device)
    I  = torch.eye(d, device=device)
    for b in range(B):
        Qb = (scale[b].item()) * Sigma
        Ab = a[b].unsqueeze(1)                   # (d,1)
        Qb_reg = Qb + ridge_base * I
        ub = torch.linalg.solve(Qb_reg, Ab).view(-1)
        u0[b] = ub

    # 유클리드 단순체(∑<=L) 투영: 견고 버전
    def proj_simplex_euclid(v, L=1.0):
        vpos = v.clamp_min(0.0)
        s = float(vpos.sum().item())
        if s <= L:
            return vpos
        u_sorted = torch.sort(vpos, descending=True).values
        cssv = torch.cumsum(u_sorted, dim=0) - L
        j = torch.arange(1, d+1, device=device, dtype=vpos.dtype)
        cond = u_sorted > (cssv / j)
        if cond.any():
            rho_idx = int(torch.nonzero(cond, as_tuple=False)[-1].item())
            theta = cssv[rho_idx] / float(rho_idx + 1)
        else:
            theta = cssv[-1] / float(d)  # fallback
        return (vpos - theta).clamp_min(0.0)

    u = torch.stack([proj_simplex_euclid(u0[b], L_cap) for b in range(B)], dim=0)

    # ---- 장벽 부호 수정: 최대화에서는 +eps*log(u) + eps*log(slack) ----
    def Hbar(uu, b):
        Qb = scale[b].item() * Sigma
        val = torch.dot(a[b], uu) - 0.5 * (uu @ Qb @ uu)
        slack = L_cap - uu.sum()
        bad = (uu <= 0).any() or (slack <= 0)
        if bad:
            return torch.tensor(-float("inf"), device=device, dtype=uu.dtype)
        val = val + eps_bar * torch.log(uu).sum()
        val = val + eps_bar * torch.log(slack)
        return val  # 0-dim Tensor

    def grad_Hbar(uu, b):
        Qb = scale[b].item() * Sigma
        slack = L_cap - uu.sum()
        g = a[b] - (Qb @ uu)
        g = g + eps_bar * (1.0 / uu)
        g = g - (eps_bar / slack) * one
        return g

    def hess_Hbar(uu, b, extra_ridge):
        Qb = scale[b].item() * Sigma
        slack = L_cap - uu.sum()
        # 음정(=concave) 강화: -Q  - eps*diag(1/u^2)  - eps/(slack^2) 11^T  + ridge I
        H = -Qb
        H = H - torch.diag(eps_bar / (uu * uu))
        H = H - (eps_bar / (slack * slack)) * (one.unsqueeze(1) @ one.unsqueeze(0))
        H = H + extra_ridge * torch.eye(d, device=device)
        return H

    # 뉴턴-백트래킹 (+자동 ridge 증강 + pinv fallback)
    for b in range(B):
        ub = u[b].clone()
        extra_ridge = ridge_base
        for _ in range(max_newton):
            val = Hbar(ub, b)
            if not torch.isfinite(val):
                ub = proj_simplex_euclid(u0[b], L_cap)
                val = Hbar(ub, b)
            g = grad_Hbar(ub, b)
            if g.abs().max().item() < tol_grad:
                break

            H = hess_Hbar(ub, b, extra_ridge)
            try:
                dlt = torch.linalg.solve(H, -g)
            except RuntimeError:
                # 수치 이슈: ridge 증강 → 그래도 안되면 pinv fallback
                extra_ridge = max(extra_ridge * 10.0, 1e-6)
                H = hess_Hbar(ub, b, extra_ridge)
                try:
                    dlt = torch.linalg.solve(H, -g)
                except RuntimeError:
                    dlt = torch.linalg.pinv(H) @ (-g)

            # fraction-to-the-boundary
            alpha = 1.0
            neg_idx = (dlt < 0)
            if neg_idx.any():
                alpha = min(alpha, float(torch.min(-tau * ub[neg_idx] / dlt[neg_idx]).item()))
            dsum = float(dlt.sum().item())
            slack = float((L_cap - ub.sum()).item())
            if dsum > 0.0:
                alpha = min(alpha, tau * slack / dsum)

            # Armijo (maximize)
            f0 = val; gTd = float(g.dot(dlt).item())
            for _ls in range(20):
                uc = ub + alpha * dlt
                if (uc > 0).all() and (uc.sum().item() < L_cap):
                    f1 = Hbar(uc, b)
                    if torch.isfinite(f1) and (f1 >= f0 + armijo * alpha * gTd):
                        ub = uc; val = f1
                        break
                alpha *= back
            else:
                break
        u[b] = ub

    return u
