# 파일: tests/mt_nd_short/user_pgdpo_with_projection.py
# 모델: 공매도 금지 + 레버리지 캡(∑ u_i <= L_cap) - 벡터화 & 최종 수정 버전

from __future__ import annotations
import torch
from user_pgdpo_base import alpha as ALPHA_CONST, Sigma as SIGMA_CONST, gamma as GAMMA_CONST

# 코어가 요구하는 코스테이트 키
PP_NEEDS = ("JX", "JXX")

# 하이퍼파라미터
PP_OPTS = {
    "L_cap": 1.0,       # 무차입: sum(u) <= L_cap
    "eps_bar": 1e-6,    # 장벽 세기(상황 따라 1e-4 ~ 1e-2 조정)
    "ridge": 1e-10,     # 뉴턴 선형계 안정화
    "tau": 0.95,        # fraction-to-the-boundary
    "armijo": 1e-4,     # Armijo (maximize)
    "backtrack": 0.5,   # 역추적 배율
    "max_newton": 30,   # 뉴턴 반복
    "tol_grad": 1e-8,   # ∥grad∥_∞ 허용오차
    "max_ls": 20,       # 라인서치 최대 반복
}

@torch.no_grad()
def _proj_simplex_batch(v: torch.Tensor, L: float) -> torch.Tensor:
    """
    배치 유클리드 단순체 투영: {u >= 0, sum(u) <= L}.
    v: (B,d)  ->  반환: (B,d)
    """
    B, d = v.shape
    vpos = v.clamp_min(0.0)
    s = vpos.sum(dim=1, keepdim=True)                              # (B,1)
    ok = s <= L
    if ok.all():
        return vpos
    # 표준 l1-ball 투영의 배치 구현
    u_sorted, _ = torch.sort(vpos, dim=1, descending=True)         # (B,d)
    cssv = torch.cumsum(u_sorted, dim=1) - L
    j = torch.arange(1, d + 1, device=v.device, dtype=v.dtype).view(1, -1)
    cond = u_sorted > (cssv / j)
    rho = cond.sum(dim=1, keepdim=True) - 1                        # (B,1)
    theta = cssv.gather(1, rho) / (rho + 1).to(v.dtype)            # (B,1)
    w = (vpos - theta).clamp_min(0.0)
    return torch.where(ok, vpos, w)

@torch.no_grad()
def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    로그-배리어를 사용한 PMP 사영(포트폴리오만; u >= 0, 1^T u <= L_cap).
    반환 shape: (B, d)
    """
    # ----- 입력 정리 -----
    X   = states["X"].view(-1, 1)          # (B,1)
    JX  = costates["JX"].view(-1, 1)       # (B,1)
    JXX = costates["JXX"].view(-1, 1)      # (B,1)
    dev, dt = X.device, X.dtype
    B = X.shape[0]

    # alpha 배치화: states에 개별 alpha가 있으면 사용
    a_in = states.get("alpha", None)
    if a_in is None:
        alpha = ALPHA_CONST.to(dev, dt).view(1, -1).expand(B, -1)         # (B,d)
    else:
        a2 = a_in if a_in.dim() == 2 else a_in.view(1, -1).expand(B, -1)
        alpha = a2.to(dev, dt)                                             # (B,d)
    d = alpha.shape[1]

    # Sigma 배치화: (d,d) 또는 (B,d,d) 지원
    S_in = states.get("Sigma", None)
    if S_in is None:
        Sigma = SIGMA_CONST.to(dev, dt)
        Sigma_b = Sigma.expand(B, -1, -1)                                  # (B,d,d)
    else:
        if S_in.dim() == 2:
            Sigma_b = S_in.to(dev, dt).expand(B, -1, -1)
        elif S_in.dim() == 3:
            Sigma_b = S_in.to(dev, dt)
        else:
            Sigma_b = S_in[0].to(dev, dt).expand(B, -1, -1)

    # ----- 파라미터 -----
    L_cap   = float(PP_OPTS["L_cap"])
    eps_bar = float(PP_OPTS["eps_bar"])
    ridge   = float(PP_OPTS["ridge"])
    tau     = float(PP_OPTS["tau"])
    armijo  = float(PP_OPTS["armijo"])
    back    = float(PP_OPTS["backtrack"])
    max_newton = int(PP_OPTS["max_newton"])
    tol_grad   = float(PP_OPTS["tol_grad"])
    max_ls     = int(PP_OPTS.get("max_ls", 20))

    # ----- PMP 계수 -----
    negJXX = (-JXX).clamp_min(1e-12)                            # (B,1)  > 0
    a_vec  = (JX * X) * alpha                                   # (B,d)
    scale  = (X * X * negJXX).view(B, 1, 1)                     # (B,1,1)
    Q_b    = scale * Sigma_b                                    # (B,d,d)

    # ----- 초기값 (무제약 정상식) + 단순체 투영 -----
    I_b = torch.eye(d, device=dev, dtype=dt).expand(B, d, d)    # (B,d,d)
    Qreg = Q_b + ridge * I_b
    # 배치 선형계: (Qreg) u0 = a
    try:
        u0 = torch.linalg.solve(Qreg, a_vec.unsqueeze(-1)).squeeze(-1)     # (B,d)
    except RuntimeError:
        u0 = (torch.linalg.pinv(Qreg) @ a_vec.unsqueeze(-1)).squeeze(-1)
    u = _proj_simplex_batch(u0, L_cap)                                     # (B,d)

    # 너무 경계에 붙으면 배리어 로그가 터지므로 살짝 안쪽으로 당겨줌
    s = u.sum(dim=1, keepdim=True)                                         # (B,1)
    on_edge = s >= (L_cap - 1e-10)
    if on_edge.any():
        scale_in = (L_cap * (1.0 - 1e-8)) / s.clamp_min(1e-12)
        u = torch.where(on_edge, u * scale_in, u)

    # ----- 뉴턴 반복 (배치) -----
    one = torch.ones(B, d, device=dev, dtype=dt)                           # (B,d)
    Jmat = torch.ones(B, d, d, device=dev, dtype=dt)                       # (B,d,d)

    for _ in range(max_newton):
        slack = (L_cap - u.sum(dim=1, keepdim=True)).clamp_min(1e-12)      # (B,1)

        # grad H_bar(u) = a - Q u + eps * (1/u) - eps/slack * 1
        Qu = torch.bmm(Q_b, u.unsqueeze(-1)).squeeze(-1)                   # (B,d)
        g = a_vec - Qu
        g = g + eps_bar * (1.0 / u.clamp_min(1e-12))
        g = g - (eps_bar / slack) * one                                    # (B,d)

        # 수렴 검사: 배치 무한노름
        if g.abs().amax(dim=1).max().item() < tol_grad:
            break

        # Hess H_bar(u) = -Q - eps*diag(1/u^2) - eps/(slack^2) * 11^T + ridge I
        H = -Q_b
        H = H - torch.diag_embed(eps_bar / (u.clamp_min(1e-12) ** 2))      # (B,d,d)
        H = H - (eps_bar / (slack ** 2)).view(B, 1, 1) * Jmat
        H = H + ridge * I_b

        rhs = (-g).unsqueeze(-1)                                           # (B,d,1)
        try:
            dlt = torch.linalg.solve(H, rhs).squeeze(-1)                   # (B,d)
        except RuntimeError:
            dlt = (torch.linalg.pinv(H) @ rhs).squeeze(-1)                 # (B,d)

        # fraction-to-the-boundary (배치)
        alpha = torch.ones(B, 1, device=dev, dtype=dt)                     # (B,1)
        neg = dlt < 0
        if neg.any():
            denom = (-dlt).clamp_min(1e-24)
            step_pos = (u / denom) * tau                                   # (B,d)
            step_pos = torch.where(neg, step_pos, torch.full_like(step_pos, float("inf")))
            alpha_pos = step_pos.amin(dim=1, keepdim=True)                 # (B,1)
            alpha = torch.minimum(alpha, alpha_pos)

        dsum = dlt.sum(dim=1, keepdim=True)                                # (B,1)
        needs_sum_cap = dsum > 0
        if needs_sum_cap.any():
            alpha_sum = tau * slack / (dsum + 1e-24)                       # (B,1)
            alpha = torch.minimum(alpha, torch.where(needs_sum_cap, alpha_sum, alpha))

        # Armijo 백트래킹(배치)
        # f(u) = a^T u - 0.5 u^T Q u + eps*sum(log u) + eps*log(slack)
        def Hbar_batch(uu: torch.Tensor) -> torch.Tensor:
            Quu = torch.bmm(Q_b, uu.unsqueeze(-1)).squeeze(-1)             # (B,d)
            val = (a_vec * uu).sum(dim=1, keepdim=True) - 0.5 * (uu * Quu).sum(dim=1, keepdim=True)
            slk = (L_cap - uu.sum(dim=1, keepdim=True))
            feas = (uu > 0).all(dim=1, keepdim=True) & (slk > 0)
            val = val + eps_bar * torch.log(uu.clamp_min(1e-24)).sum(dim=1, keepdim=True)
            val = val + eps_bar * torch.log(slk.clamp_min(1e-24))
            # 비가역 점은 -inf로
            val = torch.where(feas, val, torch.full_like(val, -float("inf")))
            return val                                                     # (B,1)

        f0 = Hbar_batch(u)                                                 # (B,1)
        gTd = (g * dlt).sum(dim=1, keepdim=True)                           # (B,1)
        u_new = u.clone()
        accepted = torch.zeros(B, 1, dtype=torch.bool, device=dev)

        for _ls in range(max_ls):
            uc = u + alpha * dlt                                           # (B,d)
            f1 = Hbar_batch(uc)                                            # (B,1)
            ok = f1 >= (f0 + armijo * alpha * gTd)                         # (B,1) bool
            # 수락된 배치 갱신
            if ok.any():
                mask = ok & (~accepted)
                u_new = torch.where(mask, uc, u_new)
                f0 = torch.where(mask, f1, f0)
                accepted = accepted | ok
            # 미수락 배치의 alpha 축소
            if (~accepted).any():
                alpha = torch.where(accepted, alpha, alpha * back)
            else:
                break

        # 라인서치에서 일부만 갱신됐어도 그 배치들만 반영
        u = torch.where(accepted, u_new, u)

        # 수치 튀김 방지: 경계 근처를 살짝 안쪽으로
        s = u.sum(dim=1, keepdim=True)
        on_edge = s >= (L_cap - 1e-12)
        if on_edge.any():
            scale_in = (L_cap * (1.0 - 1e-8)) / s.clamp_min(1e-12)
            u = torch.where(on_edge, u * scale_in, u)

    return u
