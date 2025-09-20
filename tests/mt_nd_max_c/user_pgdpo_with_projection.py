# tests/<model>/user_pgdpo_with_projection.py
# ---------------------------------------------------------------------
# P-PGDPO 사영(project_pmp): 코스테이트(JX,JXX)를 받아 포트폴리오 u를
# 장벽 뉴턴법으로 즉시 계산합니다.
# - 제약: u >= 0, 1^T u <= L_cap  (무차입)
# - 목적(단계별 해밀토니안 근사):
#     H(u) = s * ( alpha^T u - 0.5 * gamma * u^T Sigma u )
#             + eps_bar * ( sum_i log u_i + log(L_cap - 1^T u) )
#   여기서 s = - JX / (X * JXX)  (CRRA에선 s = 1/gamma에 수렴).
#
# 주: 소비 C는 상대상한 C <= alpha_rel * X 로 베이스에서 처리합니다.
#     본 모듈은 u만 투영합니다.
# ---------------------------------------------------------------------

from __future__ import annotations
import torch

# 베이스 심볼 가져오기
try:
    from user_pgdpo_base import alpha as ALPHA_CONST, Sigma as SIGMA_CONST, Sigma_inv as SIGMAi_CONST, gamma as GAMMA_CONST
except Exception as e:
    raise RuntimeError("[with_projection] user_pgdpo_base에서 alpha, Sigma, Sigma_inv, gamma를 찾지 못했습니다.") from e

# 선택: L_cap, alpha_rel을 가져오되, 없으면 기본값 사용
try:
    from user_pgdpo_base import L_cap as L_CAP_CONST
except Exception:
    L_CAP_CONST = 1.0

# 코어가 필요로 하는 코스테이트 키
PP_NEEDS = ("JX", "JXX")

# 하이퍼파라미터
PP_OPTS = {
    "L_cap": float(L_CAP_CONST),  # 합계 캡
    "eps_bar": 1e-3,              # 장벽 세기
    "ridge": 1e-12,               # 선형계 안정화
    "tau": 0.95,                  # fraction-to-the-boundary
    "armijo": 1e-4,               # Armijo(최대화)
    "backtrack": 0.5,             # 역추적 비율
    "max_newton": 30,             # 뉴턴 반복
    "tol_grad": 1e-8,             # ∥grad∥_∞ 허용오차
    "interior_shrink": 1e-6,      # 초기 interior 여유
}


@torch.no_grad()
def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    """
    입력:
      costates: {"JX": (B,1), "JXX": (B,1)}
      states:   {"X": (B,1), "TmT": (B,1), ...}
    출력:
      u: (B, d)  — 포트폴리오(제약만족)
    """
    alpha = ALPHA_CONST.view(-1)                 # (d,)
    Sigma = SIGMA_CONST                          # (d,d)
    Sigma_inv = SIGMAi_CONST
    gamma = float(GAMMA_CONST)
    d = alpha.numel()
    device = alpha.device

    X = states["X"].view(-1, 1)                  # (B,1)
    JX = costates["JX"].view(-1, 1)              # (B,1)
    JXX = costates["JXX"].view(-1, 1)            # (B,1)

    B = X.size(0)
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

    # 스칼라 계수 s = -JX / (X * JXX) (작은 값 방지)
    denom = (X * JXX).clamp_min(1e-12)
    s = (-JX / denom).view(-1)    # (B,)

    # 초기값: 무제약 마이오픽을 양수/심플렉스로 투영, strict interior로 약간 shrink
    u_unc = (1.0 / gamma) * (Sigma_inv @ alpha)       # (d,)
    u0 = u_unc.clamp_min(0.0)
    ssum = float(u0.sum().item())
    if ssum > L_cap:
        # equality simplex로 투영
        u_sorted = torch.sort(u0, descending=True).values
        cssv = torch.cumsum(u_sorted, dim=0) - L_cap
        j = torch.arange(1, d+1, device=device, dtype=u0.dtype)
        cond = u_sorted > (cssv / j)
        if cond.any():
            rho_idx = int(torch.nonzero(cond, as_tuple=False)[-1].item())
            theta = cssv[rho_idx] / float(rho_idx + 1)
        else:
            theta = cssv[-1] / float(d)
        u0 = (u0 - theta).clamp_min(0.0)
    # strict interior로
    if u0.sum().item() >= L_cap:
        u0 = (L_cap - interior_eps) * u0 / (u0.sum() + 1e-12)
    u0 = u0.clamp_min(interior_eps)

    # 배치별 뉴턴
    u = torch.empty(B, d, device=device, dtype=alpha.dtype)
    # 미리 준비
    chol_S = None  # 미사용이지만, 동일성 유지

    # 내부 함수들
    def Hbar(u_vec: torch.Tensor, sb: float) -> torch.Tensor:
        slack = L_cap - u_vec.sum()
        if slack <= 0 or (u_vec <= 0).any():
            return torch.tensor(float("-inf"), device=device, dtype=u_vec.dtype)
        quad = -0.5 * gamma * (u_vec @ (Sigma @ u_vec))
        lin  = alpha @ u_vec
        return sb * (lin + quad) + eps_bar * (torch.log(u_vec).sum() + torch.log(slack))

    def grad_Hbar(u_vec: torch.Tensor, sb: float) -> torch.Tensor:
        # g = s*(alpha - gamma*Sigma u) + eps*(1/u) - eps/(slack) * 1
        slack = L_cap - u_vec.sum()
        inv_u = 1.0 / u_vec
        g = sb * (alpha - gamma * (Sigma @ u_vec)) + eps_bar * inv_u - (eps_bar / slack) * ones
        return g

    def hess_Hbar(u_vec: torch.Tensor, sb: float) -> torch.Tensor:
        # H = - s*gamma*Sigma - eps*diag(1/u^2) - eps/(slack^2) * 11^T
        slack = L_cap - u_vec.sum()
        H = (- sb * gamma) * Sigma.clone()
        H = H - eps_bar * torch.diag((1.0 / (u_vec ** 2)))
        H = H - (eps_bar / (slack ** 2)) * (ones.view(-1,1) @ ones.view(1,-1))
        return H

    for b in range(B):
        sb = float(s[b].item())
        ub = u0.clone()

        # 뉴턴-백트래킹 반복
        val = Hbar(ub, sb)
        ridge = ridge0
        for it in range(max_newton):
            g = grad_Hbar(ub, sb)
            if torch.isfinite(g).all() and float(g.abs().max().item()) < tol_g:
                break
            H = hess_Hbar(ub, sb)

            # 안정화: H + (-ridge)I (음의 definite 유지), 실패시 pinv fallback
            try:
                # 선형계 풀이 (최대화: H d = -g)
                dlt = torch.linalg.solve(H - ridge * torch.eye(d, device=device, dtype=H.dtype), -g)
            except RuntimeError:
                dlt = torch.linalg.pinv(H) @ (-g)

            # fraction-to-the-boundary
            # u + a d > 0  and  slack(u + a d) > 0
            a_max = 1.0
            neg_idx = (dlt < 0)
            if neg_idx.any():
                a_max = min(a_max, float(( -tau * ub[neg_idx] / dlt[neg_idx] ).min().item()))
            d_slack = - float(dlt.sum().item())
            if d_slack > 0:
                a_max = min(a_max, float(tau * (L_cap - float(ub.sum().item())) / d_slack))
            a = a_max

            # Armijo 백트래킹
            f0 = float(val.item()) if torch.isfinite(val) else float("-inf")
            gTd = float(g.dot(dlt).item())
            accepted = False
            for _ls in range(20):
                uc = ub + a * dlt
                # 엄격 내부 확인
                if (uc > 0).all() and (uc.sum().item() < L_cap):
                    f1t = Hbar(uc, sb)
                    if torch.isfinite(f1t) and (f1t.item() >= f0 + c1 * a * gTd):
                        ub = uc
                        val = f1t
                        accepted = True
                        break
                a *= back
            if not accepted:
                # 진전 없으면 종료
                break

        u[b] = ub

    return u
