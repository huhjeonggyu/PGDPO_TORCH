# tests/mt_nd_short/user_pgdpo_with_projection.py
# ------------------------------------------------------------
# Barrier–Newton projector (RUN/PROJECTION 공통 사용)
# 목적: ∇_u \tilde{H}_bar(u)=0 정지점을 직접 만족 (u_i>0, 합제약 없음)
# 핵심: c = γ X JX (CRRA 항등식)로 스케일 고정 + 초소형 장벽 eps_bar
# ------------------------------------------------------------
import os
import torch
from user_pgdpo_base import alpha as ALPHA_CONST, Sigma as SIGMA_CONST, gamma as GAMMA_CONST

# 코어가 추정할 코스테이트
PP_NEEDS = ("JX", "JXX")  # JXX는 받아오지만 여기선 c 계산에 사용하지 않음

# 환경 파라미터(원하면 env로 조정)
EPS_BAR      = float(os.getenv("PGDPO_EPS_BAR", "1e-8"))   # 장벽 세기
GRAD_TOL     = float(os.getenv("PGDPO_GRAD_TOL", "1e-6"))
MAX_ITERS    = int(os.getenv("PGDPO_NEWTON_ITERS", "20"))
TAU_FTB      = float(os.getenv("PGDPO_TAU_FTB", "0.99"))   # fraction-to-boundary
BETA_BACK    = float(os.getenv("PGDPO_BACKTRACK_BETA", "0.5"))
ARMIJO_SIGMA = float(os.getenv("PGDPO_ARMIJO_SIGMA", "1e-4"))

@torch.no_grad()
def project_pmp(costates: dict, states: dict) -> torch.Tensor:
    # ----- 배치 파라미터 -----
    X = states["X"].view(-1)                     # (B,)
    B = X.shape[0]

    alpha = states.get("alpha", ALPHA_CONST).to(X.device)
    Sigma = states.get("Sigma", SIGMA_CONST).to(X.device)
    if alpha.dim() == 1: alpha = alpha.unsqueeze(0).expand(B, -1)       # (B,n)
    if Sigma.dim() == 2: Sigma = Sigma.unsqueeze(0).expand(B, -1, -1)   # (B,n,n)
    n = alpha.shape[1]

    JX  = costates["JX"].view(-1)                # λ(=V_x) (B,)
    # JXX = costates["JXX"].view(-1)             # 받긴 받지만 사용 안 함

    # ----- CRRA 스케일: c = γ X JX  (해밀토니안 u-곡률) -----
    gamma_const = float(GAMMA_CONST)
    c = gamma_const * X * JX
    c = torch.clamp(c, min=1e-10)                # 수치 안전

    # ----- 초기값: 무제약 해 → 음수 0으로 잘라 strictly-feasible -----
    # u_unc = (1/γ) Σ^{-1} α  (배치 선형시스템 풀이)
    try:
        rhs   = (1.0 / gamma_const) * alpha                              # (B,n)
        u_unc = torch.linalg.solve(Sigma, rhs.unsqueeze(-1)).squeeze(-1) # (B,n)
    except Exception:
        # 느리지만 안전한 fallback
        u_unc = (1.0 / gamma_const) * (torch.linalg.inv(Sigma) @ alpha.transpose(1,0)).transpose(1,0)
    u = torch.clamp(u_unc, min=1e-6)

    # ----- 보조: 장벽 해밀토니안 값 -----
    def hbar_value(u_):
        # \tilde{H}_bar(u) = XJX (α·u) - 0.5 c (u^T Σ u) - eps ∑ log u
        XuJ = X * JX
        term_lin  = torch.einsum("b,bi,bi->b", XuJ, alpha, u_)
        Su        = torch.einsum("bij,bj->bi", Sigma, u_)
        term_quad = 0.5 * c * torch.einsum("bi,bi->b", u_, Su)
        term_bar  = EPS_BAR * torch.sum(torch.log(torch.clamp(u_, min=1e-12)), dim=1)
        return term_lin - term_quad - term_bar

    # ----- 뉴턴 루프 (대각 프리컨디셔너) -----
    diagS = torch.diagonal(Sigma, dim1=-2, dim2=-1)  # (B,n)
    for _ in range(MAX_ITERS):
        XuJ = X * JX
        Su  = torch.einsum("bij,bj->bi", Sigma, u)
        grad = XuJ.unsqueeze(1)*alpha - c.unsqueeze(1)*Su + EPS_BAR/torch.clamp(u, min=1e-12)
        if grad.norm(dim=1).median() < GRAD_TOL:
            break

        # M ≈ c*diag(Σ) + eps/u^2  (좌표별 뉴턴 근사)
        Mdiag = c.unsqueeze(1)*diagS + EPS_BAR/torch.clamp(u**2, min=1e-12)
        d = -grad / torch.clamp(Mdiag, min=1e-8)

        # fraction-to-the-boundary
        tmp   = torch.where(d < 0, -TAU_FTB * u / torch.clamp(d, max=-1e-16), torch.full_like(u, 1e9))
        amax  = torch.clamp(tmp.min(dim=1).values, max=1.0)
        a     = torch.where(torch.isfinite(amax), amax, torch.ones_like(amax))

        # Armijo (maximize)
        H0 = hbar_value(u)
        for _ls in range(20):
            u_new = torch.clamp(u + a.view(-1,1)*d, min=1e-12)
            H1 = hbar_value(u_new)
            rhs = H0 + ARMIJO_SIGMA * a * torch.einsum("bi,bi->b", grad, d)
            if (H1 >= rhs).all():
                u = u_new
                break
            a = a * BETA_BACK

    return u