# user_pgdpo_base.py — ND, constant opportunity set, dollar-control (time-consistent equilibrium MV)
# -------------------------------------------------------------------------------------------------
# * Equilibrium objective with anchor-group mini-batch:
#   U_{g,i} = X_T^{(g,i)} - (γ/2)(X_T^{(g,i)})^2 + (γ/2)(\bar X_T^{(g)})^2
# * Wealth-independent policy (CIS, k=0). No intertemporal hedging term.
import os, math
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

# =========================
# Required exports (core)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32

d   = int(os.getenv("PGDPO_D", 10))
k   = 0
DIM_X, DIM_Y, DIM_U = 1, k, d

T   = float(os.getenv("PGDPO_T", 1.0))
m   = int(os.getenv("PGDPO_M", 40))

epochs     = int(os.getenv("PGDPO_EPOCHS", 200))
batch_size = int(os.getenv("PGDPO_BATCH_SIZE", 1024))  # core 측 표기용(실제 eq는 G*R 사용)
lr         = float(os.getenv("PGDPO_LR", 1e-3))
seed       = int(os.getenv("PGDPO_SEED", 123))

CRN_SEED_EU   = int(os.getenv("PGDPO_CRN_SEED_EU", 7777))
N_eval_states = int(os.getenv("PGDPO_N_EVAL", 2048))

torch.manual_seed(seed); np.random.seed(seed)

gamma = float(os.getenv("PGDPO_GAMMA", 2.0))


# =========================
# Market (CIS)
# =========================
r        = float(os.getenv("PGDPO_R", 0.02))
sr_mode  = os.getenv("PGDPO_SR_MODE", "equal")  # equal | normal
sr_mean  = float(os.getenv("PGDPO_SR_MEAN", 0.40))
sr_std   = float(os.getenv("PGDPO_SR_STD", 0.10))
rho_eq   = float(os.getenv("PGDPO_RHO", 0.30))  # equi-correlation

pi_cap   = float(os.getenv("PGDPO_PI_CAP", 0.0))
pi_l2_cap= float(os.getenv("PGDPO_PI_L2_CAP", 0.0))

# Anchor-group config
EQ_GROUPS = int(os.getenv("PGDPO_EQ_GROUPS", 16))  # anchors per batch (G)
EQ_REP    = int(os.getenv("PGDPO_EQ_REP", 64))     # replicates per anchor (R)

x0_min, x0_max = float(os.getenv("PGDPO_X0_MIN", 0.5)), float(os.getenv("PGDPO_X0_MAX", 2.0))

def _parse_list_env(name: str, n: int):
    s = os.getenv(name, "").strip()
    if not s: return None
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    assert len(vals) == n, f"{name} must have {n} entries"
    return torch.tensor(vals, device=device, dtype=DTYPE)

def _make_market():
    sigma_env = _parse_list_env("PGDPO_SIGMA", d)
    if sigma_env is None:
        vols = torch.empty(d, device=device, dtype=DTYPE).uniform_(0.15, 0.35)
    else:
        vols = sigma_env
    # equi-corr SPD: ρ ∈ (-1/(d-1),1)
    rho_c = max(min(rho_eq, 0.999), -1.0/(d-1) + 1e-6)
    Psi = torch.full((d,d), rho_c, device=device, dtype=DTYPE); Psi.fill_diagonal_(1.0)
    D = torch.diag(vols)
    Sigma = D @ Psi @ D
    if sr_mode == "equal":
        SR = torch.full((d,), sr_mean, device=device, dtype=DTYPE)
    else:
        SR = torch.normal(sr_mean, sr_std, size=(d,), device=device, dtype=DTYPE)
    alpha = SR * vols  # α_i = SR_i · σ_i
    mu = r + alpha
    return dict(Sigma=Sigma, mu=mu, alpha=alpha, rho=rho_c)

_MKT  = _make_market()
Sigma = _MKT["Sigma"]
alpha = _MKT["alpha"]
G_vec = torch.linalg.solve(Sigma, alpha)  # Σ^{-1}α (참고용)

# =========================
# Policy (wealth-independent)
# =========================
class DirectPolicyEq(nn.Module):
    """MLP: input t_abs → π ∈ R^d (wealth-independent)"""
    def __init__(self):
        super().__init__()
        h = int(os.getenv("PGDPO_HIDDEN", 128))
        self.net = nn.Sequential(
            nn.Linear(1, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, d), nn.Tanh()
        )
        self.register_buffer("pi_cap",    torch.tensor(float(pi_cap)))
        self.register_buffer("pi_l2_cap", torch.tensor(float(pi_l2_cap)))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        허용되는 입력:
          - positional: (t_abs, X)
          - keywords:   t_abs 또는 t, 또는 TmT (=> t_abs = T - TmT)
                        X, gid 등 다른 키워드는 무시
        """
        if len(args) == 2 and not kwargs:
            t_abs, _X = args
        else:
            t_abs = kwargs.get("t_abs", None)
            if t_abs is None:
                t_abs = kwargs.get("t", None)
            if t_abs is None and ("TmT" in kwargs):
                t_abs = T - kwargs["TmT"]

            if t_abs is None:
                # 배치 크기 추정을 위해 임의 텐서 탐색
                any_tensor = None
                for v in kwargs.values():
                    if torch.is_tensor(v):
                        any_tensor = v
                        break
                B = any_tensor.size(0) if (any_tensor is not None and any_tensor.dim() > 0) else 1
                t_abs = torch.zeros((B, 1), device=device, dtype=DTYPE)

        if t_abs.dim() == 1:
            t_abs = t_abs.view(-1, 1)

        t_norm = (t_abs / T).clamp(0, 1)
        pi = self.net(t_norm)
        # optional L2 cap
        if self.pi_l2_cap.item() > 0.0:
            norm = pi.norm(dim=1, keepdim=True) + 1e-12
            scale = torch.clamp(self.pi_l2_cap / norm, max=1.0)
            pi = pi * scale
        if self.pi_cap.item() > 0.0:
            pi = torch.clamp(pi, -self.pi_cap, self.pi_cap)
        return pi


# core가 찾는 이름으로 alias
DirectPolicy = DirectPolicyEq

# projector 호환을 위한 전역 (선택적이지만 안전)
CURRENT_POLICY: Optional[nn.Module] = None
POLICY_VERSION: int = 0

# =========================
# Initial states (anchor groups)
# =========================
def sample_initial_states(B: int, *, rng: Optional[torch.Generator] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    G, R = EQ_GROUPS, EQ_REP
    if B != G*R:
        # B에 맞게 보수적으로 맞춤
        if R > 0: G = max(1, B // R)
        R = max(1, B // G)
        B = G*R
    # t0 ~ U[0,T), TmT = T - t0
    t0  = torch.rand((G,1), device=device, generator=rng) * T
    TmT = (T - t0).repeat_interleave(R, dim=0)  # (B,1)
    X0  = torch.empty((G,1), device=device).uniform_(x0_min, x0_max).repeat_interleave(R, dim=0)
    dt_vec = TmT / float(m)
    gid = torch.arange(G, device=device).repeat_interleave(R)
    states = {"X": X0, "TmT": TmT, "gid": gid}
    return states, dt_vec

# =========================
# Simulate & per-sample scores U (B,1)
# =========================
def _apply_caps(pi: torch.Tensor, l2_cap: float, lin_cap: float) -> torch.Tensor:
    if l2_cap > 0.0:
        norm = pi.norm(dim=1, keepdim=True) + 1e-12
        scale = torch.clamp(l2_cap / norm, max=1.0)
        pi = pi * scale
    if lin_cap > 0.0:
        pi = torch.clamp(pi, -lin_cap, lin_cap)
    return pi

def simulate(policy: nn.Module, B: int, *, train: bool = True,
             rng: Optional[torch.Generator] = None,
             initial_states_dict: Optional[dict] = None,
             random_draws: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
             m_steps: Optional[int] = None) -> torch.Tensor:
    """
    Returns U (B,1) for equilibrium batch estimator.
    random_draws: (ZX, ZY) — we use ZX (B, m_eff) if provided.
    """
    global CURRENT_POLICY, POLICY_VERSION
    CURRENT_POLICY = policy
    POLICY_VERSION += 1

    m_eff = int(m_steps if m_steps is not None else m)

    if initial_states_dict is None:
        states, dt = sample_initial_states(B, rng=rng)
    else:
        states = initial_states_dict
        dt = states["TmT"] / float(m_eff)  # (B,1)

    X   = states["X"].clone()   # (B,1)
    gid = states.get("gid", None)
    t_abs = T - states["TmT"]   # (B,1) 시작 앵커의 절대시간
    # 노이즈 준비
    ZX = None
    if random_draws is not None and random_draws[0] is not None and random_draws[0].shape[1] >= m_eff:
        ZX = random_draws[0]  # (B, m_eff)

    for k_step in range(m_eff):
        TmT_cur = states["TmT"] - k_step * dt  # (B,1)           
        pi = policy(X=X, TmT=TmT_cur, gid=gid)
        pi = _apply_caps(pi, float(pi_l2_cap), float(pi_cap))
        var = (pi @ Sigma @ pi.T).diagonal().clamp_min(1e-12).view(-1,1)  # (B,1)
        # 표준정규
        if ZX is not None:
            z = ZX[:, k_step, 0].unsqueeze(1).contiguous()
        else:
            z = torch.randn(B, 1, device=device, generator=rng, dtype=DTYPE)
        # Euler update (vectorized dt)
        sqrt_dt = torch.sqrt(dt.clamp_min(1e-12))
        X = X + r*X*dt + (pi @ alpha).view(-1,1)*dt + torch.sqrt(var) * sqrt_dt * z
        t_abs = t_abs + dt

    XT = X.view(-1,1)  # (B,1)

    # 그룹 평균 기반 점수
    if gid is None:
        mXT = XT.mean()
        U = XT - 0.5*float(os.getenv("PGDPO_GAMMA", 2.0))*(XT**2) + 0.5*float(os.getenv("PGDPO_GAMMA", 2.0))*(mXT**2)
        return U

    Gmax = int(gid.max().item()) + 1
    ones = torch.ones_like(XT)
    sum_by_g = torch.zeros(Gmax, 1, device=device).index_add_(0, gid, XT)
    cnt_by_g = torch.zeros(Gmax, 1, device=device).index_add_(0, gid, ones)
    mXT_by_g = sum_by_g / cnt_by_g.clamp_min(1.0)

    gamma = float(os.getenv("PGDPO_GAMMA", 2.0))
    U = XT - 0.5*gamma*(XT**2) + 0.5*gamma*(mXT_by_g[gid]**2)
    return U


def build_closed_form_policy():
    """
    코어 러너가 찾는 표준 API.
    성공 시 (policy, meta) 또는 policy 하나를 반환.
    """
    try:
        # tests/equi_mt_nd/closed_form_ref.py 의 빌더를 호출
        from closed_form_ref import build_closed_form_policy as _build
        params = {
            "d": d,
            "r": r,
            "mu": (r + alpha).detach().cpu().tolist() if 'mu' not in globals() else mu.detach().cpu().tolist(),
            "Sigma": Sigma.detach().cpu().numpy(),
            "pi_cap": pi_cap,
        }
        cf, meta = _build(params, T=T, gamma=gamma)  # (policy, meta) 형태
        return cf.to(device), meta
    except Exception as e:
        # 실패 시 로컬 폐형식(참값에 해당하지만 메타에 표시해서 코어가 "reference"로 취급하도록 할 수도 있음)
        class CF(nn.Module):
            def forward(self, *args, **kwargs):
                # t_abs 추출 (t_abs > t > TmT 순서로), 누락 시 배치 크기만큼 0으로 채움
                t_abs = kwargs.get("t_abs", None) or kwargs.get("t", None)
                if t_abs is None and ("TmT" in kwargs):
                    t_abs = T - kwargs["TmT"]
                if t_abs is None:
                    any_tensor = next((v for v in kwargs.values() if torch.is_tensor(v)), None)
                    B = any_tensor.size(0) if (any_tensor is not None and any_tensor.dim() > 0) else 1
                    t_abs = torch.zeros((B, 1), device=device, dtype=DTYPE)
                if t_abs.dim() == 1:
                    t_abs = t_abs.view(-1, 1)

                rho = (1.0 / gamma) * torch.exp(- r * (T - t_abs))
                pi = rho * G_vec.view(1, -1)
                if pi_cap > 0.0:
                    pi = torch.clamp(pi, -pi_cap, pi_cap)
                return pi

        # 메타에 "no true closed-form"를 넣으면 코어가 그래프 비교를 스킵함.
        # 진짜로 비교까지 하고 싶으면 meta를 None 으로 반환하세요.
        return CF().to(device), {"note": "no true closed-form", "error": str(e)}

