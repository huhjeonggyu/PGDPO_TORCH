# core/traj.py
# 목적: CRN(공통 난수)로 동일 초기상태/노이즈에서
#       학습정책(learn)·폐형해(cf)·P-PGDPO(pp) 경로(X/Y/u)를 생성하고,
#       오버레이 플롯(learn–cf, pp–cf)만 저장합니다.

from __future__ import annotations
import os, time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from pgdpo_base import (
    device, T, m, DIM_Y,
    make_generator, sample_initial_states, simulate, _draw_base_normals,
    CRN_SEED_EU,
)

# 사용자 범위(있으면) 가져오기 — X/Y 중점 고정을 위해
try:
    from user_pgdpo_base import X0_range
except Exception:
    X0_range = None
try:
    from user_pgdpo_base import Y0_range, theta_Y
except Exception:
    Y0_range, theta_Y = None, None

# P-PGDPO 유틸(있으면 활성화)
try:
    from pgdpo_with_projection import ppgdpo_u_direct, REPEATS as PP_REPEATS, SUBBATCH as PP_SUBBATCH
    _HAS_PPGDPO = True
except Exception:
    _HAS_PPGDPO = False

from viz import (
    _ensure_dir,
    plot_X_paths_overlay, plot_Y_paths_overlay, plot_u_paths_overlay,
)

# ------------------------------------------------------------
# 공통 래퍼: simulate(...) 호출 시 입력/출력 경로 기록
# ------------------------------------------------------------
class RecordingPolicy(nn.Module):
    def __init__(self, base_policy: nn.Module):
        super().__init__()
        self.base = base_policy
        self.X_buf, self.Y_buf, self.U_buf = [], [], []

    def forward(self, **states):
        X, Y = states.get("X"), states.get("Y")
        U = self.base(**states)
        if X is not None: self.X_buf.append(X.detach())
        if (Y is not None) and DIM_Y > 0: self.Y_buf.append(Y.detach())
        self.U_buf.append(U.detach())
        return U

class PPGDPOPolicy(nn.Module):
    """forward에서 즉시 u_pp=ppgdpo_u_direct(...)를 계산해 반환 + 경로 기록"""
    def __init__(self, stage1_policy: nn.Module, repeats: int, subbatch: int, seed_eval: Optional[int] = None):
        super().__init__()
        if not _HAS_PPGDPO:
            raise RuntimeError("P-PGDPO utilities are not available.")
        self.s1 = stage1_policy
        self.repeats = int(repeats)
        self.subbatch = int(subbatch)
        self.seed_eval = None if seed_eval is None else int(seed_eval)
        self.X_buf, self.Y_buf, self.U_buf = [], [], []

    def forward(self, **states):
        u_pp = ppgdpo_u_direct(self.s1, states, self.repeats, self.subbatch, seed_eval=self.seed_eval)
        X, Y = states.get("X"), states.get("Y")
        if X is not None: self.X_buf.append(X.detach())
        if (Y is not None) and DIM_Y > 0: self.Y_buf.append(Y.detach())
        self.U_buf.append(u_pp.detach())
        return u_pp

def _stack_or_none(seq):  # [B, steps, ·]
    return None if (not seq) else torch.stack(seq, dim=1)

# ------------------ X/Y 중점 계산 ------------------
def _midpoint_X() -> float:
    if X0_range is None:
        return 1.0
    lo, hi = X0_range
    def _val(a):
        if isinstance(a, (int, float)): return float(a)
        if isinstance(a, torch.Tensor):
            return float(a.mean().item() if a.numel() > 1 else a.item())
        return float(a)
    return 0.5 * (_val(lo) + _val(hi))

def _midpoint_Y() -> Optional[torch.Tensor]:
    if Y0_range is not None:
        Ymin, Ymax = Y0_range
        return (Ymin + (Ymax - Ymin) * 0.5)  # torch.Tensor 유지
    if theta_Y is not None:
        return theta_Y
    return None

# ------------------ 공통 초기상태/노이즈 ------------------
def _make_common_states_noise(B: int, steps: int, seed: Optional[int]):
    gen = make_generator(int(seed if seed is not None else CRN_SEED_EU))
    states0, _ = sample_initial_states(B, rng=gen)

    # X0: 구간 중점으로 고정
    if "X" in states0 and states0["X"] is not None:
        xmid = _midpoint_X()
        states0["X"] = torch.full_like(states0["X"], float(xmid), device=device)

    # Y0: 구간 중점(또는 theta_Y)으로 고정
    ymid = _midpoint_Y()
    if ymid is not None and "Y" in states0 and states0["Y"] is not None:
        states0["Y"] = (ymid.unsqueeze(0).expand_as(states0["Y"]).to(device)
                        if isinstance(ymid, torch.Tensor)
                        else torch.full_like(states0["Y"], float(ymid), device=device))

    # 시간: 모든 경로 T에서 시작(균등 dt=T/steps)
    if "TmT" in states0 and states0["TmT"] is not None:
        states0["TmT"] = torch.full_like(states0["TmT"], float(T), device=device)

    ZX, ZY = _draw_base_normals(B, steps, gen)
    return states0, (ZX, ZY)

# ------------------ 경로 시뮬 ------------------
@torch.no_grad()
def simulate_trajectories_policy(
    policy: nn.Module, *, B: int = 5, steps: Optional[int] = None, seed: Optional[int] = None
) -> Dict[str, Optional[torch.Tensor]]:
    steps = int(steps or m)
    states0, noise = _make_common_states_noise(B, steps, seed)
    rec = RecordingPolicy(policy).to(device)
    _ = simulate(rec, B, train=False, initial_states_dict=states0, random_draws=noise, m_steps=steps)
    return {"X": _stack_or_none(rec.X_buf), "Y": _stack_or_none(rec.Y_buf), "U": _stack_or_none(rec.U_buf)}

@torch.no_grad()
def simulate_trajectories_ppgdpo(
    stage1_policy: nn.Module, *, B: int = 5, steps: Optional[int] = None, seed: Optional[int] = None,
    repeats: Optional[int] = None, subbatch: Optional[int] = None
) -> Dict[str, Optional[torch.Tensor]]:
    if not _HAS_PPGDPO:
        raise RuntimeError("pgdpo_with_projection is not available; cannot simulate P-PGDPO trajectories.")
    steps = int(steps or m)
    states0, noise = _make_common_states_noise(B, steps, seed)
    rec = PPGDPOPolicy(
        stage1_policy,
        repeats=int(repeats if repeats is not None else (PP_REPEATS if _HAS_PPGDPO else 1024)),
        subbatch=int(subbatch if subbatch is not None else (PP_SUBBATCH if _HAS_PPGDPO else 64)),
        seed_eval=seed,
    ).to(device)
    _ = simulate(rec, B, train=False, initial_states_dict=states0, random_draws=noise, m_steps=steps)
    return {"X": _stack_or_none(rec.X_buf), "Y": _stack_or_none(rec.Y_buf), "U": _stack_or_none(rec.U_buf)}

def _default_outdir() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots", "_traj", ts)

# ------------------ 엔트리: 그림 저장 ------------------
@torch.no_grad()
def generate_and_save_trajectories(
    policy_learn: nn.Module,
    cf_policy: Optional[nn.Module] = None,
    *,
    B: int = 5,
    seed: Optional[int] = None,
    outdir: Optional[str] = None,
    pp_repeats: Optional[int] = None,
    pp_subbatch: Optional[int] = None,
) -> str:
    """
    저장 파일(두 쌍만 생성):
      - learn vs cf : traj_X_learn_vs_cf.png, traj_Y_learn_vs_cf.png, traj_u_learn_vs_cf.png
      - pp    vs cf : traj_X_pp_vs_cf.png,   traj_Y_pp_vs_cf.png,   traj_u_pp_vs_cf.png
    """
    outdir = _ensure_dir(outdir or _default_outdir())
    steps = int(m)
    t = np.linspace(0.0, float(T), steps)  # 시간축

    # ===== learn =====
    paths_learn = simulate_trajectories_policy(policy_learn, B=B, seed=seed)

    # ===== closed-form → learn vs cf
    if cf_policy is not None:
        paths_cf = simulate_trajectories_policy(cf_policy, B=B, seed=seed)
        plot_X_paths_overlay(paths_learn["X"], paths_cf["X"], outdir,
                             fname="traj_X_learn_vs_cf.png", label_a="learn", label_b="cf", x=t)
        plot_Y_paths_overlay(paths_learn["Y"], paths_cf["Y"], outdir,
                             fname="traj_Y_learn_vs_cf.png", label_a="learn", label_b="cf", x=t)
        plot_u_paths_overlay(paths_learn["U"], paths_cf["U"], outdir,
                             fname="traj_u_learn_vs_cf.png", label_a="learn", label_b="cf", x=t)

        # ===== P-PGDPO → pp vs cf
        if _HAS_PPGDPO:
            try:
                paths_pp = simulate_trajectories_ppgdpo(
                    policy_learn, B=B, seed=seed, repeats=pp_repeats, subbatch=pp_subbatch
                )
                plot_X_paths_overlay(paths_pp["X"], paths_cf["X"], outdir,
                                     fname="traj_X_pp_vs_cf.png", label_a="pp", label_b="cf", x=t)
                plot_Y_paths_overlay(paths_pp["Y"], paths_cf["Y"], outdir,
                                     fname="traj_Y_pp_vs_cf.png", label_a="pp", label_b="cf", x=t)
                plot_u_paths_overlay(paths_pp["U"], paths_cf["U"], outdir,
                                     fname="traj_u_pp_vs_cf.png", label_a="pp", label_b="cf", x=t)
            except Exception as e:
                print(f"[traj] P-PGDPO trajectories skipped: {e}")

    return outdir