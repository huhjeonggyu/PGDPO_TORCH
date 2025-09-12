# core/pgdpo_base.py
# 공통 러너/헬퍼: 전 모드에서 공통으로 쓰는 유틸과 기본 학습/비교 함수

from __future__ import annotations

import importlib
import random
from typing import Callable, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import os

# -----------------------------------------------------------------------------
# 사용자 모델 심볼 로드 (tests/<model>/user_pgdpo_base.py가 제공)
# -----------------------------------------------------------------------------
try:
    from user_pgdpo_base import (
        device, T, m, d, k,
        DIM_X, DIM_Y, DIM_U,
        epochs, batch_size, lr,
        CRN_SEED_EU,
        sample_initial_states,
        simulate,
        DirectPolicy,
        N_eval_states,
    )
    from user_pgdpo_base import seed as default_seed
    seed = default_seed
except Exception as e:
    raise RuntimeError(f"[pgdpo_base] Failed to import symbols from user_pgdpo_base: {e}")

PGDPO_TRAJ_B  = int(os.getenv("PGDPO_TRAJ_B", 5))

# -----------------------------------------------------------------------------
# RNG 유틸
# -----------------------------------------------------------------------------
def make_generator(seed_local: Optional[int] = None) -> torch.Generator:
    if seed_local is None: seed_local = 0
    try: gen = torch.Generator(device=device)
    except Exception: gen = torch.Generator()
    gen.manual_seed(int(seed_local))
    return gen

def set_global_seeds(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def _draw_base_normals(B: int, steps: int, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    Z = torch.randn(B, steps, d + k, device=device, generator=gen)
    ZX, ZY = Z[:, :, :d], Z[:, :, d:]
    return ZX, ZY

# -----------------------------------------------------------------------------
# 공통 실행기
# -----------------------------------------------------------------------------
def run_common(
    *,
    train_fn: Callable[..., nn.Module],
    rmse_fn: Callable[..., Any],
    seed_train: Optional[int] = None,
    train_kwargs: Optional[Dict[str, Any]] = None,
    rmse_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    set_global_seeds(int(seed_train) if seed_train is not None else int(default_seed))
    train_kwargs = train_kwargs or {}
    rmse_kwargs = rmse_kwargs or {}

    policy = train_fn(**train_kwargs)

    cf_policy = None
    try:
        upb = importlib.import_module("user_pgdpo_base")
        if hasattr(upb, "build_closed_form_policy") and callable(upb.build_closed_form_policy):
            res = upb.build_closed_form_policy()
            cf_policy = (res[0] if isinstance(res, tuple) else res).to(device)
            print("✅ Closed-form policy loaded via build_closed_form_policy()")
    except Exception as e:
        print(f"[WARN] build_closed_form_policy() failed: {e}")

    rmse_fn(policy, cf_policy, **rmse_kwargs)

    try:
        from core.traj import generate_and_save_trajectories
        saved = generate_and_save_trajectories(
            policy_learn=policy, 
            policy_cf=cf_policy, 
            B=PGDPO_TRAJ_B,
            seed_crn=int(CRN_SEED_EU),  # 'seed=' -> 'seed_crn='으로 수정
            outdir=rmse_kwargs.get("outdir", None),
        )
        print(f"[traj] saved CRN trajectories to: {saved}")
    except Exception as e:
        import traceback
        print(f"[traj] skipped due to error: {e}")
        traceback.print_exc()

# -----------------------------------------------------------------------------
# 기본 학습 루프(Base)
# -----------------------------------------------------------------------------
def train_stage1_base(
    epochs_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    seed_train: Optional[int] = None,
    outdir: Optional[str] = None,
) -> nn.Module:
    """
    기본 시뮬레이터 기반 Stage-1 학습 (분산감소/프로젝션 없음).
    - seed_local을 직접 넘기지 않고, 초기상태/노이즈를 생성해 simulate(...)에 전달.
    """
    _epochs = int(epochs_override if epochs_override is not None else epochs)
    _lr = float(lr_override if lr_override is not None else lr)

    policy = DirectPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=_lr)

    loss_hist: list[float] = []
    for ep in range(1, _epochs + 1):
        opt.zero_grad()

        # 재현성 있는 초기상태 + 노이즈
        gen = make_generator((seed_train or 0) + ep)
        states, _ = sample_initial_states(batch_size, rng=gen)
        ZX, ZY = _draw_base_normals(batch_size, m, gen)

        # simulate는 다음 형태를 가정:
        #   U = simulate(policy, B, initial_states_dict=states, random_draws=(ZX, ZY), m_steps=m)
        U = simulate(
            policy,
            batch_size,
            initial_states_dict=states,
            random_draws=(ZX, ZY),
            m_steps=m,
        )

        loss = -U.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            print(f"[{ep:04d}] loss={loss.item():.6f}")
        loss_hist.append(float(loss.item()))

    # (옵션) 손실 저장
    if outdir is not None:
        try:
            from viz import save_loss_curve, save_loss_csv
            save_loss_csv(loss_hist, outdir, "loss_history.csv")
            save_loss_curve(loss_hist, outdir, "loss_curve.png")
        except Exception as e:
            print(f"[WARN] base: could not save loss plots: {e}")

    return policy

# -----------------------------------------------------------------------------
# base 모드 평가기: learn↔cf 비교/플롯
# -----------------------------------------------------------------------------
@torch.no_grad()
def print_policy_rmse_and_samples_base(
    pol_s1: nn.Module,
    pol_cf: Optional[nn.Module],
    *,
    repeats: int = 0,            # run.py 호환용(미사용)
    sub_batch: int = 0,          # run.py 호환용(미사용)
    seed_eval: Optional[int] = None,
    tile: Optional[int] = None,  # 호환 인자(미사용)
    outdir: Optional[str] = None,
) -> None:
    gen = make_generator(seed_eval or CRN_SEED_EU)
    states_dict, _ = sample_initial_states(N_eval_states, rng=gen)

    u_learn = pol_s1(**states_dict)

    if pol_cf is None:
        print("[INFO] No closed-form policy available; skipping base comparison.")
        return

    u_cf = pol_cf(**states_dict)
    rmse = torch.sqrt(((u_learn - u_cf) ** 2).mean()).item()
    print(f"[Policy RMSE] ||u_learn - u_closed-form||_RMSE: {rmse:.6f}")

    if outdir is not None:
        try:
            from viz import save_policy_scatter, save_overlaid_delta_hists, append_metrics_csv
            save_policy_scatter(
                u_ref=u_cf, u_pred=u_learn, outdir=outdir, coord=0,
                fname="scatter_base_learn_vs_cf_dim0.png", xlabel="u_cf", ylabel="u_learn",
            )
            save_overlaid_delta_hists(
                u_learn=u_learn, u_pp=None, u_cf=u_cf,
                outdir=outdir, coord=0, fname="delta_base_overlaid_hist.png", bins=60, density=True,
            )
            append_metrics_csv({"rmse_learn_cf_base": rmse}, outdir)
        except Exception as e:
            print(f"[WARN] base: could not save compare plots: {e}")

    # 대표 샘플 3개 출력
    B = next(iter(states_dict.values())).size(0)
    for i in [0, B // 2, B - 1]:
        parts, vec = [], False
        for k_, v in states_dict.items():
            ts = v[i]
            if ts.numel() > 1:
                parts.append(f"{k_}[0]={ts[0].item():.3f}"); vec = True
            else:
                parts.append(f"{k_}={ts.item():.3f}")
        if vec: parts.append("...")
        sstr = ", ".join(parts)
        print(f"  ({sstr}) -> (u_learn[0]={u_learn[i,0].item():.4f}, u_cf[0]={u_cf[i,0].item():.4f}, ...)")

# -----------------------------------------------------------------------------
# (선택) 간단 비교기 유지
# -----------------------------------------------------------------------------
@torch.no_grad()
def compare_policy_functions(
    policy: nn.Module,
    cf_policy: Optional[nn.Module],
    *,
    seed_eval: Optional[int] = None,
    outdir: Optional[str] = None,
) -> None:
    gen = make_generator(seed_eval or CRN_SEED_EU)
    states_dict, _ = sample_initial_states(N_eval_states, rng=gen)

    u_learn = policy(**states_dict)
    if cf_policy is None:
        print("[INFO] No closed-form policy available; skipping base comparison.")
        return

    u_cf = cf_policy(**states_dict)
    rmse = torch.sqrt(((u_learn - u_cf) ** 2).mean()).item()
    print(f"[Policy RMSE] ||u_learn - u_closed-form||_RMSE: {rmse:.6f}")

    if outdir is not None:
        try:
            from viz import save_policy_scatter, save_overlaid_delta_hists, append_metrics_csv
            save_policy_scatter(
                u_ref=u_cf, u_pred=u_learn, outdir=outdir, coord=0,
                fname="scatter_base_learn_vs_cf_dim0.png", xlabel="u_cf", ylabel="u_learn",
            )
            save_overlaid_delta_hists(
                u_learn=u_learn, u_pp=None, u_cf=u_cf,
                outdir=outdir, coord=0, fname="delta_base_overlaid_hist.png", bins=60, density=True,
            )
            append_metrics_csv({"rmse_learn_cf_base": rmse}, outdir)
        except Exception as e:
            print(f"[WARN] base: could not save compare plots: {e}")

# -----------------------------------------------------------------------------
# __all__
# -----------------------------------------------------------------------------
__all__ = [
    # 환경/격자/차원/하이퍼
    "device", "T", "m", "d", "k", "epochs", "batch_size", "lr", "seed",
    "CRN_SEED_EU", "N_eval_states",
    "DIM_X", "DIM_Y", "DIM_U",
    # 유틸
    "make_generator", "set_global_seeds", "_draw_base_normals",
    # 사용자 훅/심볼 재노출
    "sample_initial_states", "simulate", "DirectPolicy",
    # 공통 실행기/루프/비교
    "run_common",
    "train_stage1_base",
    "print_policy_rmse_and_samples_base",
    "compare_policy_functions",
]