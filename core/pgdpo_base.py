# 파일: core/pgdpo_base.py

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

# --- (수정) 순환 참조를 유발하던 최상단 import 제거 ---

# -----------------------------------------------------------------------------
# 사용자 모델 심볼 로드
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

PREVIEW_COORDS = int(os.getenv("PGDPO_PREVIEW_COORDS", 3))

def _fmt_coords(label: str, mat: torch.Tensor, i: int, k: int) -> str:
    n = mat.size(1)
    K = min(k, n)
    parts = [f"{label}[{j}]={mat[i,j].item():.4f}" for j in range(K)]
    suffix = ", ..." if n > K else ""
    return ", ".join(parts) + suffix

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
    effective_seed = int(seed_train) if seed_train is not None else int(default_seed)
    set_global_seeds(effective_seed)
    train_kwargs = train_kwargs or {}
    train_kwargs['seed_train'] = effective_seed

    rmse_kwargs = rmse_kwargs or {}
    policy = train_fn(**train_kwargs)
    policy_for_comparison = None

    try:
        upb = importlib.import_module("user_pgdpo_base")
        if hasattr(upb, "build_closed_form_policy") and callable(upb.build_closed_form_policy):
            res = upb.build_closed_form_policy()
            if res and res[0] is not None:
                policy_obj, meta = res if isinstance(res, tuple) and len(res) > 1 else (res, None)
                is_true_cf = not (isinstance(meta, dict) and "no true closed-form" in meta.get("note", ""))
                if is_true_cf:
                    policy_for_comparison = policy_obj.to(device)
                    print("✅ True closed-form policy loaded via build_closed_form_policy().")
                else:
                    print("✅ Reference policy loaded. All comparisons will be skipped.")
    except Exception as e:
        print(f"[WARN] build_closed_form_policy() loading failed: {e}")

    rmse_fn(policy, policy_for_comparison, **rmse_kwargs)

    try:
        from core.traj import generate_and_save_trajectories
        saved = generate_and_save_trajectories(
            policy_learn=policy,
            policy_cf=policy_for_comparison,
            seed_crn=int(CRN_SEED_EU),
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
    _epochs = int(epochs_override if epochs_override is not None else epochs)
    _lr = float(lr_override if lr_override is not None else lr)
    policy = DirectPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=_lr)
    loss_hist: list[float] = []
    for ep in range(1, _epochs + 1):
        opt.zero_grad()
        gen = make_generator((seed_train or 0) + ep)
        states, _ = sample_initial_states(batch_size, rng=gen)
        ZX, ZY = _draw_base_normals(batch_size, m, gen)
        U = simulate(policy, batch_size, initial_states_dict=states, random_draws=(ZX, ZY), m_steps=m)
        loss = -U.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if ep % 25 == 0 or ep == 1:
            print(f"[{ep:04d}] loss={loss.item():.6f}")
        loss_hist.append(float(loss.item()))
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
    repeats: int = 0,
    sub_batch: int = 0,
    seed_eval: Optional[int] = None,
    tile: Optional[int] = None,
    outdir: Optional[str] = None,
) -> None:
    # --- (수정) 필요한 함수들을 함수 내에서 import ---
    try:
        from viz import append_metrics_csv, save_combined_scatter
    except ImportError as e:
        print(f"[WARN] Could not import visualization functions from viz: {e}")
        return

    gen = make_generator(seed_eval or CRN_SEED_EU)
    states_dict, _ = sample_initial_states(N_eval_states, rng=gen)
    u_learn_full = pol_s1(**states_dict)
    u_cf_full = pol_cf(**states_dict) if pol_cf is not None else None
    is_consumption_model = u_learn_full.size(1) > d
    u_learn, c_learn = (u_learn_full[:, :d], u_learn_full[:, d:]) if is_consumption_model else (u_learn_full, None)
    u_cf, c_cf = (None, None)
    if u_cf_full is not None:
        u_cf, c_cf = (u_cf_full[:, :d], u_cf_full[:, d:]) if is_consumption_model else (u_cf_full, None)

    if u_cf is not None:
        rmse_learn_u = torch.sqrt(((u_learn - u_cf) ** 2).mean()).item()
        print(f"[Policy RMSE (u)] ||u_learn - u_closed-form||_RMSE: {rmse_learn_u:.6f}")
        metrics = {"rmse_learn_cf_u_base": rmse_learn_u}
        if is_consumption_model and c_cf is not None:
            rmse_learn_c = torch.sqrt(((c_learn - c_cf) ** 2).mean()).item()
            print(f"[Policy RMSE (C)] ||c_learn - c_closed-form||_RMSE: {rmse_learn_c:.6f}")
            metrics["rmse_learn_cf_c_base"] = rmse_learn_c
        if outdir is not None:
            append_metrics_csv(metrics, outdir)
    else:
        print("[INFO] No closed-form policy available; skipping base comparison.")

    B = u_learn.size(0)
    for i in [0, B // 2, B - 1]:
        parts, vec = [], False
        for k_, v in states_dict.items():
            if v is None: continue
            ts = v[i]
            if ts.numel() > 1:
                parts.append(f"{k_}[0]={ts[0].item():.3f}"); vec = True
            else:
                parts.append(f"{k_}={ts.item():.3f}")
        if vec: parts.append("...")
        sstr = ", ".join(parts)
        msg_parts = [f"  ({sstr}) -> ("]
        msg_parts.append(_fmt_coords('u_learn', u_learn, i, PREVIEW_COORDS))
        if c_learn is not None: msg_parts.append(f", c_learn={c_learn[i].item():.4f}")
        if u_cf is not None:
            msg_parts.append(", " + _fmt_coords('u_cf', u_cf, i, PREVIEW_COORDS))
            if c_cf is not None: msg_parts.append(f", c_cf={c_cf[i].item():.4f}")
        msg_parts.append(")")
        print("".join(msg_parts))

    if outdir is not None and u_cf is not None:
        try:
            save_combined_scatter(
                u_ref=u_cf, u_learn=u_learn, u_pp=None,
                outdir=outdir, coord=0,
                fname="scatter_base_learn_vs_cf_u_dim0.png", xlabel="u_cf"
            )
            if is_consumption_model and c_cf is not None and c_learn is not None:
                save_combined_scatter(
                    u_ref=c_cf, u_learn=c_learn, u_pp=None,
                    outdir=outdir, coord=0,
                    fname="scatter_base_learn_vs_cf_c.png", xlabel="c_cf", title="Consumption Comparison"
                )
        except Exception as e:
            print(f"[WARN] base: could not save compare plots: {e}")

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
            from viz import save_combined_scatter, save_overlaid_delta_hists, append_metrics_csv
            save_combined_scatter(
                u_ref=u_cf, u_learn=u_learn, u_pp=None,  # u_pp=None으로 전달
                outdir=outdir, coord=0,
                fname="scatter_base_learn_vs_cf_dim0.png", xlabel="u_cf"
            )
            save_overlaid_delta_hists(
                u_learn=u_learn, u_pp=None, u_cf=u_cf,
                outdir=outdir, coord=0, fname="delta_base_overlaid_hist.png", bins=60,
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
    "make_generator", "_draw_base_normals",
    # 사용자 훅/심볼 재노출
    "sample_initial_states", "simulate",
    # 공통 실행기/루프/비교
    "run_common",
    "train_stage1_base",
    "print_policy_rmse_and_samples_base",
    #"compare_policy_functions",
]