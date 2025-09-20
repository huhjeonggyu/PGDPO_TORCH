# core/pgdpo_run.py
# -*- coding: utf-8 -*-
# 역할: Variance-Reduced 러너(안티테틱) + P-PGDPO 사영(런타임 버전) + 시각화/로그 저장

from __future__ import annotations
from typing import Optional, Dict, Iterable, Tuple

import os
import torch
import torch.nn as nn

from pgdpo_base import (
    device, T, m, d, k, N_eval_states,
    make_generator, run_common,
    sample_initial_states, simulate,
    _draw_base_normals,
    batch_size, lr, epochs, seed,
    CRN_SEED_EU,
)

from pgdpo_with_projection import (
    REPEATS, SUBBATCH, PP_NEEDS,
    project_pmp, estimate_costates,
    VERBOSE, SAMPLE_PREVIEW_N
)

# --- ✨ [수정] viz.py의 새로운 통합 함수를 import 하도록 변경 ---
from viz import (
    save_combined_scatter, # save_pairwise_scatters 대신 사용
    append_metrics_csv, save_overlaid_delta_hists,
    save_loss_curve, save_loss_csv
)

PREVIEW_COORDS = int(os.getenv("PGDPO_PREVIEW_COORDS", 3))

def _fmt_coords(label: str, mat: torch.Tensor, i: int, k: int) -> str:
    """행 i에서 앞 k개 좌표를 'label[j]=v' 형태로 이어붙여 문자열 생성"""
    n = mat.size(1)
    K = min(k, n)
    parts = [f"{label}[{j}]={mat[i,j].item():.4f}" for j in range(K)]
    suffix = ", ..." if n > K else ""
    return ", ".join(parts) + suffix

# ------------------------------------------------------------
# Variance-Reduced 시뮬레이터 (학습/평가 공용): antithetic 평균
# (이하 simulate_run, estimate_costates_run, ppgdpo_u_run, _divisors_desc 함수는 변경 없음)
# ------------------------------------------------------------
def simulate_run(
    policy: nn.Module, B: int, *, train: bool = True,
    initial_states_dict: Optional[Dict[str, torch.Tensor]] = None,
    random_draws: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    m_steps: Optional[int] = None, rng: Optional[torch.Generator] = None,
    seed_local: Optional[int] = None, **_kwargs,
) -> torch.Tensor:
    if rng is None: rng = make_generator(seed_local)
    states = (sample_initial_states(B, rng=rng)[0] if initial_states_dict is None else initial_states_dict)
    steps = int(m_steps) if (m_steps is not None) else m
    ZX_c, ZY_c = _draw_base_normals(B, steps, rng)
    U_c_p = simulate(policy, B, train=train, rng=rng, initial_states_dict=states, random_draws=(+ZX_c, +ZY_c), m_steps=steps)
    U_c_m = simulate(policy, B, train=train, rng=rng, initial_states_dict=states, random_draws=(-ZX_c, -ZY_c), m_steps=steps)
    return 0.5 * (U_c_p + U_c_m)

def estimate_costates_run(
    policy_net: nn.Module, initial_states: Dict[str, torch.Tensor],
    repeats: int, sub_batch: int, seed_eval: Optional[int] = None, *,
    needs: Iterable[str] = ("JX", "JXX", "JXY"),
) -> Dict[str, torch.Tensor]:
    return estimate_costates(simulate_run, policy_net, initial_states, repeats=repeats, sub_batch=sub_batch, seed_eval=seed_eval, needs=tuple(needs))

def ppgdpo_u_run(
    policy_s1: nn.Module, states: Dict[str, torch.Tensor],
    repeats: int, sub_batch: int, seed_eval: Optional[int] = None, *,
    needs: Iterable[str] = ("JX", "JXX", "JXY"),
) -> torch.Tensor:
    with torch.enable_grad():
        costates = estimate_costates_run(policy_s1, states, repeats, sub_batch, seed_eval=seed_eval, needs=tuple(needs))
        u = project_pmp(costates, states)
        return u.detach()

def _divisors_desc(n: int):
    return sorted([d for d in range(1, n + 1) if n % d == 0], reverse=True)

# ------------------------------------------------------------
# ✨ [수정] 평가/시각화 함수: 새로운 통합 플롯 함수를 호출하도록 변경
# ------------------------------------------------------------
@torch.no_grad()
def print_policy_rmse_and_samples_run(
    pol_s1: nn.Module,
    pol_cf: nn.Module | None,
    *,
    repeats: int,
    sub_batch: int,
    seed_eval: int | None = None,
    outdir: str | None = None,
    tile: int | None = None,
    enable_pp: bool = True,
    prefix: str = "run",
    needs: Iterable[str] = ("JX", "JXX", "JXY"),
) -> None:
    gen = make_generator(seed_eval or CRN_SEED_EU)
    states_dict, _ = sample_initial_states(N_eval_states, rng=gen)
    u_learn = pol_s1(**states_dict)
    u_pp_run = None
    if enable_pp:
        # Check if any states exist to get the batch size
        valid_states = [v for v in states_dict.values() if v is not None]
        if not valid_states:
            raise ValueError("Cannot determine batch size because all states are None.")
        B = valid_states[0].size(0)

        divisors = _divisors_desc(B)
        start_idx = next((i for i, d in enumerate(divisors) if d <= (tile or B)), 0)
        u_pp_run = torch.empty_like(u_learn)
        for idx in range(start_idx, len(divisors)):
            cur_tile = divisors[idx]
            try:
                for s in range(0, B, cur_tile):
                    e = min(B, s + cur_tile)
                    # ✨ FIX: Add 'if v is not None' to handle cases like sir model's Y state.
                    tile_states = {k: v[s:e] for k, v in states_dict.items() if v is not None}
                    u_pp_run[s:e] = ppgdpo_u_run(pol_s1, tile_states, repeats, sub_batch, seed_eval=(seed_eval if seed_eval is not None else 0), needs=tuple(needs))
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    if idx + 1 < len(divisors): print(f"[Eval] OOM; reducing tile -> {divisors[idx+1]}"); continue
                    else: print("[Eval] OOM; could not reduce tile further."); raise
                else: raise
    u_cf = pol_cf(**states_dict) if pol_cf is not None else None

    # RMSE 출력 및 메트릭 저장
    if u_cf is not None:
        rmse_learn = torch.sqrt(((u_learn - u_cf) ** 2).mean()).item()
        print(f"[Policy RMSE] ||u_learn - u_closed-form||_RMSE: {rmse_learn:.6f}")
        if enable_pp and (u_pp_run is not None):
            rmse_pp = torch.sqrt(((u_pp_run - u_cf) ** 2).mean()).item()
            print(f"[Policy RMSE-PP({prefix})] ||u_pp({prefix}) - u_closed-form||_RMSE: {rmse_pp:.6f}")
        if outdir is not None:
            metrics = {f"rmse_learn_cf_{prefix}": rmse_learn}
            if enable_pp and (u_pp_run is not None):
                metrics[f"rmse_pp_cf_{prefix}"] = rmse_pp
            append_metrics_csv(metrics, outdir)
    else:
        print("[INFO] No closed-form policy provided for comparison.")

    # 샘플 프리뷰
    if VERBOSE:
        n = min(SAMPLE_PREVIEW_N, u_learn.size(0))
        print("\n--- Sample Previews ---")
        for i in range(n):
            parts, vec = [], False
            # 상태 프리뷰
            for k_, v in states_dict.items():
                if v is None:
                    continue
                ts = v[i]
                if ts.numel() > 1:
                    parts.append(f"{k_}[0]={ts[0].item():.3f}"); vec = True
                else:
                    parts.append(f"{k_}={ts.item():.3f}")
            if vec: parts.append("...")
            sstr = ", ".join(parts)
    
            # 다좌표 정책 프리뷰
            if u_cf is not None and enable_pp and (u_pp_run is not None):
                msg = (
                    f"  ({sstr}) -> ("
                    f"{_fmt_coords('u_learn', u_learn, i, PREVIEW_COORDS)}, "
                    f"{_fmt_coords(f'u_pp({prefix})', u_pp_run, i, PREVIEW_COORDS)}, "
                    f"{_fmt_coords('u_cf', u_cf, i, PREVIEW_COORDS)})"
                )
            elif u_cf is not None:
                msg = (
                    f"  ({sstr}) -> ("
                    f"{_fmt_coords('u_learn', u_learn, i, PREVIEW_COORDS)}, "
                    f"{_fmt_coords('u_cf', u_cf, i, PREVIEW_COORDS)})"
                )
            elif enable_pp and (u_pp_run is not None):
                msg = (
                    f"  ({sstr}) -> ("
                    f"{_fmt_coords('u_learn', u_learn, i, PREVIEW_COORDS)}, "
                    f"{_fmt_coords(f'u_pp({prefix})', u_pp_run, i, PREVIEW_COORDS)})"
                )
            else:
                msg = f"  ({sstr}) -> ({_fmt_coords('u_learn', u_learn, i, PREVIEW_COORDS)})"
            print(msg)

    # 그림 저장
    if outdir is not None:
        save_overlaid_delta_hists(u_learn=u_learn, u_pp=u_pp_run, u_cf=u_cf, outdir=outdir, coord=0, fname=f"delta_{prefix}_overlaid_hist.png", bins=60)
        if u_cf is not None:
            save_combined_scatter(
                u_ref=u_cf, u_learn=u_learn, u_pp=u_pp_run,
                outdir=outdir, coord=0, fname=f"scatter_{prefix}_comparison.png"
            )

# ------------------------------------------------------------
# 학습 루프 (변경 없음)
# ------------------------------------------------------------
def train_stage1_run(
    epochs_override: Optional[int] = None, lr_override: Optional[float] = None,
    seed_train: Optional[int] = None, outdir: Optional[str] = None,
) -> nn.Module:
    _epochs = int(epochs_override if epochs_override is not None else epochs)
    _lr = float(lr_override if lr_override is not None else lr)
    _seed = seed_train if seed_train is not None else seed
    try: from user_pgdpo_base import DirectPolicy
    except Exception as e: raise RuntimeError(f"[pgdpo_run] DirectPolicy not found in user_pgdpo_base: {e}")
    policy = DirectPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=_lr)
    loss_hist = []
    for ep in range(1, _epochs + 1):
        opt.zero_grad()
        U = simulate_run(policy, batch_size, seed_local=int(_seed) + ep)
        loss = -U.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if ep % 25 == 0 or ep == 1: print(f"[{ep:04d}] loss={loss.item():.6f}")
        loss_hist.append(float(loss.item()))
    if outdir is not None:
        save_loss_csv(loss_hist, outdir, "loss_history_run.csv")
        save_loss_curve(loss_hist, outdir, "loss_curve_run.png")
    return policy

# ------------------------------------------------------------
# 독립 실행용 (테스트) (변경 없음)
# ------------------------------------------------------------
def main():
    run_common(train_fn=train_stage1_run, rmse_fn=print_policy_rmse_and_samples_run, seed_train=seed, train_kwargs={}, rmse_kwargs={"repeats": REPEATS, "sub_batch": SUBBATCH, "seed_eval": CRN_SEED_EU, "needs" : PP_NEEDS})

if __name__ == "__main__":
    main()

__all__ = ["simulate_run", "estimate_costates_run", "ppgdpo_u_run", "print_policy_rmse_and_samples_run", "train_stage1_run"]