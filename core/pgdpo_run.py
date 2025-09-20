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
# 파일: core/pgdpo_run.py

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
    u_learn_full = pol_s1(**states_dict)
    
    u_pp_run = None
    if enable_pp:
        valid_states = [v for v in states_dict.values() if v is not None]
        if not valid_states:
            raise ValueError("Cannot determine batch size because all states are None.")
        B = valid_states[0].size(0)

        divisors = _divisors_desc(B)
        start_idx = next((i for i, d in enumerate(divisors) if d <= (tile or B)), 0)
        u_pp_run_calc = torch.empty(B, d, device=u_learn_full.device)
        for idx in range(start_idx, len(divisors)):
            cur_tile = divisors[idx]
            try:
                for s in range(0, B, cur_tile):
                    e = min(B, s + cur_tile)
                    tile_states = {k: v[s:e] for k, v in states_dict.items() if v is not None}
                    u_pp_run_calc[s:e] = ppgdpo_u_run(pol_s1, tile_states, repeats, sub_batch, seed_eval=(seed_eval if seed_eval is not None else 0), needs=tuple(needs))
                u_pp_run = u_pp_run_calc
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    if idx + 1 < len(divisors): print(f"[Eval] OOM; reducing tile -> {divisors[idx+1]}"); continue
                    else: print("[Eval] OOM; could not reduce tile further."); raise
                else: raise

    u_cf_full = pol_cf(**states_dict) if pol_cf is not None else None

    is_consumption_model = u_learn_full.size(1) > d
    
    u_learn, c_learn = (u_learn_full[:, :d], u_learn_full[:, d:]) if is_consumption_model else (u_learn_full, None)
    u_pp, c_pp = (u_pp_run, c_learn) if u_pp_run is not None else (None, None)
    u_cf, c_cf = (None, None)
    if u_cf_full is not None:
        u_cf, c_cf = (u_cf_full[:, :d], u_cf_full[:, d:]) if is_consumption_model else (u_cf_full, None)

    # --- (핵심 수정) RMSE 출력 및 메트릭 저장 (투자 u와 소비 c 모두) ---
    if u_cf is not None:
        # 투자(u) RMSE
        rmse_learn_u = torch.sqrt(((u_learn - u_cf) ** 2).mean()).item()
        print(f"[Policy RMSE (u)] ||u_learn - u_closed-form||_RMSE: {rmse_learn_u:.6f}")
        
        metrics = {f"rmse_learn_cf_u_{prefix}": rmse_learn_u}

        if enable_pp and (u_pp is not None):
            rmse_pp_u = torch.sqrt(((u_pp - u_cf) ** 2).mean()).item()
            print(f"[Policy RMSE-PP (u)] ||u_pp({prefix}) - u_closed-form||_RMSE: {rmse_pp_u:.6f}")
            metrics[f"rmse_pp_cf_u_{prefix}"] = rmse_pp_u

        # 소비(c) RMSE (소비 모델인 경우에만)
        if is_consumption_model and c_cf is not None:
            rmse_learn_c = torch.sqrt(((c_learn - c_cf) ** 2).mean()).item()
            print(f"[Policy RMSE (C)] ||c_learn - c_closed-form||_RMSE: {rmse_learn_c:.6f}")
            metrics[f"rmse_learn_cf_c_{prefix}"] = rmse_learn_c

            # P-PGDPO의 소비는 learn의 소비를 따르므로, c_pp vs c_cf RMSE도 계산
            if c_pp is not None:
                rmse_pp_c = torch.sqrt(((c_pp - c_cf) ** 2).mean()).item()
                print(f"[Policy RMSE-PP (C)] ||c_pp({prefix}) - c_closed-form||_RMSE: {rmse_pp_c:.6f}")
                metrics[f"rmse_pp_cf_c_{prefix}"] = rmse_pp_c
        
        if outdir is not None:
            append_metrics_csv(metrics, outdir)
    else:
        print("[INFO] No closed-form policy provided for comparison.")

    # --- (이하 샘플 프리뷰 및 그림 저장은 이전과 동일) ---
    if VERBOSE:
        n = min(SAMPLE_PREVIEW_N, u_learn.size(0))
        print("\n--- Sample Previews ---")
        for i in range(n):
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
    
            msg_parts = [f"({sstr}) -> ("]
            msg_parts.append(_fmt_coords('u_learn', u_learn, i, PREVIEW_COORDS))
            if c_learn is not None: msg_parts.append(f", c_learn={c_learn[i].item():.4f}")

            if u_pp is not None:
                msg_parts.append(", " + _fmt_coords(f'u_pp({prefix})', u_pp, i, PREVIEW_COORDS))

            if u_cf is not None:
                msg_parts.append(", " + _fmt_coords('u_cf', u_cf, i, PREVIEW_COORDS))
                if c_cf is not None: msg_parts.append(f", c_cf={c_cf[i].item():.4f}")
            
            msg_parts.append(")")
            print("".join(msg_parts))

    if outdir is not None:
        save_overlaid_delta_hists(u_learn=u_learn, u_pp=u_pp, u_cf=u_cf, outdir=outdir, coord=0, fname=f"delta_u_{prefix}_overlaid_hist.png", bins=60)
        if u_cf is not None:
            save_combined_scatter(
                u_ref=u_cf, u_learn=u_learn, u_pp=u_pp,
                outdir=outdir, coord=0, fname=f"scatter_u_{prefix}_comparison.png"
            )
        if is_consumption_model and c_learn is not None and c_cf is not None:
             save_combined_scatter(
                u_ref=c_cf, u_learn=c_learn, u_pp=c_pp,
                outdir=outdir, coord=0, fname=f"scatter_c_{prefix}_comparison.png",
                xlabel="c_closed-form", title="Consumption Comparison"
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