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
    policy_learn,
    policy_for_comparison,
    *,
    repeats,
    sub_batch,
    seed_eval=None,
    needs=("JX", "JXX"),
    outdir=None,
    **kwargs,
):
    """
    Variance-Reduced RUN 경로에서 u_pp(run)과 닫힌형을 비교해 RMSE/프리뷰를 출력.
    (소비 C 포함/미포함 자동 처리, tests/<model>/user_pgdpo_base.py에
     make_eval_states가 없으면 기본 평가셋을 내부에서 생성)
    """
    import os, torch
    import torch.nn.functional as F

    # ---------- 유틸 ----------
    def _align_cols(t: torch.Tensor, target_cols: int) -> torch.Tensor:
        if t.shape[1] == target_cols:
            return t
        if t.shape[1] > target_cols:
            return t[:, :target_cols]
        pad = torch.zeros(t.shape[0], target_cols - t.shape[1], device=t.device, dtype=t.dtype)
        return torch.cat([t, pad], dim=1)

    def _infer_asset_dim(states: dict, d_out_guess: int) -> int:
        if "alpha" in states:
            a = states["alpha"]
            if a.dim() == 1:
                return int(a.shape[0])
            if a.dim() >= 2:
                return int(a.shape[-1])
        if "Sigma" in states:
            S = states["Sigma"]
            if S.dim() == 2:
                return int(S.shape[-1])
            if S.dim() == 3:
                return int(S.shape[-1])
        return int(d_out_guess)

    def _vec_head(name: str, V: torch.Tensor, i: int, k: int) -> str:
        k = min(k, V.shape[1])
        return ", ".join(f"{name}[{j}]={float(V[i, j]):.4f}" for j in range(k))

    def _default_make_eval_states(device, N=100):
        # k=0 계열도 동작하도록 X, TmT만 생성 (범위는 넉넉히)
        X   = torch.linspace(0.2, 2.5, N, device=device).view(N, 1)
        TmT = torch.linspace(0.0, 1.5, N, device=device).view(N, 1)
        return {"X": X, "TmT": TmT}

    # ---------- 평가 상태 생성 ----------
    device = next(policy_learn.parameters()).device
    try:
        from user_pgdpo_base import make_eval_states as _make_eval_states
        eval_states = _make_eval_states(device)
    except Exception:
        N_eval = int(os.getenv("PGDPO_EVAL_N", "100"))
        eval_states = _default_make_eval_states(device, N=N_eval)

    N_eval = int(eval_states["X"].shape[0])

    # ---------- 출력 차원 탐지 ----------
    with torch.no_grad():
        probe = {k: v[:1] for k, v in eval_states.items()}
        try:
            d_out = int(policy_for_comparison(**probe).shape[1])
        except Exception:
            d_out = int(policy_learn(**probe).shape[1])

    # ---------- u_pp(run) 계산(청크 처리) ----------
    CHUNK = int(os.getenv("PGDPO_EVAL_CHUNK", "100"))
    u_pp_run_calc = torch.empty((N_eval, d_out), device=device)
    pol_s1 = policy_learn
    seval = (seed_eval if seed_eval is not None else 0)

    for s in range(0, N_eval, CHUNK):
        e = min(s + CHUNK, N_eval)
        tile_states = {k: v[s:e] for k, v in eval_states.items()}
        out = ppgdpo_u_run(
            pol_s1,
            tile_states,
            repeats,
            sub_batch,
            seed_eval=seval,
            needs=tuple(needs),
        )
        u_pp_run_calc[s:e] = _align_cols(out, d_out)

    # ---------- 학습/닫힌형 ----------
    with torch.no_grad():
        u_learn = _align_cols(policy_learn(**eval_states).detach(), d_out)
        u_cf    = _align_cols(policy_for_comparison(**eval_states).detach(), d_out)

    # ---------- 소비 포함 여부 판정 (robust) ----------
    # 1) tests/<model>/user_pgdpo_base.py의 alpha 길이로 d 힌트 확보
    d_asset_hint = None
    try:
        from user_pgdpo_base import alpha as _ALPHA_HINT
        d_asset_hint = int(_ALPHA_HINT.numel())
    except Exception:
        pass

    # 2) 환경변수로 강제 지정 가능 (0/1)
    env_has_c = os.getenv("PGDPO_HAS_CONSUMPTION", "")

    if env_has_c != "":
        has_consumption = (env_has_c != "0")
        d_asset = (d_out - 1) if has_consumption else d_out
    else:
        d_asset = d_asset_hint if d_asset_hint is not None else _infer_asset_dim(eval_states, d_out)
        # d_out이 d 또는 d+1 중 하나라고 가정
        has_consumption = (d_out == d_asset + 1) or (d_out > d_asset and d_asset >= 1)

    # ---------- RMSE ----------
    if has_consumption and d_asset >= 1 and d_out >= d_asset + 1:
        d = d_asset
        rmse_u     = F.mse_loss(u_learn[:, :d],       u_cf[:, :d]).sqrt().item()
        rmse_pp_u  = F.mse_loss(u_pp_run_calc[:, :d], u_cf[:, :d]).sqrt().item()
        rmse_c     = F.mse_loss(u_learn[:, d:],       u_cf[:, d:]).sqrt().item()
        rmse_pp_c  = F.mse_loss(u_pp_run_calc[:, d:], u_cf[:, d:]).sqrt().item()
        print(f"[Policy RMSE (u)] ||u_learn - u_closed-form||_RMSE: {rmse_u:.6f}")
        print(f"[Policy RMSE (C)] ||c_learn - c_closed-form||_RMSE: {rmse_c:.6f}")
        print(f"[Policy RMSE-PP (u)] ||u_pp(run) - u_closed-form||_RMSE: {rmse_pp_u:.6f}")
        print(f"[Policy RMSE-PP (C)] ||c_pp(run) - c_closed-form||_RMSE: {rmse_pp_c:.6f}")
    else:
        rmse_u    = F.mse_loss(u_learn,       u_cf).sqrt().item()
        rmse_pp_u = F.mse_loss(u_pp_run_calc, u_cf).sqrt().item()
        print(f"[Policy RMSE (u)] ||u_learn - u_closed-form||_RMSE: {rmse_u:.6f}")
        print(f"[Policy RMSE-PP (u)] ||u_pp(run) - u_closed-form||_RMSE: {rmse_pp_u:.6f}")

    # ---------- 샘플 프리뷰 ----------
    PREVIEW_ROWS   = int(os.getenv("PGDPO_PREVIEW_ROWS", "3"))
    PREVIEW_COORDS = int(os.getenv("PGDPO_PREVIEW_COORDS", "3"))

    print("\n--- Sample Previews ---")
    rows = min(PREVIEW_ROWS, N_eval)
    for i in range(rows):
        bits = []
        if "X" in eval_states:
            try: bits.append(f"X={float(eval_states['X'][i]):.3f}")
            except: pass
        if "TmT" in eval_states:
            try: bits.append(f"TmT={float(eval_states['TmT'][i]):.3f}")
            except: pass
        if "Y" in eval_states:
            try:
                Y = eval_states["Y"]
                if Y.dim() >= 2 and Y.shape[1] > 0:
                    bits.append(f"Y[0]={float(Y[i,0]):.3f}")
            except: pass
        prefix = "  (" + ", ".join(bits) + ") -> "

        if has_consumption and d_asset >= 1 and d_out >= d_asset + 1:
            d = d_asset
            k = min(PREVIEW_COORDS, d)
            line = (
                f"{_vec_head('u_learn', u_learn, i, k)}, "
                f"c_learn={float(u_learn[i, d]):.4f}, "
                f"{_vec_head('u_pp(run)', u_pp_run_calc, i, k)}, "
                f"c_pp={float(u_pp_run_calc[i, d]):.4f}, "
                f"{_vec_head('u_cf', u_cf, i, k)}, "
                f"c_cf={float(u_cf[i, d]):.4f}"
            )
        else:
            k = min(PREVIEW_COORDS, d_out)
            line = (
                f"{_vec_head('u_learn', u_learn, i, k)}, "
                f"{_vec_head('u_pp(run)', u_pp_run_calc, i, k)}, "
                f"{_vec_head('u_cf', u_cf, i, k)}"
            )
        print(prefix + "(" + line + ")")




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