# core/pgdpo_with_projection.py
# -*- coding: utf-8 -*-
"""
P-PGDPO (Projected PG-DPO) core — DIRECT path only
- Mandatory user projector (project_pmp) import; if missing, raise immediately.
- Flexible DIRECT costate estimator with injected simulator (user_pgdpo_base.simulate).
- Direct evaluator with verbose prints (+ sample preview), and a simple standalone test harness.
"""

from __future__ import annotations
import os
import inspect
from typing import Callable, Dict, Optional, Iterable

import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Mandatory imports from user space
# ---------------------------------------------------------------------
try:
    # device, eval sampling, CRN seed, initial state sampler
    from user_pgdpo_base import (
        device,
        N_eval_states,
        CRN_SEED_EU,
        sample_initial_states,   # (B, rng=None) -> (states_dict, dt_vec)
    )
except Exception as e:
    raise RuntimeError(
        "[core] Could not import required symbols from user_pgdpo_base "
        "(device, N_eval_states, CRN_SEED_EU, sample_initial_states)."
    ) from e

# 🔴 유저 projector는 필수. 없으면 즉시 오류로 중단.
try:
    # Must be defined in tests/<model>/user_pgdpo_with_projection.py
    from user_pgdpo_with_projection import project_pmp  # (costates:dict, states:dict) -> torch.Tensor
except Exception as e:
    raise RuntimeError(
        "[core] Required user projector 'project_pmp' not found in user_pgdpo_with_projection.py.\n"
        "Add:\n"
        "    def project_pmp(costates: dict, states: dict) -> torch.Tensor:\n"
        "        ...\n"
    ) from e

# ---------------------------------------------------------------------
# Defaults (env-overridable)
# ---------------------------------------------------------------------
REPEATS  = int(os.getenv("PGDPO_PP_REPEATS", 2560))
SUBBATCH = int(os.getenv("PGDPO_PP_SUBBATCH", 256))

# Verbose controls
VERBOSE            = os.getenv("PGDPO_VERBOSE", "1") == "1"   # 풍부 출력 on/off
SAMPLE_PREVIEW_N   = int(os.getenv("PGDPO_SAMPLE_PREVIEW_N", 3))  # 미리보기 라인 수
PRINT_COORD        = int(os.getenv("PGDPO_PRINT_COORD", 0))       # 다차원 u일 때 인덱스

# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------
def make_generator(seed: Optional[int] = None) -> torch.Generator:
    g = torch.Generator(device=device if device.type != "cpu" else "cpu")
    if seed is not None:
        g.manual_seed(int(seed))
    return g

def _divisors_desc(n: int):
    return sorted([d for d in range(1, n + 1) if n % d == 0], reverse=True)

def _as_numpy_1d(t: torch.Tensor, coord: int = 0):
    x = t.detach().cpu().numpy()
    if x.ndim == 2 and x.shape[1] > 1:
        return x[:, coord]
    return x.reshape(-1)

def _save_overlaid_deltas_safe(u_learn, u_pp, u_cf, outdir: Optional[str], coord: int, fname: str):
    if not outdir:
        return
    try:
        from viz import save_overlaid_delta_hists
        save_overlaid_delta_hists(
            u_learn=u_learn, u_pp=u_pp, u_cf=u_cf,
            outdir=outdir, coord=coord, fname=fname
        )
    except Exception:
        pass

# ---------------------------------------------------------------------
# Generic DIRECT costate estimator (reward-based) with injected simulator
#   simulate_fn(policy, B, train, initial_states_dict=..., random_draws=None,
#               m_steps=None, rng=None, **kwargs) -> U (B,1) or (B,)
# ---------------------------------------------------------------------
SimulateFn = Callable[..., torch.Tensor]

def estimate_costates(
    simulate_fn: SimulateFn,
    policy_net: nn.Module,
    initial_states: Dict[str, torch.Tensor],
    *,
    repeats: int,
    sub_batch: int,
    seed_eval: Optional[int] = None,
    needs: Iterable[str] = ("JX", "JXX", "JXY"),
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Estimate reward-based costates for given initial states (DIRECT simulator).
      - JX  = ∂U/∂X
      - JY  = ∂U/∂Y   (if Y present & requested)
      - JXX = ∂²U/∂X²
      - JXY = ∂²U/∂X∂Y
      - JYY = ∂²U/∂Y²   (computed only if requested)
    Only the derivatives requested in `needs` are computed.
    """
    assert repeats > 0 and sub_batch > 0
    needs = tuple(needs)
    B = next(iter(initial_states.values())).size(0)

    # Prepare leaf states
    leaf: Dict[str, torch.Tensor] = {}
    for k in ("X", "Y"):
        if k in initial_states:
            leaf[k] = initial_states[k].detach().clone().requires_grad_(True)
    if not leaf:
        raise RuntimeError("[core] estimate_costates: no differentiable states found (need X and/or Y).")

    TmT_val = initial_states.get("TmT")
    if TmT_val is not None:
        TmT_val = TmT_val.detach().clone()  # not differentiated

    # Freeze policy params
    params = list(policy_net.parameters())
    req_bak = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)

    # 누적 버퍼 (JYY 포함)
    sums: Dict[str, Optional[torch.Tensor]] = {"JX": None, "JY": None, "JXX": None, "JXY": None, "JYY": None}

    def _acc(name: str, val: Optional[torch.Tensor], weight: int):
        if val is None:
            return
        if sums[name] is None:
            sums[name] = val.detach() * weight
        else:
            sums[name] = sums[name] + val.detach() * weight

    # simulate_fn 시그니처 검사 (rng 지원 여부)
    sim_sig = inspect.signature(simulate_fn)
    accepts_rng = "rng" in sim_sig.parameters

    done = 0
    try:
        while done < repeats:
            rpts = min(sub_batch, repeats - done)

            # replicate states rpts times → total B*rpts
            batch_states: Dict[str, torch.Tensor] = {k: v.repeat(rpts, 1) for k, v in leaf.items()}
            if TmT_val is not None:
                batch_states["TmT"] = TmT_val.repeat(rpts, 1)

            rng_arg = make_generator((seed_eval or 0) + done) if accepts_rng else None

            with torch.enable_grad():
                # simulate
                args = [policy_net, B * rpts]
                kwargs = {"train": True, "initial_states_dict": batch_states, "m_steps": None}
                if accepts_rng:
                    kwargs["rng"] = rng_arg
                U = simulate_fn(*args, **kwargs)  # (B*rpts, 1) or (B*rpts,)
                if not U.requires_grad:
                    raise RuntimeError(
                        "simulate() returned a tensor without grad.\n"
                        "Check tests/<model>/user_pgdpo_base.py::simulate:\n"
                        "  - MUST consume the provided initial_states_dict['X'] (and 'Y') directly (no resampling)\n"
                        "  - MUST NOT detach() X/Y or wrap the sim in @torch.no_grad()\n"
                        "  - MUST return a torch tensor that depends on X/Y (e.g., terminal utility)"
                    )
                U = U.view(B * rpts, -1).mean(dim=1).view(rpts, B).mean(dim=0)  # (B,)

                # 1st derivatives
                need_2nd = any(k in needs for k in ("JXX", "JXY", "JYY"))
                targets = []
                if "X" in leaf: targets.append(leaf["X"])
                if "Y" in leaf: targets.append(leaf["Y"])
                grads = torch.autograd.grad(
                    U.sum(),
                    tuple(targets),
                    create_graph=need_2nd,
                    retain_graph=need_2nd,
                    allow_unused=True
                )
                grad_map: Dict[str, Optional[torch.Tensor]] = {"JX": None, "JY": None}
                idx = 0
                if "X" in leaf:
                    grad_map["JX"] = grads[idx]; idx += 1
                if "Y" in leaf:
                    grad_map["JY"] = grads[idx] if len(grads) > idx else None

                # 2nd derivatives (if requested)
                JXX_batch = None
                JXY_batch = None
                JYY_batch = None

                if need_2nd:
                    if ("JXX" in needs) and ("X" in leaf) and (grad_map["JX"] is not None):
                        JXX_batch, = torch.autograd.grad(
                            grad_map["JX"].sum(), leaf["X"],
                            retain_graph = (("JXY" in needs) or ("JYY" in needs)) and ("Y" in leaf)
                        )
                    if ("JXY" in needs) and ("Y" in leaf) and (grad_map["JX"] is not None):
                        JXY_batch, = torch.autograd.grad(
                            grad_map["JX"].sum(), leaf["Y"],
                            retain_graph = ("JYY" in needs) and ("Y" in leaf) and (grad_map["JY"] is not None)
                        )
                    if ("JYY" in needs) and ("Y" in leaf) and (grad_map["JY"] is not None):
                        JYY_batch, = torch.autograd.grad(grad_map["JY"].sum(), leaf["Y"])

                # accumulate
                if "JX" in needs and grad_map["JX"] is not None:
                    _acc("JX", grad_map["JX"], rpts)
                if "JY" in needs and grad_map["JY"] is not None:
                    _acc("JY", grad_map["JY"], rpts)
                if "JXX" in needs and JXX_batch is not None:
                    _acc("JXX", JXX_batch, rpts)
                if "JXY" in needs and JXY_batch is not None:
                    _acc("JXY", JXY_batch, rpts)
                if "JYY" in needs and JYY_batch is not None:
                    _acc("JYY", JYY_batch, rpts)

            done += rpts

        invN = 1.0 / repeats
        return {k: (v * invN if v is not None else None) for k, v in sums.items()}
    finally:
        for p, r in zip(params, req_bak):
            p.requires_grad_(r)

# ---------------------------------------------------------------------
# Direct projector helper: compute u_pp via user projector on given states
# ---------------------------------------------------------------------
def ppgdpo_u_direct(
    pol_s1: nn.Module,
    states: Dict[str, torch.Tensor],
    *,
    repeats: int = REPEATS,
    sub_batch: int = SUBBATCH,
    seed_eval: Optional[int] = None,
    needs: Iterable[str] = ("JX", "JXX", "JXY"),
) -> torch.Tensor:
    """
    Compute projected control u_pp on given states using DIRECT simulate (user_pgdpo_base.simulate).
    """
    from user_pgdpo_base import simulate as simulate_base
    costates = estimate_costates(
        simulate_base, pol_s1, states,
        repeats=repeats, sub_batch=sub_batch, seed_eval=seed_eval, needs=needs
    )
    u = project_pmp(costates, states)
    return u

# ---------------------------------------------------------------------
# Public evaluator (DIRECT) used by run.py or standalone main
# ---------------------------------------------------------------------
def print_policy_rmse_and_samples_direct(
    pol_s1: nn.Module,
    pol_cf: Optional[nn.Module],
    *,
    repeats: int = REPEATS,
    sub_batch: int = SUBBATCH,
    seed_eval: Optional[int] = None,
    outdir: Optional[str] = None,
    tile: Optional[int] = None,
    enable_pp: bool = True,
    prefix: str = "direct",
    needs: Iterable[str] = ("JX", "JXX", "JXY"),
) -> None:
    """
    DIRECT evaluator (simple/base simulator):
      - simulate_fn = user_pgdpo_base.simulate
      - compares: learned vs (optional) closed-form vs (optional) projected
    """
    # prepare eval points
    gen = make_generator(seed_eval or CRN_SEED_EU)
    states_dict, _ = sample_initial_states(N_eval_states, rng=gen)

    # learned policy output
    u_learn = pol_s1(**states_dict)

    # projected policy (optional)
    u_pp = None
    if enable_pp:
        B = next(iter(states_dict.values())).size(0)
        divisors = _divisors_desc(B)
        start_idx = next((i for i, d in enumerate(divisors) if d <= (tile or B)), 0)
        u_pp = torch.empty_like(u_learn)
        for idx in range(start_idx, len(divisors)):
            cur_tile = divisors[idx]
            try:
                for s in range(0, B, cur_tile):
                    e = min(B, s + cur_tile)
                    tile_states = {k: v[s:e] for k, v in states_dict.items()}
                    u_pp[s:e] = ppgdpo_u_direct(
                        pol_s1, tile_states,
                        repeats=repeats, sub_batch=sub_batch, seed_eval=seed_eval, needs=tuple(needs)
                    )
                break
            except RuntimeError as oom:
                if "out of memory" in str(oom).lower():
                    continue
                raise

    # closed-form (optional)
    u_cf = pol_cf(**states_dict) if pol_cf is not None else None

    # RMSE (learn vs CF)
    if u_cf is not None:
        rmse = torch.sqrt(((u_learn - u_cf) ** 2).mean()).item()
        print(f"[Policy RMSE] ||u_learn - u_closed-form||_RMSE: {rmse:.6f}")

    # RMSE (PP-direct vs CF) — 풍부 출력 복원
    if enable_pp and (u_pp is not None) and (u_cf is not None):
        rmse_pp = torch.sqrt(((u_pp - u_cf) ** 2).mean()).item()
        print(f"[Policy RMSE-PP(direct)] ||u_pp(direct) - u_closed-form||_RMSE: {rmse_pp:.6f}")

    # sample preview (first N)
    if VERBOSE and enable_pp:
        n = min(SAMPLE_PREVIEW_N, u_learn.size(0))
        X_vals   = _as_numpy_1d(states_dict.get("X"), 0)[:n] if "X" in states_dict else [None]*n
        TmT_vals = _as_numpy_1d(states_dict.get("TmT"), 0)[:n] if "TmT" in states_dict else [None]*n
        uL = _as_numpy_1d(u_learn, PRINT_COORD)[:n]
        uP = _as_numpy_1d(u_pp,    PRINT_COORD)[:n] if u_pp is not None else [None]*n
        uC = _as_numpy_1d(u_cf,    PRINT_COORD)[:n] if u_cf is not None else [None]*n
        for i in range(n):
            if u_pp is not None and u_cf is not None:
                print(f"  (X={X_vals[i]:.3f}, TmT={TmT_vals[i]:.3f}) -> "
                      f"(u_learn[{i}]={uL[i]:.4f}, u_pp(dir)[{i}]={uP[i]:.4f}, u_cf[{i}]={uC[i]:.4f})")
            elif u_pp is not None:
                print(f"  (X={X_vals[i]:.3f}, TmT={TmT_vals[i]:.3f}) -> "
                      f"(u_learn[{i}]={uL[i]:.4f}, u_pp(dir)[{i}]={uP[i]:.4f})")
            elif u_cf is not None:
                print(f"  (X={X_vals[i]:.3f}, TmT={TmT_vals[i]:.3f}) -> "
                      f"(u_learn[{i}]={uL[i]:.4f}, u_cf[{i}]={uC[i]:.4f})")

    # overlay deltas (optional viz)
    _save_overlaid_deltas_safe(u_learn, u_pp, u_cf, outdir, coord=PRINT_COORD, fname=f"{prefix}_delta_overlaid_hist.png")

# -----------------------------------------------------------------------------
# 독립 실행 테스트
# -----------------------------------------------------------------------------
def main():
    # base 학습 → direct 평가 (간이 테스트)
    from pgdpo_base import run_common, train_stage1_base  # lazy import to avoid cycles
    run_common(
        train_fn=train_stage1_base,
        rmse_fn=print_policy_rmse_and_samples_direct,
        train_kwargs={},  # 필요 시 epochs/lr override 가능
        rmse_kwargs={"repeats": REPEATS, "sub_batch": SUBBATCH, "seed_eval": CRN_SEED_EU},
    )

if __name__ == "__main__":
    main()

__all__ = [
    "REPEATS",
    "SUBBATCH",
    "estimate_costates",
    "project_pmp",
    "ppgdpo_u_direct",
    "print_policy_rmse_and_samples_direct",
    "VERBOSE", "SAMPLE_PREVIEW_N"
]