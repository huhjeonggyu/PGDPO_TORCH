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
    from user_pgdpo_with_projection import project_pmp, PP_NEEDS
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
def print_policy_rmse_and_samples_direct(policy, cf_policy, **kwargs):
    """
    DIRECT path evaluator:
      - Samples CRN eval states via user_pgdpo_base.sample_initial_states (N_eval_states, CRN_SEED_EU)
      - Computes u_learn, u_pp(direct), u_cf
      - Prints RMSEs + 3 sample triplets
      - Saves dim-0 scatter PNGs if out_dir is provided
    Optional kwargs:
      - out_dir (str), device (torch.device), tile (int),
        repeats, sub_batch, seed_eval, needs  (overrides for ppgdpo_u_direct)
    """
    import os, math, numpy as np, torch
    # headless-safe plotting (optional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    # ---------- helpers ----------
    def _as_device(x, dev, dtype=None):
        return x.to(device=dev, dtype=(dtype or x.dtype)) if torch.is_tensor(x) else x

    def _call_cf(cf, states_slice):
        with torch.no_grad():
            try:
                return cf(**states_slice)                   # (**states)
            except TypeError:
                try:
                    return cf(states_slice)                 # (states)
                except TypeError:
                    return cf(states_slice["X"],            # (X, TmT)
                              states_slice.get("TmT", None))

    def _rmse(a, b):
        return float(torch.sqrt(torch.mean((a - b) ** 2)).item())

    def _save_scatter_dim0(y_pred, y_true, path, title):
        if plt is None or path is None:
            return float("nan")
        yp = y_pred.detach().cpu().numpy() if hasattr(y_pred, "detach") else np.asarray(y_pred)
        yt = y_true.detach().cpu().numpy() if hasattr(y_true, "detach") else np.asarray(y_true)
        x = np.asarray(yt[:, 0], dtype=np.float64)
        y = np.asarray(yp[:, 0], dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size == 0:
            return float("nan")
        sst = np.sum((x - x.mean()) ** 2)
        ssr = np.sum((y - x) ** 2)
        r2 = float("nan") if sst <= 1e-12 else 1.0 - ssr / sst

        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure()
        plt.scatter(x, y, s=6, alpha=0.6)
        lo = float(np.min([x.min(), y.min()])); hi = float(np.max([x.max(), y.max()]))
        plt.plot([lo, hi], [lo, hi], lw=1)
        plt.xlabel("u_cf (dim 0)"); plt.ylabel(title)
        plt.title(f"{title} vs CF — R^2={r2:.4f}" if math.isfinite(r2) else f"{title} vs CF — R^2=nan")
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return r2

    # ---------- config ----------
    out_dir   = kwargs.get("out_dir", os.getenv("PGDPO_OUT_DIR"))
    dev       = kwargs.get("device", device)  # `device` is imported at module top from user_pgdpo_base
    tile      = kwargs.get("tile", 4096)
    repeats   = int(kwargs.get("repeats", REPEATS))
    sub_batch = int(kwargs.get("sub_batch", SUBBATCH))
    seed_eval = kwargs.get("seed_eval", CRN_SEED_EU)
    needs     = kwargs.get("needs", PP_NEEDS)

    B_eval = int(N_eval_states)

    # ---------- sample eval states (CRN) ----------
    rng = torch.Generator(device=dev.type)  # 'cuda' 또는 'cpu'
    rng.manual_seed(int(seed_eval))
    states_all, _ = sample_initial_states(B_eval, rng=rng)
    for k in list(states_all.keys()):
        states_all[k] = _as_device(states_all[k], dev)

    # ---------- u_learn ----------
    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        u_learn = policy(**states_all)  # (B,d)

    # ---------- u_pp via direct projector ----------
    u_pp = torch.empty_like(u_learn)
    s = 0
    while s < B_eval:
        e = min(B_eval, s + tile)
        states_slice = {k: v[s:e] for k, v in states_all.items()}
        try:
            u_pp[s:e] = ppgdpo_u_direct(
                pol_s1=policy, states=states_slice,
                repeats=repeats, sub_batch=sub_batch, seed_eval=seed_eval, needs=needs
            )
            s = e
        except RuntimeError as ex:
            msg = str(ex).lower()
            if ("out of memory" in msg or "cuda" in msg) and tile > 256:
                tile //= 2
                torch.cuda.empty_cache()
                print(f"[Eval] OOM; reducing tile -> {tile}")
            else:
                raise

    # ---------- u_cf ----------
    if isinstance(cf_policy, (tuple, list)) and callable(cf_policy[0]):
        cf_policy = cf_policy[0]
    with torch.no_grad():
        u_cf = _call_cf(cf_policy, states_all)
        u_cf = _as_device(u_cf, dev, dtype=u_learn.dtype)

    # ---------- RMSE & samples ----------
    rmse_learn = _rmse(u_learn, u_cf)
    rmse_pp    = _rmse(u_pp,    u_cf)
    print(f"[Policy RMSE] ||u_learn - u_closed-form||_RMSE: {rmse_learn:.6f}")
    print(f"[Policy RMSE-PP(direct)] ||u_pp(direct) - u_closed-form||_RMSE: {rmse_pp:.6f}")

    k_show = min(3, B_eval)
    X  = states_all.get("X");  TmT = states_all.get("TmT")
    for i in range(k_show):
        if X is not None and TmT is not None:
            x0  = float(X[i, 0].item()) if X.ndim == 2 else float(X[i].item())
            tmt = float(TmT[i, 0].item()) if TmT.ndim == 2 else float(TmT[i].item())
            print(f"  (X={x0:.3f}, TmT={tmt:.3f}) -> "
                  f"(u_learn[{i}]={float(u_learn[i,0].item()):.4f}, "
                  f"u_pp(dir)[{i}]={float(u_pp[i,0].item()):.4f}, "
                  f"u_cf[{i}]={float(u_cf[i,0].item()):.4f})")
        else:
            print(f"  [{i}] -> "
                  f"(u_learn={float(u_learn[i,0].item()):.4f}, "
                  f"u_pp(dir)={float(u_pp[i,0].item()):.4f}, "
                  f"u_cf={float(u_cf[i,0].item()):.4f})")

    # ---------- force-save scatters (if out_dir provided) ----------
    if out_dir:
        p0 = os.path.join(out_dir, "scatter_projection_learn_vs_cf_dim0.png")
        p1 = os.path.join(out_dir, "scatter_projection_pp_vs_cf_dim0.png")
        r2_learn = _save_scatter_dim0(u_learn, u_cf, p0, "u_learn")
        r2_pp    = _save_scatter_dim0(u_pp,    u_cf, p1, "u_pp(direct)")
        if math.isfinite(r2_learn) or math.isfinite(r2_pp):
            print(f"[Scatter] saved:\n  {p0} (R^2={r2_learn:.4f})\n  {p1} (R^2={r2_pp:.4f})")

    if was_training:
        policy.train()


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
        rmse_kwargs={"repeats": REPEATS, "sub_batch": SUBBATCH, "seed_eval": CRN_SEED_EU, "needs" : PP_NEEDS},
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
    "VERBOSE", "SAMPLE_PREVIEW_N", "PP_NEEDS"
]