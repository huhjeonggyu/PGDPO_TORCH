# 파일: core/pgdpo_with_projection.py
# 모델: 소비 모델의 PMP 출력을 올바르게 처리하도록 수정된 버전

from __future__ import annotations
import os
import inspect
from typing import Callable, Dict, Optional, Iterable

import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# (이하 import 및 유틸 함수는 변경 없음)
# ...
try:
    from user_pgdpo_base import (
        d, device, N_eval_states, CRN_SEED_EU, sample_initial_states,
    )
except Exception as e:
    raise RuntimeError(
        f"[core] Could not import required symbols from user_pgdpo_base: {e}"
    ) from e

try:
    from user_pgdpo_with_projection import project_pmp, PP_NEEDS
except Exception as e:
    raise RuntimeError(
        f"[core] Required user projector 'project_pmp' not found in user_pgdpo_with_projection.py."
    ) from e

from viz import (
    save_combined_scatter, save_overlaid_delta_hists, append_metrics_csv
)

PREVIEW_COORDS = int(os.getenv("PGDPO_PREVIEW_COORDS", 3))

def _fmt_coords(label: str, mat: torch.Tensor, i: int, k: int) -> str:
    """행 i에서 앞 k개 좌표를 'label[j]=v' 포맷으로 이어붙임"""
    n = mat.size(1)
    K = min(k, n)
    parts = [f"{label}[{j}]={mat[i,j].item():.4f}" for j in range(K)]
    return (", ".join(parts)) + (", ..." if n > K else "")

REPEATS  = int(os.getenv("PGDPO_PP_REPEATS", 2560*d))
SUBBATCH = int(os.getenv("PGDPO_PP_SUBBATCH", 2560))
VERBOSE            = os.getenv("PGDPO_VERBOSE", "1") == "1"
SAMPLE_PREVIEW_N   = int(os.getenv("PGDPO_SAMPLE_PREVIEW_N", 3))

def make_generator(seed: Optional[int] = None) -> torch.Generator:
    g = torch.Generator(device=device if device.type != "cpu" else "cpu")
    if seed is not None:
        g.manual_seed(int(seed))
    return g

def _divisors_desc(n: int):
    return sorted([d for d in range(1, n + 1) if n % d == 0], reverse=True)

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
    assert repeats > 0 and sub_batch > 0
    needs = tuple(needs)
    
    valid_states = [v for v in initial_states.values() if v is not None]
    if not valid_states: raise RuntimeError("[core] estimate_costates: all states are None.")
    B = valid_states[0].size(0)

    leaf: Dict[str, torch.Tensor] = {}
    for k in ("X", "Y"):
        if k in initial_states and initial_states[k] is not None:
            leaf[k] = initial_states[k].detach().clone().requires_grad_(True)
            
    if not leaf:
        raise RuntimeError("[core] estimate_costates: no differentiable states found (need X and/or Y).")

    TmT_val = initial_states.get("TmT")
    if TmT_val is not None:
        TmT_val = TmT_val.detach().clone()

    params = list(policy_net.parameters())
    req_bak = [p.requires_grad for p in params]
    for p in params: p.requires_grad_(False)
    
    sums: Dict[str, Optional[torch.Tensor]] = {k: None for k in ("JX", "JY", "JXX", "JXY", "JYY")}
    def _acc(name: str, val: Optional[torch.Tensor], weight: int):
        if val is None: return
        if sums[name] is None: sums[name] = val.detach() * weight
        else: sums[name] += val.detach() * weight

    sim_sig = inspect.signature(simulate_fn)
    accepts_rng = "rng" in sim_sig.parameters
    done = 0
    try:
        while done < repeats:
            rpts = min(sub_batch, repeats - done)
            batch_states: Dict[str, torch.Tensor] = {k: v.repeat(rpts, *([1]*(v.dim()-1))) for k, v in leaf.items()}
            if TmT_val is not None: batch_states["TmT"] = TmT_val.repeat(rpts, 1)

            rng_arg = make_generator((seed_eval or 0) + done) if accepts_rng else None
            with torch.enable_grad():
                args = [policy_net, B * rpts]
                kwargs = {"train": True, "initial_states_dict": batch_states, "m_steps": None}
                if accepts_rng: kwargs["rng"] = rng_arg
                U = simulate_fn(*args, **kwargs)

                if not U.requires_grad: raise RuntimeError("simulate() returned a tensor without grad.")
                U = U.view(B * rpts, -1).mean(dim=1).view(rpts, B).mean(dim=0)
                
                need_2nd = any(k in needs for k in ("JXX", "JXY", "JYY"))
                targets = [leaf[k] for k in ("X", "Y") if k in leaf]
                grads = torch.autograd.grad(U.sum(), tuple(targets), create_graph=need_2nd, retain_graph=need_2nd, allow_unused=True)
                grad_map: Dict[str, Optional[torch.Tensor]] = {}
                
                grad_idx = 0
                if "X" in leaf: grad_map["JX"] = grads[grad_idx]; grad_idx += 1
                if "Y" in leaf: grad_map["JY"] = grads[grad_idx] if len(grads) > grad_idx else None

                if need_2nd:
                    if ("JXX" in needs) and ("X" in leaf) and (grad_map.get("JX") is not None):
                        JXX_batch, = torch.autograd.grad(grad_map["JX"].sum(), leaf["X"], retain_graph=True, allow_unused=True)
                        _acc("JXX", JXX_batch, rpts)
                    if ("JXY" in needs) and ("Y" in leaf) and (grad_map.get("JX") is not None):
                        JXY_batch, = torch.autograd.grad(grad_map["JX"].sum(), leaf["Y"], retain_graph=True, allow_unused=True)
                        _acc("JXY", JXY_batch, rpts)
                    if ("JYY" in needs) and ("Y" in leaf) and (grad_map.get("JY") is not None):
                        JYY_batch, = torch.autograd.grad(grad_map["JY"].sum(), leaf["Y"], allow_unused=True)
                        _acc("JYY", JYY_batch, rpts)

                if "JX" in needs: _acc("JX", grad_map.get("JX"), rpts)
                if "JY" in needs: _acc("JY", grad_map.get("JY"), rpts)
            done += rpts
            
        return {k: (v * (1.0 / repeats) if v is not None else None) for k, v in sums.items()}
    finally:
        for p, r in zip(params, req_bak): p.requires_grad_(r)

def ppgdpo_u_direct(
    pol_s1: nn.Module,
    states: Dict[str, torch.Tensor],
    *,
    repeats: int = REPEATS,
    sub_batch: int = SUBBATCH,
    seed_eval: Optional[int] = None,
    needs: Iterable[str] = ("JX", "JXX", "JXY"),
) -> torch.Tensor:
    from user_pgdpo_base import simulate as simulate_base
    with torch.enable_grad():
        costates = estimate_costates(
            simulate_base, pol_s1, states,
            repeats=repeats, sub_batch=sub_batch, seed_eval=seed_eval, needs=needs
        )
        u = project_pmp(costates, states)
        return u.detach()

# ---------------------------------------------------------------------
# (✨✨✨ 핵심 수정 부분 시작 ✨✨✨)
# ---------------------------------------------------------------------
@torch.no_grad()
def print_policy_rmse_and_samples_direct(
    pol_s1: nn.Module,
    pol_cf: nn.Module | None,
    *,
    repeats: int,
    sub_batch: int,
    seed_eval: int | None = None,
    outdir: str | None = None,
    tile: int | None = None,
    needs: Iterable[str] = ("JX", "JXX", "JXY"),
) -> None:
    gen = make_generator(seed_eval or CRN_SEED_EU)
    states_dict, _ = sample_initial_states(N_eval_states, rng=gen)
    
    # 1. 학습된 정책과 벤치마크 정책의 출력을 u와 c로 분리 (기존과 동일)
    u_learn_full = pol_s1(**states_dict)
    is_consumption_model = u_learn_full.size(1) > d
    u_learn, c_learn = (u_learn_full[:, :d], u_learn_full[:, d:]) if is_consumption_model else (u_learn_full, None)
    
    u_cf_full = pol_cf(**states_dict) if pol_cf is not None else None
    u_cf, c_cf = (None, None)
    if u_cf_full is not None:
        u_cf, c_cf = (u_cf_full[:, :d], u_cf_full[:, d:]) if is_consumption_model else (u_cf_full, None)

    # 2. P-PGDPO 정책 계산
    u_pp, c_pp = None, None
    try:
        valid_states = [v for v in states_dict.values() if v is not None]
        if not valid_states: raise ValueError("Cannot determine batch size because all states are None.")
        B = valid_states[0].size(0)

        divisors = _divisors_desc(B)
        start_idx = next((i for i, d_ in enumerate(divisors) if d_ <= (tile or B)), 0)
        
        # 3. ✨ 수정: PMP의 전체 출력(u와 C)을 받을 텐서 준비
        #    is_consumption_model에 따라 동적으로 크기 결정 (d+1 또는 d)
        pp_full_output_size = d + 1 if is_consumption_model else d
        u_pp_full_calc = torch.empty(B, pp_full_output_size, device=u_learn.device, dtype=u_learn.dtype)
        
        for idx in range(start_idx, len(divisors)):
            cur_tile = divisors[idx]
            try:
                for s in range(0, B, cur_tile):
                    e = min(B, s + cur_tile)
                    tile_states = {k: v[s:e] for k, v in states_dict.items() if v is not None}
                    # ✨ 수정: 전체 PMP 출력을 u_pp_full_calc에 저장
                    u_pp_full_calc[s:e] = ppgdpo_u_direct(pol_s1, tile_states, repeats=repeats, sub_batch=sub_batch, seed_eval=(seed_eval if seed_eval is not None else 0), needs=tuple(needs))
                
                # 4. ✨ 수정: 계산된 전체 PMP 출력을 u_pp와 c_pp로 분리
                if is_consumption_model:
                    u_pp = u_pp_full_calc[:, :d]
                    c_pp = u_pp_full_calc[:, d:]
                else:
                    u_pp = u_pp_full_calc

                break 
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    if idx + 1 < len(divisors): print(f"[Eval] OOM; reducing tile -> {divisors[idx+1]}"); continue
                    else: print("\n[Eval] ERROR: Unrecoverable OOM error."); break
                else: raise
    except Exception as e:
        print(f"\n[Eval] ERROR: An unexpected error occurred during PMP projection: {e}")
        u_pp, c_pp = None, None

    # 5. RMSE 및 샘플 출력 (이제 u_pp와 c_pp가 모두 준비되어 있으므로, 기존 로직 그대로 사용 가능)
    prefix = "direct"
    if u_cf is not None:
        rmse_learn_u = torch.sqrt(((u_learn - u_cf) ** 2).mean()).item()
        print(f"\n[Policy RMSE (u)] ||u_learn - u_closed-form||_RMSE: {rmse_learn_u:.6f}")
        metrics = {f"rmse_learn_cf_u_{prefix}": rmse_learn_u}
        if u_pp is not None:
            rmse_pp_u = torch.sqrt(((u_pp - u_cf) ** 2).mean()).item()
            print(f"[Policy RMSE-PP (u)] ||u_pp({prefix}) - u_closed-form||_RMSE: {rmse_pp_u:.6f}")
            metrics[f"rmse_pp_cf_u_{prefix}"] = rmse_pp_u

        if is_consumption_model and c_cf is not None:
            rmse_learn_c = torch.sqrt(((c_learn - c_cf) ** 2).mean()).item()
            print(f"[Policy RMSE (C)] ||c_learn - c_closed-form||_RMSE: {rmse_learn_c:.6f}")
            metrics[f"rmse_learn_cf_c_{prefix}"] = rmse_learn_c
            if c_pp is not None:
                rmse_pp_c = torch.sqrt(((c_pp - c_cf) ** 2).mean()).item()
                print(f"[Policy RMSE-PP (C)] ||c_pp({prefix}) - c_closed-form||_RMSE: {rmse_pp_c:.6f}")
                metrics[f"rmse_pp_cf_c_{prefix}"] = rmse_pp_c

        if outdir is not None:
            append_metrics_csv(metrics, outdir)
            save_overlaid_delta_hists(u_learn=u_learn, u_pp=u_pp, u_cf=u_cf, outdir=outdir, coord=0, fname=f"delta_{prefix}_overlaid_hist.png")
            save_combined_scatter(u_ref=u_cf, u_learn=u_learn, u_pp=u_pp, outdir=outdir, coord=0, fname=f"scatter_{prefix}_comparison.png")
    else:
        print("[INFO] No closed-form policy provided for comparison.")

    if VERBOSE:
        n = min(SAMPLE_PREVIEW_N, u_learn.size(0))
        print("\n--- Sample Previews ---")
        for i in range(n):
            parts, vec = [], False
            for k_, v in states_dict.items():
                if v is None: continue
                ts = v[i]
                if ts.numel() > 1: parts.append(f"{k_}[0]={ts[0].item():.3f}"); vec = True
                else: parts.append(f"{k_}={ts.item():.3f}")
            if vec: parts.append("...")
            sstr = ", ".join(parts)
    
            msg_parts = [f"  ({sstr}) -> ("]
            msg_parts.append(_fmt_coords('u_learn', u_learn, i, PREVIEW_COORDS))
            if c_learn is not None: msg_parts.append(f", c_learn={c_learn[i].item():.4f}")
            if u_pp is not None:
                msg_parts.append(", " + _fmt_coords(f'u_pp({prefix})', u_pp, i, PREVIEW_COORDS))
                if c_pp is not None: msg_parts.append(f", c_pp={c_pp[i].item():.4f}")
            if u_cf is not None:
                msg_parts.append(", " + _fmt_coords('u_cf', u_cf, i, PREVIEW_COORDS))
                if c_cf is not None: msg_parts.append(f", c_cf={c_cf[i].item():.4f}")
            msg_parts.append(")")
            print("".join(msg_parts))
# ---------------------------------------------------------------------
# (✨✨✨ 핵심 수정 부분 끝 ✨✨✨)
# ---------------------------------------------------------------------

def main():
    from pgdpo_base import run_common, train_stage1_base
    run_common(
        train_fn=train_stage1_base,
        rmse_fn=print_policy_rmse_and_samples_direct,
        train_kwargs={},
        rmse_kwargs={"repeats": REPEATS, "sub_batch": SUBBATCH, "seed_eval": CRN_SEED_EU, "needs" : PP_NEEDS},
    )

if __name__ == "__main__":
    main()

__all__ = [
    "REPEATS", "SUBBATCH", "estimate_costates", "project_pmp",
    "ppgdpo_u_direct", "print_policy_rmse_and_samples_direct",
    "VERBOSE", "SAMPLE_PREVIEW_N", "PP_NEEDS"
]