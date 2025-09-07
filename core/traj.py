# core/traj.py — schema-driven, multi-domain trajectory visualization
# - Features PP_MODE ('runner' vs 'direct') to toggle between simulation types.
# - The 'direct' mode now correctly calls the same function used for RMSE calculation.
# - Generates separate, clear "learn vs cf" and "pp vs cf" plots.

from __future__ import annotations
import os, json, time, random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============ Import user problem and core modules ============
from pgdpo_base import (
    device, T, m, DIM_X, DIM_Y, DIM_U, CRN_SEED_EU, N_eval_states,
    sample_initial_states, simulate,
)

# --- MODIFIED: Import BOTH runner and direct PP functions ---
PP_MODE = os.getenv('PP_MODE','runner').lower()
#PP_MODE = os.getenv('PP_MODE','direct').lower()
_HAS_RUNNER, _HAS_DIRECT = False, False
try:
    from pgdpo_run import ppgdpo_u_run, REPEATS as REPEATS_RUN, SUBBATCH as SUBBATCH_RUN
    _HAS_RUNNER = True
except ImportError: pass
try:
    from pgdpo_with_projection import ppgdpo_u_direct, REPEATS as REPEATS_DIR, SUBBATCH as SUBBATCH_DIR
    _HAS_DIRECT = True
except ImportError: pass

# ============ Schema loading & defaults (Unchanged) ============
def _try_load_user_schema() -> Optional[Dict[str, Any]]:
    try: from user_pgdpo_base import get_traj_schema; return get_traj_schema()
    except (ImportError, AttributeError): pass
    try: from user_pgdpo_base import TRAJ_SCHEMA; return TRAJ_SCHEMA
    except (ImportError, AttributeError): pass
    return None
def _default_labels(prefix: str, dim: int) -> List[str]:
    if dim == 0: return []
    return [f"{prefix}_{i+1}" for i in range(dim)]
def _build_default_schema() -> Dict[str, Any]:
    views: List[Dict[str, Any]] = []
    if DIM_X and int(DIM_X) > 0:
        xdim = int(DIM_X)
        views.append({"name": "X_first", "block": "X", "mode": "indices", "indices": list(range(min(3, xdim))), "ylabel": "X components"})
        if xdim > 1: views.append({"name": "X_sum", "block": "X", "mode": "aggregate", "agg": "sum", "ylabel": "Σ X"})
    if DIM_Y and int(DIM_Y) > 0:
        ydim = int(DIM_Y)
        views.append({"name": "Y_first", "block": "Y", "mode": "indices", "indices": list(range(min(2, ydim))), "ylabel": "Y components"})
    if DIM_U and int(DIM_U) > 0:
        udim = int(DIM_U)
        views.append({"name": "U_first", "block": "U", "mode": "indices", "indices": list(range(min(2, udim))), "ylabel": "U components"})
        if udim > 1: views.append({"name": "U_l2", "block": "U", "mode": "aggregate", "agg": "l2norm", "ylabel": "‖u‖₂"})
    return {"roles": {"X": {"dim": int(DIM_X or 0), "labels": _default_labels("X", int(DIM_X or 0))},"Y": {"dim": int(DIM_Y or 0), "labels": _default_labels("Y", int(DIM_Y or 0))},"U": {"dim": int(DIM_U or 0), "labels": _default_labels("u", int(DIM_U or 0))},}, "views": views, "sampling": {"Bmax": 5, "strategy": "head", "seed": 777},}
SCHEMA: Dict[str, Any] = _try_load_user_schema() or _build_default_schema()

# ============ Utilities & Policy Wrappers ============
# ---- optional VPP reference hooks ----
try:
    from user_pgdpo_base import ref_signals_fn as _REF_FN   # expects: def ref_signals_fn(t_np) -> {"Nagg": array}
except Exception:
    _REF_FN = None
try:
    from user_pgdpo_base import R_INFO as _R_INFO           # e.g., {"alpha": 0.3} or {"R_diag":[...]} or {"R": ...}
except Exception:
    _R_INFO = {}

def _ensure_outdir(outdir: Optional[str]) -> str:
    if outdir is None: outdir = os.path.join("plots", "_traj", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True); return outdir
def _to_np(x) -> np.ndarray:
    if isinstance(x, np.ndarray): return x
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    raise TypeError(f"Cannot convert type {type(x)} to numpy array.")
def _linspace_time(n_steps: int) -> np.ndarray:
    if n_steps <= 1: return np.array([0.0])
    return np.linspace(0.0, float(T), num=n_steps)
def _u_rnorm(U: np.ndarray) -> np.ndarray:
    # U: (B, steps, d) numpy
    info = _R_INFO or {}
    if "R" in info:
        R = np.asarray(info["R"])
        UR = np.einsum("bsd,dd->bsd", U, R)
        return np.sqrt(np.einsum("bsd,bsd->bs", U, UR))
    if "R_diag" in info:
        r = np.asarray(info["R_diag"]).reshape(1,1,-1)
        return np.sqrt((U * r * U).sum(axis=-1))
    a = float(info.get("alpha", 1.0))   # default: R = alpha I
    return (a**0.5) * np.sqrt((U*U).sum(axis=-1))

def _manifest_write(path: str, payload: Dict[str, Any]):
    try:
        with open(path, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e: print(f"[traj] Warning: could not write manifest file. Error: {e}")

class RecordingPolicy(nn.Module):
    def __init__(self, base_policy: nn.Module):
        super().__init__(); self.base_policy = base_policy; self.X_frames, self.Y_frames, self.U_frames = [], [], []
    def forward(self, **states_dict):
        X, Y = states_dict.get("X"), states_dict.get("Y")
        if isinstance(X, torch.Tensor): self.X_frames.append(X.detach())
        if isinstance(Y, torch.Tensor): self.Y_frames.append(Y.detach())
        U = self.base_policy(**states_dict)
        if isinstance(U, torch.Tensor): self.U_frames.append(U.detach())
        return U
    def stacked(self) -> Dict[str, Optional[torch.Tensor]]:
        def _stack(frames: List[torch.Tensor]) -> Optional[torch.Tensor]:
            if not frames: return None
            return torch.stack(frames, dim=1).contiguous()
        return {"X": _stack(self.X_frames), "Y": _stack(self.Y_frames), "U": _stack(self.U_frames)}

# --- **MODIFIED: Correctly implemented PP policies for both modes** ---
class PPDirectPolicy(nn.Module):
    def __init__(self, stage1_policy: nn.Module, seed: int):
        super().__init__(); self.stage1_policy = stage1_policy; self.seed = seed
    def forward(self, **states_dict):
        if not _HAS_DIRECT: raise RuntimeError("P-PGDPO 'direct' utilities not available.")
        return ppgdpo_u_direct(self.stage1_policy, states_dict, REPEATS_DIR, SUBBATCH_DIR, self.seed)

class PPRUNNERPolicy(nn.Module):
    def __init__(self, stage1_policy: nn.Module, seed: int):
        super().__init__(); self.stage1_policy = stage1_policy; self.seed = seed
    def forward(self, **states_dict):
        if not _HAS_RUNNER: raise RuntimeError("P-PGDPO 'runner' utilities not available.")
        return ppgdpo_u_run(self.stage1_policy, states_dict, REPEATS_RUN, SUBBATCH_RUN, self.seed)

# ============ Plotting and Data Processing Logic (Unchanged) ============
def _series_for_view(view: Dict[str, Any], traj_np: Dict[str, Any], schema: Dict[str, Any], Bsel: List[int]) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    block_name = view.get("block", "X")
    arr = _to_np(traj_np[block_name]) if traj_np and traj_np.get(block_name) is not None else None
    if arr is None: return np.array([0.]), [], []
    arr_sel, steps = arr[Bsel], arr.shape[1]; x_time = _linspace_time(steps)
    labels_info = schema["roles"].get(block_name, {}).get("labels", [])
    mode = view.get("mode", "indices").lower()
    if mode == "aggregate":
        agg, idx = view.get("agg", "sum").lower(), view.get("indices", None)
        sub = arr_sel if idx is None else arr_sel[..., [i for i in idx if i < arr_sel.shape[-1]] or [0]]
        if agg in ("sum", "mean"): tmp = sub.sum(axis=-1) if agg == "sum" else sub.mean(axis=-1)
        elif agg in ("l2", "l2norm", "norm2", "euclid"): tmp = (sub**2).sum(axis=-1)**0.5
        else: tmp = sub.mean(axis=-1)
        series, label = [tmp.mean(axis=0)], view.get("ylabel") or f"{block_name}:{agg}"
        return x_time, series, [label]
    indices = view.get("indices", [0])
    valid_indices = [i for i in indices if i < arr_sel.shape[-1]] or [0]
    sub_arr, mean_series = arr_sel[..., valid_indices], arr_sel[..., valid_indices].mean(axis=0)
    series = [mean_series[:, i] for i in range(mean_series.shape[1])]
    labels = [labels_info[idx] if idx < len(labels_info) else f"{block_name}[{idx}]" for idx in valid_indices]
    return x_time, series, labels
def _plot_lines(x_time, series_map, title, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.0)); styles = {"cf": "-", "learn": "--", "pp": ":"}; colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx, used_labels = 0, set()
    for label, policies in sorted(series_map.items()):
        color = colors[color_idx % len(colors)]
        for policy, data in sorted(policies.items()):
            full_label = f"{label} — {policy}";
            if full_label not in used_labels: ax.plot(x_time, data, linestyle=styles.get(policy, '-'), color=color, label=full_label); used_labels.add(full_label)
        color_idx += 1
    ax.set(xlabel="Time", ylabel=ylabel, title=title); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc="best", fontsize=9); fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)

# ============ Main API ============

def _simulate_and_record(policy: nn.Module, B: int, rng: torch.Generator) -> Dict[str, Optional[torch.Tensor]]:
    recorder = RecordingPolicy(policy).to(device)
    try: init = sample_initial_states(B, rng=rng)
    except TypeError: init = sample_initial_states(B)
    initial_states = init[0] if isinstance(init, (tuple, list)) else init
    for kwargs in [{"initial_states_dict": initial_states, "m_steps": m, "train": False, "rng": rng}, {"initial_states_dict": initial_states, "m_steps": m}, {"initial_states_dict": initial_states}, {}]:
        try: simulate(recorder, B, **kwargs); return recorder.stacked()
        except TypeError: continue
    raise RuntimeError("Failed to call simulate with any known signature.")

def _handle_custom_view(view: dict, traj_np: dict, Bsel: list):
    mode = (view.get("mode","") or "").lower()
    if mode == "tracking_vpp":
        U = _to_np(traj_np.get("U")) if traj_np else None
        if U is None: return None
        U_sel = U[Bsel]                                 # (B', steps, d)
        steps = U_sel.shape[1]; x_time = _linspace_time(steps)
        u_sum_mean = U_sel.sum(axis=-1).mean(axis=0)    # (steps,)
        series, labels = [u_sum_mean], ["sum u (mean)"]
        if _REF_FN is not None:
            try:
                ref = _REF_FN(x_time) or {}
                if "Nagg" in ref:
                    series.append(np.asarray(ref["Nagg"]).reshape(-1))
                    labels.append("N_agg (ref)")
            except Exception:
                pass
        return x_time, series, labels
    if mode == "u_rnorm":
        U = _to_np(traj_np.get("U")) if traj_np else None
        if U is None: return None
        U_sel = U[Bsel]
        steps = U_sel.shape[1]; x_time = _linspace_time(steps)
        rn = _u_rnorm(U_sel).mean(axis=0)               # (steps,)
        return x_time, [rn], ["||u||_R (mean)"]
    return None


def _series_for_view_wrapper(view, traj_np, schema, Bsel):
    # 1) 커스텀 VPP 뷰 먼저
    out = _handle_custom_view(view, traj_np, Bsel)
    if out is not None:
        return out
    # 2) 기존 구현으로 폴백
    try:
        return _series_for_view(view, traj_np, schema, Bsel)  # 기존 함수가 있는 경우
    except NameError:
        # 최소 폴백: indices 0을 평균으로 그림
        block = view.get("block","X")
        arr = _to_np(traj_np.get(block)) if traj_np else None
        if arr is None:
            return _linspace_time(1), [], []
        arr_sel = arr[Bsel]; steps = arr_sel.shape[1]; x = _linspace_time(steps)
        idx = view.get("indices",[0]); idx = [i for i in idx if i < arr_sel.shape[-1]] or [0]
        sub = arr_sel[..., idx].mean(axis=0)            # (steps, d')
        series = [sub[:, j] for j in range(sub.shape[1])]
        labels = [f"{block}[{i}]" for i in idx]
        return x, series, labels

@torch.no_grad()
def generate_and_save_trajectories(policy_learn: nn.Module, policy_cf: Optional[nn.Module] = None, outdir: Optional[str] = None, B: Optional[int] = None, seed_crn: Optional[int] = None, seed: Optional[int] = None) -> str:
    outdir = _ensure_outdir(outdir); B_all = int(B or N_eval_states or 32); _seed_eff = seed_crn if seed_crn is not None else (seed or CRN_SEED_EU or 777); seed_crn = int(_seed_eff)
    gen_device = "cuda" if device.type == "cuda" else "cpu"; rng = torch.Generator(device=gen_device)
    
    rng.manual_seed(seed_crn); traj_learn = _simulate_and_record(policy_learn.to(device), B_all, rng)
    rng.manual_seed(seed_crn); traj_cf = _simulate_and_record(policy_cf.to(device), B_all, rng) if policy_cf else None

    traj_pp, _pp_error, pp_mode_used = None, None, 'none'
    if policy_cf:
        # --- **MODIFIED: Select correct PP policy based on mode** ---
        if PP_MODE == 'direct' and _HAS_DIRECT:
            pp_mode_used = 'direct'
            policy_class = PPDirectPolicy
        elif _HAS_RUNNER: # Default to runner
            pp_mode_used = 'runner'
            policy_class = PPRUNNERPolicy
        else:
            policy_class = None

        if policy_class:
            print(f"[traj] P-PGDPO simulation starting (mode: {pp_mode_used})...")
            try:
                rng.manual_seed(seed_crn)
                pp_policy = policy_class(policy_learn.to(device), seed=seed_crn)
                traj_pp = _simulate_and_record(pp_policy, B_all, rng)
                print("[traj] P-PGDPO simulation successful.")
            except Exception as e:
                _pp_error = f"{type(e).__name__}: {e}"; import traceback; print(f"\n[traj] P-PGDPO SIMULATION FAILED. Plots will be skipped.\nError: {_pp_error}"); traceback.print_exc(); print("")

    trajectories = {"learn": traj_learn, "cf": traj_cf, "pp": traj_pp}; Bsel = list(range(min(B_all, SCHEMA.get("sampling", {}).get("Bmax", 5)))); saved_files = []
    for view in SCHEMA.get("views", []):
        view_name, series_map_learn_cf, series_map_pp_cf = view.get("name", "view"), {}, {}; x_time = None
        for name, traj in trajectories.items():
            if traj is None: continue
            curr_x, series, labels = _series_for_view_wrapper(view, traj, SCHEMA, Bsel)
            if x_time is None and curr_x.size > 0: x_time = curr_x
            for s, lab in zip(series, labels):
                if name in ['learn', 'cf']:
                    if lab not in series_map_learn_cf: series_map_learn_cf[lab] = {}
                    series_map_learn_cf[lab][name] = s
                if name in ['pp', 'cf']:
                    if lab not in series_map_pp_cf: series_map_pp_cf[lab] = {}
                    series_map_pp_cf[lab][name] = s
        if x_time is not None and any('cf' in d for d in series_map_learn_cf.values()) and any('learn' in d for d in series_map_learn_cf.values()):
            path = os.path.join(outdir, f"{view_name}__learn_vs_cf.png"); _plot_lines(x_time, series_map_learn_cf, f"{view_name}: Learn vs CF", view.get("ylabel"), path); saved_files.append(path)
        if x_time is not None and any('cf' in d for d in series_map_pp_cf.values()) and any('pp' in d for d in series_map_pp_cf.values()):
            path = os.path.join(outdir, f"{view_name}__pp_vs_cf.png"); _plot_lines(x_time, series_map_pp_cf, f"{view_name}: PP vs CF", view.get("ylabel"), path); saved_files.append(path)
    _manifest_write(os.path.join(outdir, "traj_manifest.json"), {"outdir": outdir, "B_all": B_all, "seed_crn": seed_crn, "has_ppgdpo": traj_pp is not None, "pp_error": _pp_error, "saved_files": saved_files, "pp_mode": pp_mode_used})
    return outdir

if __name__ == "__main__":
    from user_pgdpo_base import DirectPolicy, build_closed_form_policy, N_eval_states
    learn_policy = DirectPolicy(); cf_policy, _ = build_closed_form_policy()
    if cf_policy is None: print("Warning: No closed-form policy found for CLI test.")
    output_directory = generate_and_save_trajectories(learn_policy, cf_policy, B=N_eval_states)
    print(f"\n[traj] Saved figures to: {output_directory}")