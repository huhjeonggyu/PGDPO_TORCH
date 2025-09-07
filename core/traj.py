# core/traj.py — schema-driven, multi-domain trajectory visualization
# - Features PP_MODE ('runner' vs 'direct') to toggle between simulation types.
# - The 'direct' mode now correctly calls the same function used for RMSE calculation.
# - Generates separate, clear "learn vs cf" and "pp vs cf" plots.

from __future__ import annotations
import os, json, time, random
from typing import Any, Dict, List, Optional, Tuple

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

PP_MODE = os.getenv('PP_MODE','runner').lower()
_HAS_RUNNER, _HAS_DIRECT = False, False
try:
    from pgdpo_run import ppgdpo_u_run, REPEATS as REPEATS_RUN, SUBBATCH as SUBBATCH_RUN
    _HAS_RUNNER = True
except ImportError: pass
try:
    from pgdpo_with_projection import ppgdpo_u_direct, REPEATS as REPEATS_DIR, SUBBATCH as SUBBATCH_DIR
    _HAS_DIRECT = True
except ImportError: pass

# ============ Schema loading & defaults (Logic Corrected) ============
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
        views.append({"name": "X_first_components", "block": "X", "mode": "indices", "indices": list(range(min(3, xdim))), "ylabel": "X components"})
        if xdim > 1: views.append({"name": "X_sum", "block": "X", "mode": "aggregate", "agg": "sum", "ylabel": "Σ X"})
    if DIM_U and int(DIM_U) > 0:
        udim = int(DIM_U)
        views.append({"name": "U_first_components", "block": "U", "mode": "indices", "indices": list(range(min(2, udim))), "ylabel": "U components"})
        if udim > 1: views.append({"name": "U_l2", "block": "U", "mode": "aggregate", "agg": "l2norm", "ylabel": "‖u‖₂"})

    return {
        "roles": {
            "X": {"dim": int(DIM_X or 0), "labels": _default_labels("X", int(DIM_X or 0))},
            "Y": {"dim": int(DIM_Y or 0), "labels": _default_labels("Y", int(DIM_Y or 0))},
            "U": {"dim": int(DIM_U or 0), "labels": _default_labels("u", int(DIM_U or 0))},
        },
        "views": views,
        "sampling": {"Bmax": 5, "strategy": "head", "seed": 777},
    }

# --- 스키마 로딩 및 병합 로직 수정 ---
SCHEMA = _try_load_user_schema()
if SCHEMA is None:
    SCHEMA = _build_default_schema()
else:
    if "views" not in SCHEMA: SCHEMA["views"] = []
    
    # 'indices' 모드로 첫 번째 성분을 그리는 뷰가 있는지 확인
    has_x0_view = any(v.get("block") == "X" and v.get("mode") == "indices" and 0 in v.get("indices", []) for v in SCHEMA["views"])
    has_u0_view = any(v.get("block") == "U" and v.get("mode") == "indices" and 0 in v.get("indices", []) for v in SCHEMA["views"])

    if DIM_X and int(DIM_X) > 0 and not has_x0_view:
        SCHEMA["views"].insert(0, {"name": "X_first_element", "block": "X", "mode": "indices", "indices": [0], "ylabel": f"X[0] Component"})
    if DIM_U and int(DIM_U) > 0 and not has_u0_view:
        SCHEMA["views"].insert(1, {"name": "U_first_element", "block": "U", "mode": "indices", "indices": [0], "ylabel": f"U[0] Component"})

# ============ Utilities & Policy Wrappers ============
try: from user_pgdpo_base import ref_signals_fn as _REF_FN
except Exception: _REF_FN = None
try: from user_pgdpo_base import R_INFO as _R_INFO
except Exception: _R_INFO = {}

def _ensure_outdir(outdir: Optional[str]) -> str:
    if outdir is None: outdir = os.path.join("plots", "_traj", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True); return outdir
def _to_np(x) -> np.ndarray:
    if isinstance(x, np.ndarray): return x
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    raise TypeError(f"Cannot convert type {type(x)} to numpy array.")
def _linspace_time(n_steps: int) -> np.ndarray:
    return np.linspace(0.0, float(T), num=n_steps) if n_steps > 1 else np.array([0.0])
def _u_rnorm(U: np.ndarray) -> np.ndarray:
    if "R" in _R_INFO: return np.sqrt(np.einsum("bsd,dd,bsd->bs", U, np.asarray(_R_INFO["R"]), U))
    if "R_diag" in _R_INFO: return np.sqrt(np.sum(U**2 * np.asarray(_R_INFO["R_diag"]).reshape(1,1,-1), axis=-1))
    return (float(_R_INFO.get("alpha", 1.0))**0.5) * np.linalg.norm(U, axis=-1)
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
            return torch.stack(frames, dim=1).contiguous() if frames else None
        return {"X": _stack(self.X_frames), "Y": _stack(self.Y_frames), "U": _stack(self.U_frames)}

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

def _series_for_view(view: Dict[str, Any], traj_np: Dict[str, Any], schema: Dict[str, Any], Bsel: List[int]) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    block_name = view.get("block", "X")
    arr = _to_np(traj_np.get(block_name)) if traj_np and traj_np.get(block_name) is not None else None
    if arr is None: return np.array([0.]), [], []
    arr_sel, steps = arr[Bsel], arr.shape[1]; x_time = _linspace_time(steps)
    labels_info = schema.get("roles", {}).get(block_name, {}).get("labels", []) or _default_labels(block_name, arr.shape[-1])
    mode = view.get("mode", "indices").lower()
    if mode == "aggregate":
        agg, idx = view.get("agg", "sum").lower(), view.get("indices", None)
        sub = arr_sel if idx is None else arr_sel[..., [i for i in idx if i < arr_sel.shape[-1]] or [0]]
        if agg in ("sum", "mean"): tmp = sub.sum(axis=-1) if agg == "sum" else sub.mean(axis=-1)
        elif agg in ("l2", "l2norm", "norm2", "euclid"): tmp = np.linalg.norm(sub, axis=-1)
        else: tmp = sub.mean(axis=-1)
        series, label = [tmp.mean(axis=0)], [view.get("ylabel") or f"{block_name}:{agg}"]
        return x_time, series, label
    indices = view.get("indices", [0])
    valid_indices = [i for i in indices if i < arr_sel.shape[-1]] or [0]
    mean_series = arr_sel[..., valid_indices].mean(axis=0)
    series = [mean_series[:, i] for i in range(mean_series.shape[1])]
    labels = [labels_info[idx] if idx < len(labels_info) else f"{block_name}[{idx}]" for idx in valid_indices]
    return x_time, series, labels

def _plot_lines(x_time, series_map, title, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.0)); styles = {"cf": "-", "learn": "--", "pp": ":"}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    used_labels = set()
    for i, (label, policies) in enumerate(sorted(series_map.items())):
        color = colors[i % len(colors)]
        for policy, data in sorted(policies.items()):
            full_label = f"{label} — {policy}"
            if full_label not in used_labels:
                ax.plot(x_time, data, linestyle=styles.get(policy, '-'), color=color, label=full_label); used_labels.add(full_label)
    ax.set(xlabel="Time", ylabel=ylabel, title=title); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc="best", fontsize=9); fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)

def _simulate_and_record(policy: nn.Module, B: int, rng: torch.Generator) -> Dict[str, Optional[torch.Tensor]]:
    recorder = RecordingPolicy(policy).to(device)
    init = sample_initial_states(B, rng=rng)[0]
    simulate(recorder, B, initial_states_dict=init, m_steps=m, train=False, rng=rng)
    return recorder.stacked()

def _handle_custom_view(view: dict, traj_np: dict, Bsel: list):
    mode = (view.get("mode","") or "").lower()
    U = _to_np(traj_np.get("U")) if traj_np else None
    if U is None: return None
    steps = U[Bsel].shape[1]; x_time = _linspace_time(steps)
    if mode == "tracking_vpp":
        u_sum_mean = U[Bsel].sum(axis=-1).mean(axis=0)
        series, labels = [u_sum_mean], ["sum u (mean)"]
        if _REF_FN:
            ref = _REF_FN(x_time) or {}
            if "Nagg" in ref: series.append(np.asarray(ref["Nagg"]).reshape(-1)); labels.append("N_agg (ref)")
        return x_time, series, labels
    if mode == "u_rnorm":
        return x_time, [_u_rnorm(U[Bsel]).mean(axis=0)], ["||u||_R (mean)"]
    return None

def _series_for_view_wrapper(view, traj_np, schema, Bsel):
    custom_result = _handle_custom_view(view, traj_np, Bsel)
    return custom_result if custom_result is not None else _series_for_view(view, traj_np, schema, Bsel)

@torch.no_grad()
def generate_and_save_trajectories(policy_learn: nn.Module, policy_cf: Optional[nn.Module] = None, outdir: Optional[str] = None, B: Optional[int] = None, seed_crn: Optional[int] = None, seed: Optional[int] = None) -> str:
    outdir = _ensure_outdir(outdir); B_all = int(B or N_eval_states or 32); _seed_eff = seed_crn if seed_crn is not None else (seed or CRN_SEED_EU or 777); seed_crn = int(_seed_eff)
    gen_device = "cuda" if device.type == "cuda" else "cpu"; rng = torch.Generator(device=gen_device)

    rng.manual_seed(seed_crn); traj_learn = _simulate_and_record(policy_learn.to(device), B_all, rng)
    rng.manual_seed(seed_crn); traj_cf = _simulate_and_record(policy_cf.to(device), B_all, rng) if policy_cf else None

    traj_pp, _pp_error, pp_mode_used = None, None, 'none'
    if policy_cf:
        policy_class = PPRUNNERPolicy if PP_MODE == 'runner' and _HAS_RUNNER else PPDirectPolicy if PP_MODE == 'direct' and _HAS_DIRECT else None
        if policy_class:
            pp_mode_used = PP_MODE
            print(f"[traj] P-PGDPO simulation starting (mode: {pp_mode_used})...")
            try:
                rng.manual_seed(seed_crn)
                pp_policy = policy_class(policy_learn.to(device), seed=seed_crn)
                traj_pp = _simulate_and_record(pp_policy, B_all, rng)
                print("[traj] P-PGDPO simulation successful.")
            except Exception as e:
                _pp_error = f"{type(e).__name__}: {e}"; import traceback; print(f"\n[traj] P-PGDPO SIMULATION FAILED. Plots will be skipped.\nError: {_pp_error}"); traceback.print_exc(); print("")

    trajectories = {"learn": traj_learn, "cf": traj_cf, "pp": traj_pp}
    Bsel = list(range(min(B_all, SCHEMA.get("sampling", {}).get("Bmax", 5))))
    saved_files = []

    for view in SCHEMA.get("views", []):
        view_name = view.get("name", "view")
        series_map_learn_cf, series_map_pp_cf = {}, {}
        x_time = None
        
        # --- 버그 수정된 데이터 집계 로직 ---
        for name, traj in trajectories.items():
            if traj is None: continue
            curr_x, series, labels = _series_for_view_wrapper(view, traj, SCHEMA, Bsel)
            if x_time is None and curr_x.size > 0: x_time = curr_x
            for s, lab in zip(series, labels):
                if name == 'learn':
                    if lab not in series_map_learn_cf: series_map_learn_cf[lab] = {}
                    series_map_learn_cf[lab]['learn'] = s
                elif name == 'pp':
                    if lab not in series_map_pp_cf: series_map_pp_cf[lab] = {}
                    series_map_pp_cf[lab]['pp'] = s
                elif name == 'cf':
                    if lab not in series_map_learn_cf: series_map_learn_cf[lab] = {}
                    series_map_learn_cf[lab]['cf'] = s
                    if lab not in series_map_pp_cf: series_map_pp_cf[lab] = {}
                    series_map_pp_cf[lab]['cf'] = s

        if x_time is not None and any('cf' in d and 'learn' in d for d in series_map_learn_cf.values()):
            path = os.path.join(outdir, f"{view_name}__learn_vs_cf.png"); _plot_lines(x_time, series_map_learn_cf, f"{view_name}: Learn vs CF", view.get("ylabel",""), path); saved_files.append(path)
        if x_time is not None and any('cf' in d and 'pp' in d for d in series_map_pp_cf.values()):
            path = os.path.join(outdir, f"{view_name}__pp_vs_cf.png"); _plot_lines(x_time, series_map_pp_cf, f"{view_name}: PP vs CF", view.get("ylabel",""), path); saved_files.append(path)

    _manifest_write(os.path.join(outdir, "traj_manifest.json"), {"outdir": outdir, "B_all": B_all, "seed_crn": seed_crn, "has_ppgdpo": traj_pp is not None, "pp_error": _pp_error, "saved_files": saved_files, "pp_mode": pp_mode_used})
    return outdir


if __name__ == "__main__":
    from user_pgdpo_base import DirectPolicy, build_closed_form_policy, N_eval_states
    learn_policy = DirectPolicy(); cf_policy, _ = build_closed_form_policy()
    if cf_policy is None: print("Warning: No closed-form policy found for CLI test.")
    output_directory = generate_and_save_trajectories(learn_policy, cf_policy, B=N_eval_states)
    print(f"\n[traj] Saved figures to: {output_directory}")