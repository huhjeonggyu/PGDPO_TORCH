# core/traj.py — schema-driven, multi-domain trajectory visualization
# - [업데이트] cf, pp, learn 정책 궤적을 하나의 그래프에 통합하여 출력
# - ✨ [추가] 시각화에 사용된 모든 궤적 데이터를 CSV 파일로 저장하는 기능

from __future__ import annotations
import os, json, time, csv
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pgdpo_base import (
    device, T, m, DIM_X, DIM_U, CRN_SEED_EU, N_eval_states,
    sample_initial_states, simulate,
)

PGDPO_CURRENT_MODE = os.getenv("PGDPO_CURRENT_MODE", "unknown")
PP_MODE = os.getenv('PP_MODE','runner').lower()
_HAS_RUNNER, _HAS_DIRECT = False, False
try: from pgdpo_run import ppgdpo_u_run, REPEATS as REPEATS_RUN, SUBBATCH as SUBBATCH_RUN; _HAS_RUNNER = True
except ImportError: pass
try: from pgdpo_with_projection import ppgdpo_u_direct, REPEATS as REPEATS_DIR, SUBBATCH as SUBBATCH_DIR; _HAS_DIRECT = True
except ImportError: pass

try: from user_pgdpo_base import ref_signals_fn as _REF_FN
except Exception: _REF_FN = None
try: from user_pgdpo_base import R_INFO as _R_INFO
except Exception: _R_INFO = {}

# --- Schema Logic (변경 없음) ---
def _try_load_user_schema() -> Optional[Dict[str, Any]]:
    try: from user_pgdpo_base import get_traj_schema; return get_traj_schema()
    except (ImportError, AttributeError): pass
    return None
def _get_base_default_views() -> List[Dict[str, Any]]:
    views: List[Dict[str, Any]] = []
    if DIM_X and int(DIM_X) > 0: views.append({"name": "X_first_components", "block": "X", "mode": "indices", "indices": [0], "ylabel": "X[0] Component"})
    if DIM_U and int(DIM_U) > 0: views.append({"name": "U_first_components", "block": "U", "mode": "indices", "indices": [0], "ylabel": "U[0] Component"})
    return views
def _load_and_merge_schemas() -> Dict[str, Any]:
    base_views = _get_base_default_views(); user_schema = _try_load_user_schema()
    if user_schema:
        if "views" not in user_schema: user_schema["views"] = []
        existing_view_names = {v.get("name") for v in user_schema["views"]}
        for view in base_views:
            if view["name"] not in existing_view_names: user_schema["views"].append(view)
        return user_schema
    else:
        return {"roles": {"X": {"dim": int(DIM_X or 0), "labels": [f"X_{i+1}" for i in range(int(DIM_X or 0))]}, "U": {"dim": int(DIM_U or 0), "labels": [f"u_{i+1}" for i in range(int(DIM_U or 0))]}}, "views": base_views, "sampling": {"Bmax": 5}}
SCHEMA = _load_and_merge_schemas()

# --- Helper Functions (변경 없음) ---
def _ensure_outdir(outdir: Optional[str]) -> str:
    if outdir is None: outdir = os.path.join("plots", "_traj", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True); return outdir
def _to_np(x: torch.Tensor) -> np.ndarray: return x.detach().cpu().numpy()
def _linspace_time(n_steps: int) -> np.ndarray: return np.linspace(0.0, float(T), num=n_steps)
class PPDirectPolicy(nn.Module):
    def __init__(self, stage1_policy: nn.Module, seed: int):
        super().__init__(); self.stage1_policy = stage1_policy; self.seed = seed
    def forward(self, **states_dict):
        if not _HAS_DIRECT: raise RuntimeError("'direct' utilities not available.")
        return ppgdpo_u_direct(self.stage1_policy, states_dict, REPEATS_DIR, SUBBATCH_DIR, self.seed)
class PPRUNNERPolicy(nn.Module):
    def __init__(self, stage1_policy: nn.Module, seed: int):
        super().__init__(); self.stage1_policy = stage1_policy; self.seed = seed
    def forward(self, **states_dict):
        if not _HAS_RUNNER: raise RuntimeError("'runner' utilities not available.")
        return ppgdpo_u_run(self.stage1_policy, states_dict, REPEATS_RUN, SUBBATCH_RUN, self.seed)
class RecordingPolicy(nn.Module):
    def __init__(self, base_policy: nn.Module):
        super().__init__(); self.base_policy = base_policy; self.X_frames, self.U_frames = [], []
    def forward(self, **states_dict):
        self.X_frames.append(states_dict["X"].detach())
        if "Y" in states_dict and states_dict["Y"] is not None:
            if not hasattr(self, 'Y_frames'): self.Y_frames = []
            self.Y_frames.append(states_dict["Y"].detach())
        U = self.base_policy(**states_dict)
        self.U_frames.append(U.detach()); return U
    def stacked(self) -> Dict[str, torch.Tensor]:
        data = {"X": torch.stack(self.X_frames, 1), "U": torch.stack(self.U_frames, 1)}
        if hasattr(self, 'Y_frames'): data["Y"] = torch.stack(self.Y_frames, 1)
        return data
def _simulate_and_record(policy: nn.Module, B: int, rng: torch.Generator, sync_time: bool = False) -> Dict[str, torch.Tensor]:
    recorder = RecordingPolicy(policy).to(device); init, _ = sample_initial_states(B, rng=rng)
    if sync_time: init['TmT'].fill_(T)
    simulate(recorder, B, initial_states_dict=init, m_steps=m, train=False, rng=rng)
    return recorder.stacked()
def _series_for_view(view: Dict[str, Any], traj_np: Dict[str, Any], Bsel: List[int]) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    block_name = view.get("block", "X")
    if block_name not in traj_np: return None, [], []
    arr = _to_np(traj_np.get(block_name)); arr_sel, steps = arr[Bsel], arr.shape[1]; x_time = _linspace_time(steps)
    labels_info = SCHEMA.get("roles", {}).get(block_name, {}).get("labels", []); indices = view.get("indices", [0]); valid_indices = [i for i in indices if i < arr_sel.shape[-1]]
    mean_series = arr_sel[..., valid_indices].mean(axis=0); series = [mean_series[:, i] for i in range(mean_series.shape[1])]; labels = [labels_info[idx] for idx in valid_indices]
    return x_time, series, labels
def _handle_custom_view(view: dict, traj_np: dict, Bsel: list, policy_tag: str | None = None) -> Optional[Tuple[np.ndarray, List[np.ndarray], List[str]]]:
    mode = (view.get("mode","") or "").lower()
    if mode == "u_rnorm":
        U = _to_np(traj_np["U"])[Bsel]; x_time = _linspace_time(U.shape[1])
        if "R" in _R_INFO: R = _R_INFO["R"].cpu().numpy(); u_Ru = np.einsum('bti,ij,btj->bt', U, R, U); r_norm_series = np.sqrt(u_Ru).mean(axis=0)
        else:
            alpha = _R_INFO.get("alpha", 0.0)
            if alpha <= 0: print("[WARN] Alpha for R-norm is not positive or R matrix not found. Norm will be zero."); r_norm_series = np.zeros(U.shape[1])
            else: r_norm_series = np.sqrt(alpha) * np.linalg.norm(U, axis=-1).mean(axis=0)
        return x_time, [r_norm_series], [view.get("block", "U")]
    if mode == "tracking_vpp":
        U = _to_np(traj_np["U"])[Bsel]; x_time = _linspace_time(U.shape[1]); u_total_power = U.sum(axis=-1).mean(axis=0); series, labels = [u_total_power], ["Output_Power_Sum"]
        if _REF_FN:
            ref_signals = _REF_FN(x_time)
            for key, values in ref_signals.items(): series.append(np.asarray(values).reshape(-1)); labels.append(f"{key}_ref")
        return x_time, series, labels
    return None
def _series_for_view_wrapper(view, traj_np, Bsel, policy_tag=None):
    if (view.get("name","").lower().startswith("tracking")): view = {**view, "mode": "tracking_vpp"}
    custom_result = _handle_custom_view(view, traj_np, Bsel, policy_tag)
    if custom_result is not None: return custom_result
    return _series_for_view(view, traj_np, Bsel)

# --- ✨ [신규] CSV 저장 헬퍼 함수 ---
def _save_series_to_csv(x_time, all_views_data, out_path):
    """
    모든 뷰의 시계열 데이터 맵을 하나의 CSV 파일로 저장합니다.
    """
    header = ["Time"]
    all_columns = {}
    for view_name, series_map in all_views_data.items():
        for component, policies in sorted(series_map.items()):
            for policy, data in sorted(policies.items()):
                col_name = f"{component}_{policy}"
                if "ref" in component: col_name = component # 참조 신호는 이름 그대로 사용
                if col_name not in header: header.append(col_name)
                all_columns[col_name] = data

    # 데이터를 순서대로 리스트에 담기
    data_rows = [x_time]
    for h in header:
        if h == "Time": continue
        data_rows.append(all_columns.get(h, [""] * len(x_time))) # 데이터가 없는 경우 빈 문자열
    
    # 행렬 형태로 변환 (전치)
    rows = np.stack(data_rows, axis=-1)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[traj] Trajectory data saved to: {os.path.basename(out_path)}")

# --- ✨ [업데이트] 통합 라인 플롯 (정책별 고유 색상/스타일 적용) ---
def _plot_lines(x_time, series_map, title, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    POLICY_STYLES = {
        "cf":    {"color": "royalblue", "ls": "-",  "lw": 2.5, "zorder": 5, "label": "Closed-Form (cf)"},
        "pp":    {"color": "orangered", "ls": "--", "lw": 1.8, "zorder": 4, "label": "P-PGDPO (pp)", "marker": 'o', "ms": 4, "alpha": 0.7},
        "learn": {"color": "forestgreen", "ls": ":",  "lw": 1.8, "zorder": 3, "label": "Learned", "marker": 's', "ms": 4, "alpha": 0.7},
        "ref":   {"color": "gray", "ls": "-.", "lw": 1.5, "zorder": 2}
    }
    for component_label, policies in sorted(series_map.items()):
        if "ref" in component_label:
            ref_label = POLICY_STYLES["ref"].get("label", component_label)
            if 'cf' in policies: ax.plot(x_time, policies['cf'], label=ref_label, **POLICY_STYLES["ref"])
            continue
        for policy_key in ['cf', 'pp', 'learn']:
            if policy_key in policies:
                data = policies[policy_key]
                style_props = POLICY_STYLES.get(policy_key, {})
                full_label = f"{component_label} - {style_props.get('label')}"
                ax.plot(x_time, data, label=full_label, **{k:v for k,v in style_props.items() if k != 'label'})
    ax.set(xlabel="Time", ylabel=ylabel, title=title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=9); fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)

# --- ✨ [업데이트] Main Trajectory Generation Function ---
@torch.no_grad()
def generate_and_save_trajectories(policy_learn: nn.Module, policy_cf: Optional[nn.Module] = None, outdir: Optional[str] = None, B: Optional[int] = None, seed_crn: Optional[int] = None) -> str:
    outdir = _ensure_outdir(outdir); B_all = int(B or N_eval_states); seed_crn = int(seed_crn or CRN_SEED_EU)
    def _g(): return torch.Generator(device=device).manual_seed(seed_crn)
    
    print("[traj] Simulating trajectories for 'learn' policy...")
    traj_learn = _simulate_and_record(policy_learn.to(device), B_all, _g(), sync_time=True)
    
    traj_cf = _simulate_and_record(policy_cf.to(device), B_all, _g(), sync_time=True) if policy_cf else None
    
    traj_pp = None
    if policy_cf and PGDPO_CURRENT_MODE in ["run", "projection", "residual"]:
        if PP_MODE == 'direct' and _HAS_DIRECT:
            print("[traj] Generating 'pp' trajectories using DIRECT method.")
            traj_pp = _simulate_and_record(PPDirectPolicy(policy_learn.to(device), seed_crn), B_all, _g(), sync_time=True)
        elif _HAS_RUNNER:
            print("[traj] Generating 'pp' trajectories using RUNNER method.")
            traj_pp = _simulate_and_record(PPRUNNERPolicy(policy_learn.to(device), seed_crn), B_all, _g(), sync_time=True)

    trajectories = {"learn": traj_learn, "cf": traj_cf, "pp": traj_pp}
    Bsel = list(range(min(B_all, SCHEMA.get("sampling", {}).get("Bmax", 5))))
    
    all_views_data = {}
    master_x_time = None

    for view in SCHEMA.get("views", []):
        series_map = {}
        x_time = None
        for name, traj in trajectories.items():
            if traj is None: continue
            
            curr_x, series, labels = _series_for_view_wrapper(view, traj, Bsel, policy_tag=name)
            if curr_x is None or not series: continue
            if x_time is None: x_time = curr_x
            if master_x_time is None: master_x_time = x_time
            
            for s, lab in zip(series, labels):
                if lab not in series_map: series_map[lab] = {}
                series_map[lab][name] = s
        
        if x_time is not None and series_map:
            _plot_lines(x_time, series_map, f"Trajectory Comparison: {view['name']}", view.get("ylabel","Value"), os.path.join(outdir, f"traj_{view['name']}_comparison.png"))
            all_views_data[view['name']] = series_map
                        
    if master_x_time is not None and all_views_data:
        _save_series_to_csv(master_x_time, all_views_data, os.path.join(outdir, "traj_comparison_data.csv"))
        
    return outdir