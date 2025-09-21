# 파일: core/traj.py

# -*- coding: utf-8 -*-
# core/traj.py — schema-driven, multi-domain trajectory visualization
# - [업데이트] cf, pp, learn 정책 궤적을 하나의 그래프에 통합하여 출력
# - ✨ [추가] 시각화에 사용된 모든 궤적 데이터를 CSV 파일로 저장하는 기능
# - ✨ [SIR] R_t, Co-state 기록 및 SIR 전용 커스텀 플롯 모드 추가

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
    device, T, m, d, k, DIM_X, DIM_U, CRN_SEED_EU, N_eval_states,
    sample_initial_states, simulate,
)

PGDPO_TRAJ_B  = int(os.getenv("PGDPO_TRAJ_B", 1))
PGDPO_CURRENT_MODE = os.getenv("PGDPO_CURRENT_MODE", "unknown")
PP_MODE = os.getenv('PP_MODE','direct').lower()
_HAS_RUNNER, _HAS_DIRECT = False, False
try: from pgdpo_run import ppgdpo_u_run, REPEATS as REPEATS_RUN, SUBBATCH as SUBBATCH_RUN; _HAS_RUNNER = True
except ImportError: pass
try: from pgdpo_with_projection import ppgdpo_u_direct, REPEATS as REPEATS_DIR, SUBBATCH as SUBBATCH_DIR, estimate_costates, PP_NEEDS; _HAS_DIRECT = True
except ImportError: estimate_costates, PP_NEEDS = None, ()


# --- User-defined functions ---
try: from user_pgdpo_base import ref_signals_fn as _REF_FN
except Exception: _REF_FN = None
try: from user_pgdpo_base import R_INFO as _R_INFO
except Exception: _R_INFO = {}
try: from user_pgdpo_base import price as harvesting_income
except ImportError: harvesting_income = None
try: from user_pgdpo_base import calculate_rt
except (ImportError, AttributeError): calculate_rt = None


# --- Schema Logic ---
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

# --- Helper Functions ---
def _ensure_outdir(outdir: Optional[str]) -> str:
    if outdir is None: outdir = os.path.join("plots", "_traj", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True); return outdir
def _to_np(x: torch.Tensor) -> np.ndarray: return x.detach().cpu().numpy()
def _linspace_time(n_steps: int) -> np.ndarray: return np.linspace(0.0, float(T), num=n_steps)

# ===== (핵심 수정) P-PGDPO 래퍼 클래스 =====
class PPDirectPolicy(nn.Module):
    def __init__(self, stage1_policy: nn.Module, seed: int):
        super().__init__(); self.stage1_policy = stage1_policy; self.seed = seed
    
    def forward(self, **states_dict):
        if not _HAS_DIRECT: raise RuntimeError("'direct' utilities not available.")
        
        # 1. P-PGDPO로 투자(u)만 프로젝션
        u_projected = ppgdpo_u_direct(self.stage1_policy, states_dict, repeats=REPEATS_DIR, sub_batch=SUBBATCH_DIR, seed_eval=self.seed)

        # 2. 원본 정책의 전체 출력을 확인하여 소비 모델인지 판별
        with torch.no_grad():
            original_output = self.stage1_policy(**states_dict)
        
        is_consumption_model = original_output.size(1) > u_projected.size(1)

        if is_consumption_model:
            # 3. 소비 모델이면, 원본 정책에서 소비(C) 값을 가져와 결합
            c_learned = original_output[:, u_projected.size(1):]
            return torch.cat([u_projected, c_learned], dim=1)
        else:
            # 4. 아니면, 프로젝션된 투자(u)만 반환
            return u_projected

class PPRUNNERPolicy(nn.Module):
    def __init__(self, stage1_policy: nn.Module, seed: int):
        super().__init__(); self.stage1_policy = stage1_policy; self.seed = seed
    
    def forward(self, **states_dict):
        if not _HAS_RUNNER: raise RuntimeError("'runner' utilities not available.")

        # PPDirectPolicy와 동일한 로직 적용
        u_projected = ppgdpo_u_run(self.stage1_policy, states_dict, REPEATS_RUN, SUBBATCH_RUN, self.seed)
        
        with torch.no_grad():
            original_output = self.stage1_policy(**states_dict)

        is_consumption_model = original_output.size(1) > u_projected.size(1)

        if is_consumption_model:
            c_learned = original_output[:, u_projected.size(1):]
            return torch.cat([u_projected, c_learned], dim=1)
        else:
            return u_projected

class RecordingPolicy(nn.Module):
    def __init__(self, base_policy: nn.Module):
        super().__init__()
        self.base_policy = base_policy
        self.X_frames, self.U_frames = [], []
        self.HarvestingIncome_frames = []
        self.Rt_frames = []
        self.Costate_I_frames = []

    def forward(self, **states_dict):
        self.X_frames.append(states_dict["X"].detach())
        if "Y" in states_dict and states_dict["Y"] is not None:
            if not hasattr(self, 'Y_frames'): self.Y_frames = []
            self.Y_frames.append(states_dict["Y"].detach())
        
        U = self.base_policy(**states_dict)
        self.U_frames.append(U.detach())

        if harvesting_income is not None:
            X = states_dict["X"].detach()
            p_device = harvesting_income.to(X.device)
            current_income = ((U * X) * p_device.view(1, -1)).sum(dim=1, keepdim=True)
            self.HarvestingIncome_frames.append(current_income.detach())

        if calculate_rt is not None:
            X = states_dict["X"].detach()
            S = X[:, 0::3]
            current_rt = calculate_rt(S).sum(dim=1, keepdim=True)
            self.Rt_frames.append(current_rt)
            
            mode = os.getenv("PGDPO_CURRENT_MODE")
            if estimate_costates and mode in ["projection", "run", "base"]:
                try:
                    with torch.enable_grad():
                        costates = estimate_costates(simulate, self.base_policy, {k:v.detach().clone().requires_grad_(True) for k,v in states_dict.items() if k in ['X', 'Y']},
                                                     repeats=16, sub_batch=16, seed_eval=123, needs=PP_NEEDS)
                    JX = costates.get("JX")
                    if JX is not None:
                        pI = JX[:, 1::3].sum(dim=1, keepdim=True)
                        self.Costate_I_frames.append(pI.detach())
                except Exception:
                    pass

        return U

    def stacked(self) -> Dict[str, torch.Tensor]:
        data = {"X": torch.stack(self.X_frames, 1), "U": torch.stack(self.U_frames, 1)}
        if hasattr(self, 'Y_frames'): data["Y"] = torch.stack(self.Y_frames, 1)
        if self.HarvestingIncome_frames: data["HarvestingIncome"] = torch.stack(self.HarvestingIncome_frames, 1)
        if self.Rt_frames: data["Rt"] = torch.stack(self.Rt_frames, 1)
        if self.Costate_I_frames: data["Costate_I"] = torch.stack(self.Costate_I_frames, 1)
        return data
        
def _simulate_and_record(policy: nn.Module, B: int, rng: torch.Generator, sync_time: bool = False) -> Dict[str, torch.Tensor]:
    recorder = RecordingPolicy(policy).to(device); init, _ = sample_initial_states(B, rng=rng)
    if sync_time: init['TmT'].fill_(T)
    simulate(recorder, B, initial_states_dict=init, m_steps=m, train=False, rng=rng)
    return recorder.stacked()

def _series_for_view(view: Dict[str, Any], traj_np: Dict[str, Any], Bsel: List[int]) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    block_names = view.get("block", "X")
    if isinstance(block_names, str): block_names = [block_names]
    
    x_time, all_series, all_labels = None, [], []

    for block_name in block_names:
        if block_name not in traj_np: continue
        arr = _to_np(traj_np.get(block_name))
        arr_sel, steps = arr[Bsel], arr.shape[1]
        if x_time is None: x_time = _linspace_time(steps)
        
        # 'roles'에서 레이블 정보와 차원(dim)을 가져옵니다.
        role_info = SCHEMA.get("roles", {}).get(block_name, {})
        labels_info = role_info.get("labels", [])
        
        # view에서 indices를 가져옵니다.
        indices = view.get("indices", [0])
        
        # 유효한 인덱스만 필터링합니다.
        valid_indices = [i for i in indices if i < arr_sel.shape[-1]]
        
        mean_series = arr_sel[..., valid_indices].mean(axis=0)
        series = [mean_series[:, i] for i in range(mean_series.shape[1])]
        labels = [labels_info[idx] for idx in valid_indices] if labels_info else [f"dim_{idx}" for idx in valid_indices]
        all_series.extend(series)
        all_labels.extend(labels)
        
    return x_time, all_series, all_labels

def _handle_custom_view(view: dict, traj_np: dict, Bsel: list, policy_tag: str | None = None) -> Optional[Tuple[np.ndarray, List[np.ndarray], List[str]]]:
    mode = (view.get("mode","") or "").lower()
    if mode == "u_rnorm":
        U = _to_np(traj_np["U"])[Bsel, :, :d] # 소비 제외
        x_time = _linspace_time(U.shape[1])
        if "R" in _R_INFO: R = _R_INFO["R"].cpu().numpy(); u_Ru = np.einsum('bti,ij,btj->bt', U, R, U); r_norm_series = np.sqrt(u_Ru).mean(axis=0)
        else:
            alpha = _R_INFO.get("alpha", 0.0)
            if alpha <= 0: r_norm_series = np.zeros(U.shape[1])
            else: r_norm_series = np.sqrt(alpha) * np.linalg.norm(U, axis=-1).mean(axis=0)
        return x_time, [r_norm_series], [view.get("block", "U")]
    if mode == "tracking_vpp":
        U = _to_np(traj_np["U"])[Bsel]; x_time = _linspace_time(U.shape[1]); u_total_power = U.sum(axis=-1).mean(axis=0); series, labels = [u_total_power], ["Output_Power_Sum"]
        if _REF_FN:
            ref_signals = _REF_FN(x_time)
            for key, values in ref_signals.items(): series.append(np.asarray(values).reshape(-1)); labels.append(f"{key}_ref")
        return x_time, series, labels
    
    if mode == "custom_sir_rt_plot":
        Rt = _to_np(traj_np.get("Rt"))[Bsel]
        x_time = _linspace_time(Rt.shape[1])
        rt_series = Rt.mean(axis=0).flatten()
        
        X = _to_np(traj_np.get("X"))[Bsel]
        i_indices = [i for i in range(1, X.shape[-1], 3)]
        i_series = X[..., i_indices].sum(axis=-1).mean(axis=0).flatten()
        return x_time, [rt_series, i_series], ["Effective_Rt", "Total_Infected"]

    if mode == "custom_sir_i_plot":
        X = _to_np(traj_np.get("X"))[Bsel]
        x_time = _linspace_time(X.shape[1])
        i_indices = [i for i in range(1, X.shape[-1], 3)]
        i_series = X[..., i_indices].sum(axis=-1).mean(axis=0).flatten()
        return x_time, [i_series], ["Total_Infected"]
        
    return None

def _series_for_view_wrapper(view, traj_np, Bsel, policy_tag=None):
    if (view.get("name","").lower().startswith("tracking")): view = {**view, "mode": "tracking_vpp"}
    custom_result = _handle_custom_view(view, traj_np, Bsel, policy_tag)
    if custom_result is not None: return custom_result
    return _series_for_view(view, traj_np, Bsel)

def _save_series_to_csv(x_time, all_views_data, out_path):
    header = ["Time"]
    all_columns = {}
    for view_name, series_map in all_views_data.items():
        for component, policies in sorted(series_map.items()):
            for policy, data in sorted(policies.items()):
                col_name = f"{component}_{policy}"
                if "ref" in component: col_name = component
                if col_name not in header: header.append(col_name)
                all_columns[col_name] = data

    data_rows = [x_time]
    for h in header:
        if h == "Time": continue
        data_rows.append(all_columns.get(h, [""] * len(x_time)))
    
    rows = np.stack(data_rows, axis=-1)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[traj] Trajectory data saved to: {os.path.basename(out_path)}")

def _save_full_trajectories_to_csv(
    trajectories: Dict[str, Optional[Dict[str, torch.Tensor]]],
    schema: Dict[str, Any],
    Bsel: List[int],
    out_path: str
):
    """모든 정책의 모든 상태/제어 변수를 CSV 파일로 저장합니다."""
    if not trajectories: return

    # 1. 시간 축 생성
    first_valid_traj = next((t for t in trajectories.values() if t is not None), None)
    if first_valid_traj is None: return
    n_steps = first_valid_traj['X'].shape[1]
    x_time = _linspace_time(n_steps)

    # 2. 헤더 생성
    header = ["Time"]
    all_data_map = {}
    
    roles = schema.get("roles", {})
    
    for policy_name, traj_data in trajectories.items():
        if traj_data is None: continue
        
        # 각 데이터 블록 (X, U 등)에 대해
        for block_name, block_tensor in traj_data.items():
            if block_name not in roles: continue
            
            # Bsel 샘플들의 평균을 계산
            block_np_avg = _to_np(block_tensor[Bsel]).mean(axis=0) # (n_steps, dim)
            
            labels = roles[block_name].get("labels", [f"{block_name}_{i}" for i in range(block_np_avg.shape[1])])
            
            for i in range(block_np_avg.shape[1]):
                col_name = f"{labels[i]}_{policy_name}"
                header.append(col_name)
                all_data_map[col_name] = block_np_avg[:, i]

    # 3. CSV 파일 작성
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for i in range(n_steps):
            row = [x_time[i]]
            for col_name in header[1:]:
                row.append(all_data_map.get(col_name, np.array([]))[i])
            writer.writerow(row)
            
    print(f"[traj] Full trajectory data saved to: {os.path.basename(out_path)}")

def _plot_lines(x_time, series_map, title, ylabel, save_path, view_opts: dict = {}):
    is_dual_axis = view_opts.get("mode") == "dual_axis"
    
    # --- ✨ 수정된 부분 시작 ---

    # 1. 기본 스타일과 라벨을 먼저 정의합니다.
    POLICY_STYLES = {
        "cf":    {"color": "royalblue", "ls": "-",  "lw": 2.5, "zorder": 5, "label": "Ref/Myopic"},
        "pp":    {"color": "orangered", "ls": "--", "lw": 1.8, "zorder": 4, "label": "P-PGDPO (pp)", "marker": 'o', "ms": 4, "alpha": 0.7},
        "learn": {"color": "forestgreen", "ls": ":",  "lw": 1.8, "zorder": 3, "label": "Learned", "marker": 's', "ms": 4, "alpha": 0.7},
        "ref":   {"color": "gray", "ls": "-.", "lw": 1.5, "zorder": 2}
    }

    # 2. user_pgdpo_base.py의 get_traj_schema에 custom 라벨이 정의되어 있으면, 기본값을 덮어씁니다.
    if "legend_labels" in SCHEMA:
        custom_labels = SCHEMA["legend_labels"]
        for policy_key, new_label in custom_labels.items():
            if policy_key in POLICY_STYLES:
                POLICY_STYLES[policy_key]["label"] = new_label
    
    # --- ✨ 수정된 부분 끝 ---

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax2 = ax1.twinx() if is_dual_axis else None
    
    ax_map = {}
    if is_dual_axis:
        for component_label in series_map.keys():
            if "costate" in component_label.lower() or "shadow" in component_label.lower():
                ax_map[component_label] = ax2
            else: ax_map[component_label] = ax1
    else:
        for component_label in series_map.keys(): ax_map[component_label] = ax1

    for component_label, policies in sorted(series_map.items()):
        ax = ax_map.get(component_label, ax1)
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

    ax1.set(xlabel="Time", title=title)
    if is_dual_axis and "ylabels" in view_opts:
        ax1.set_ylabel(view_opts["ylabels"][0])
        if ax2: ax2.set_ylabel(view_opts["ylabels"][1])
    else:
        ax1.set_ylabel(ylabel)

    if "h_lines" in view_opts:
        for line in view_opts["h_lines"]:
            ax1.axhline(y=line["y"], color=line.get("color", "gray"), ls=line.get("ls", "--"), label=line.get("label"))
    
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = (ax2.get_legend_handles_labels() if ax2 else ([],[]))
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

@torch.no_grad()
def generate_and_save_trajectories(policy_learn: nn.Module, policy_cf: Optional[nn.Module] = None, outdir: Optional[str] = None, seed_crn: Optional[int] = None) -> str:
    outdir = _ensure_outdir(outdir); B_all = int(PGDPO_TRAJ_B or N_eval_states); seed_crn = int(seed_crn or CRN_SEED_EU)
    def _g(): return torch.Generator(device=device).manual_seed(seed_crn)
    
    print("[traj] Simulating trajectories for 'learn' policy...")
    traj_learn = _simulate_and_record(policy_learn.to(device), B_all, _g(), sync_time=True)
    
    traj_cf = _simulate_and_record(policy_cf.to(device), B_all, _g(), sync_time=True) if policy_cf else None
    
    traj_pp = None
    if PGDPO_CURRENT_MODE in ["run", "projection", "residual"]:
        if PP_MODE == 'direct' and _HAS_DIRECT:
            print("[traj] Generating 'pp' trajectories using DIRECT method.")
            traj_pp = _simulate_and_record(PPDirectPolicy(policy_learn.to(device), seed_crn), B_all, _g(), sync_time=True)
        elif _HAS_RUNNER:
            print("[traj] Generating 'pp' trajectories using RUNNER method.")
            traj_pp = _simulate_and_record(PPRUNNERPolicy(policy_learn.to(device), seed_crn), B_all, _g(), sync_time=True)

    trajectories = {"learn": traj_learn, "cf": traj_cf, "pp": traj_pp}
    Bsel = list(range(min(B_all, SCHEMA.get("sampling", {}).get("Bmax", 5))))
    
    # --- 기존 시각화 및 요약 CSV 저장 로직 ---
    all_views_data = {}
    master_x_time = None

    for view in SCHEMA.get("views", []):
        series_map = {}
        x_time = None
        for name, traj in trajectories.items():
            if traj is None: continue
            
            curr_x, series, labels = _series_for_view_wrapper(view, traj, Bsel, policy_tag=name)
            if curr_x is None or not series: continue
            if x_time is None: 
                x_time = curr_x
                if master_x_time is None:
                    master_x_time = x_time
            
            for s, lab in zip(series, labels):
                if lab not in series_map: series_map[lab] = {}
                series_map[lab][name] = s
        
        if x_time is not None and series_map:
            _plot_lines(x_time, series_map, f"Trajectory Comparison: {view['name']}", 
                        view.get("ylabel","Value"), os.path.join(outdir, f"traj_{view['name']}_comparison.png"),
                        view_opts=view)
            all_views_data[view['name']] = series_map
                        
    if master_x_time is not None and all_views_data:
        _save_series_to_csv(master_x_time, all_views_data, os.path.join(outdir, "traj_comparison_data.csv"))
        
    # --- (신규 추가) 모든 데이터를 저장하는 로직 호출 ---
    _save_full_trajectories_to_csv(
        trajectories=trajectories,
        schema=SCHEMA,
        Bsel=Bsel,
        out_path=os.path.join(outdir, "traj_full_data.csv")
    )
        
    return outdir