# core/traj.py — schema-driven, multi-domain trajectory visualization
from __future__ import annotations
import os, json, time
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

#PP_MODE = os.getenv('PP_MODE','runner').lower()
PP_MODE = os.getenv('PP_MODE','direct').lower()
_HAS_RUNNER, _HAS_DIRECT = False, False
try: from pgdpo_run import ppgdpo_u_run, REPEATS as REPEATS_RUN, SUBBATCH as SUBBATCH_RUN; _HAS_RUNNER = True
except ImportError: pass
try: from pgdpo_with_projection import ppgdpo_u_direct, REPEATS as REPEATS_DIR, SUBBATCH as SUBBATCH_DIR; _HAS_DIRECT = True
except ImportError: pass

try: from user_pgdpo_base import ref_signals_fn as _REF_FN
except Exception: _REF_FN = None
try: from user_pgdpo_base import R_INFO as _R_INFO
except Exception: _R_INFO = {}

# --- Schema Logic ---
def _try_load_user_schema() -> Optional[Dict[str, Any]]:
    try: from user_pgdpo_base import get_traj_schema; return get_traj_schema()
    except (ImportError, AttributeError): pass
    return None

def _get_base_default_views() -> List[Dict[str, Any]]:
    views: List[Dict[str, Any]] = []
    if DIM_X and int(DIM_X) > 0:
        views.append({"name": "X_first_components", "block": "X", "mode": "indices", "indices": [0], "ylabel": "X[0] Component"})
    if DIM_U and int(DIM_U) > 0:
        views.append({"name": "U_first_components", "block": "U", "mode": "indices", "indices": [0], "ylabel": "U[0] Component"})
    return views

def _load_and_merge_schemas() -> Dict[str, Any]:
    base_views = _get_base_default_views()
    user_schema = _try_load_user_schema()
    if user_schema:
        if "views" not in user_schema: user_schema["views"] = []
        existing_view_names = {v.get("name") for v in user_schema["views"]}
        for view in base_views:
            if view["name"] not in existing_view_names:
                user_schema["views"].append(view)
        return user_schema
    else:
        return {
            "roles": {
                "X": {"dim": int(DIM_X or 0), "labels": [f"X_{i+1}" for i in range(int(DIM_X or 0))]},
                "U": {"dim": int(DIM_U or 0), "labels": [f"u_{i+1}" for i in range(int(DIM_U or 0))]},
            },
            "views": base_views, "sampling": {"Bmax": 5},
        }
SCHEMA = _load_and_merge_schemas()
# --- End of Schema Logic ---

def _ensure_outdir(outdir: Optional[str]) -> str:
    if outdir is None: outdir = os.path.join("plots", "_traj", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True); return outdir
def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
def _linspace_time(n_steps: int) -> np.ndarray:
    return np.linspace(0.0, float(T), num=n_steps)

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
        U = self.base_policy(**states_dict)
        self.U_frames.append(U.detach())
        return U
    def stacked(self) -> Dict[str, torch.Tensor]:
        return {"X": torch.stack(self.X_frames, 1), "U": torch.stack(self.U_frames, 1)}

def _simulate_and_record(policy: nn.Module, B: int, rng: torch.Generator, sync_time: bool = False) -> Dict[str, torch.Tensor]:
    recorder = RecordingPolicy(policy).to(device)
    init, _ = sample_initial_states(B, rng=rng)
    if sync_time:
        init['TmT'].fill_(T)
    simulate(recorder, B, initial_states_dict=init, m_steps=m, train=False, rng=rng)
    return recorder.stacked()

def _plot_lines(x_time, series_map, title, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    # 선/마커 스타일(겹침 방지)
    STYLE = {
        "cf":    {"ls": "-",  "marker": None, "lw": 2.4, "zorder": 3},
        "pp":    {"ls": "-.", "marker": "o",  "lw": 1.9, "ms": 3.0, "markevery": None, "zorder": 4},
        "learn": {"ls": "--", "marker": "s",  "lw": 1.9, "ms": 3.0, "markevery": None, "zorder": 4},
    }

    # 마커 밀도 자동 설정(대략 10개 마커)
    n = len(x_time)
    default_markevery = max(1, n // 10)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    used = set()

    for i, (label, policies) in enumerate(sorted(series_map.items())):
        color = colors[i % len(colors)]
        for policy, data in sorted(policies.items()):
            # 표시용 라벨/중복키를 먼저 만든다
            is_ref = ("N_agg" in label)
            disp_label = label if is_ref else f"{label} — {policy}"
            key = label if is_ref else disp_label
            if key in used:
                continue

            # 스타일 선택
            st = STYLE.get("cf" if is_ref else policy, {"ls": "-", "lw": 2.0})
            ls = st.get("ls", "-")
            marker = st.get("marker", None)
            lw = st.get("lw", 2.0)
            z = st.get("zorder", 2)
            ms = st.get("ms", None)
            me = st.get("markevery", default_markevery if marker else None)

            ax.plot(
                x_time, data,
                linestyle=ls, color=color, label=disp_label, lw=lw, zorder=z,
                marker=marker, ms=ms, markevery=me
            )
            used.add(key)

    ax.set(xlabel="Time", ylabel=ylabel, title=title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def _series_for_view(view: Dict[str, Any], traj_np: Dict[str, Any], Bsel: List[int]) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    block_name = view.get("block", "X")
    arr = _to_np(traj_np.get(block_name))
    arr_sel, steps = arr[Bsel], arr.shape[1]; x_time = _linspace_time(steps)
    labels_info = SCHEMA.get("roles", {}).get(block_name, {}).get("labels", [])
    indices = view.get("indices", [0])
    valid_indices = [i for i in indices if i < arr_sel.shape[-1]]
    mean_series = arr_sel[..., valid_indices].mean(axis=0)
    series = [mean_series[:, i] for i in range(mean_series.shape[1])]
    labels = [labels_info[idx] for idx in valid_indices]
    return x_time, series, labels

def _handle_custom_view(view: dict, traj_np: dict, Bsel: list, policy_tag: str | None = None) -> Optional[Tuple[np.ndarray, List[np.ndarray], List[str]]]:
    mode = (view.get("mode","") or "").lower()
    
    # --- "u_rnorm" 처리 로직 ---
    if mode == "u_rnorm":
        # traj_np 딕셔너리에서 'U' 블록의 데이터를 가져옴
        U = _to_np(traj_np["U"])[Bsel]  # (B, T, d) 형태의 numpy 배열
        x_time = _linspace_time(U.shape[1])
        
        # user_pgdpo_base.py의 _R_INFO 에서 alpha 값을 가져옴
        alpha = _R_INFO.get("alpha", 0.0)
        if alpha <= 0:
            print("[WARN] Alpha for R-norm is not positive. Norm will be zero.")
            # alpha가 0이거나 음수이면 norm은 0
            r_norm_series = np.zeros(U.shape[1])
        else:
            # ||u||_R = sqrt(alpha * sum(u_i^2)) = sqrt(alpha) * L2_norm(u)
            # 1. 각 시간 스텝(axis=1)별로 d차원 u 벡터의 L2 norm을 계산 (axis=-1)
            # 2. 배치(B)에 대해 평균을 냄 (axis=0)
            r_norm_series = np.sqrt(alpha) * np.linalg.norm(U, axis=-1).mean(axis=0)
        
        # [시간축, [계산된 시리즈], [범례 레이블]] 형태로 반환
        return x_time, [r_norm_series], [view.get("block", "U")]
    # --- 로직 끝 ---

    # --- 기존 "tracking_vpp" 처리 로직 ---
    if mode == "tracking_vpp":
        U = _to_np(traj_np["U"])[Bsel]
        x_time = _linspace_time(U.shape[1])
        u_total_power = U.sum(axis=-1).mean(axis=0)
        series, labels = [u_total_power], ["Output (sum u)"]
        if _REF_FN and policy_tag == "cf":
            # N_agg 레퍼런스 신호가 있으면 함께 그림
            n_agg_line = np.asarray(_REF_FN(x_time)["Nagg"]).reshape(-1)
            series.append(n_agg_line)
            labels.append("N_agg (ref)")
        return x_time, series, labels

    # 처리할 모드가 없으면 None 반환
    return None

def _series_for_view_wrapper(view, traj_np, Bsel, policy_tag=None):
    if (view.get("name","").lower().startswith("tracking")):
        view = {**view, "mode": "tracking_vpp"}
    custom_result = _handle_custom_view(view, traj_np, Bsel, policy_tag)
    return custom_result if custom_result is not None else _series_for_view(view, traj_np, Bsel)

@torch.no_grad()
def generate_and_save_trajectories(policy_learn: nn.Module, policy_cf: Optional[nn.Module] = None, outdir: Optional[str] = None, B: Optional[int] = None, seed_crn: Optional[int] = None) -> str:
    outdir = _ensure_outdir(outdir); B_all = int(B or N_eval_states); seed_crn = int(seed_crn or CRN_SEED_EU)
    
    def _g(): return torch.Generator(device=device).manual_seed(seed_crn)

    traj_learn = _simulate_and_record(policy_learn.to(device), B_all, _g(), sync_time=True)
    traj_cf = _simulate_and_record(policy_cf.to(device), B_all, _g(), sync_time=True) if policy_cf else None
    
    traj_pp = None
    if policy_cf:
        if PP_MODE == 'direct' and _HAS_DIRECT:
            traj_pp = _simulate_and_record(PPDirectPolicy(policy_learn.to(device), seed_crn), B_all, _g(), sync_time=True)
        elif _HAS_RUNNER:
            traj_pp = _simulate_and_record(PPRUNNERPolicy(policy_learn.to(device), seed_crn), B_all, _g(), sync_time=True)

    trajectories = {"learn": traj_learn, "cf": traj_cf, "pp": traj_pp}
    Bsel = list(range(min(B_all, SCHEMA.get("sampling", {}).get("Bmax", 5))))
    
    for view in SCHEMA.get("views", []):
        series_map_learn_cf, series_map_pp_cf = {}, {}
        x_time, n_line_ref = None, None
        
        for name, traj in trajectories.items():
            if traj is None: continue
            curr_x, series, labels = _series_for_view_wrapper(view, traj, Bsel, policy_tag=name)
            if x_time is None: x_time = curr_x
            
            for s, lab in zip(series, labels):
                if "N_agg (ref)" in lab: n_line_ref = s; continue
                if name in ['learn', 'cf']: series_map_learn_cf.setdefault(lab, {})[name] = s
                if name in ['pp', 'cf']: series_map_pp_cf.setdefault(lab, {})[name] = s
        
        if n_line_ref is not None:
            series_map_learn_cf["N_agg (ref)"] = {"cf": n_line_ref}
            series_map_pp_cf["N_agg (ref)"] = {"cf": n_line_ref}

        if x_time is not None:
            _plot_lines(x_time, series_map_learn_cf, f"{view['name']}: Learn vs CF", view.get("ylabel",""), os.path.join(outdir, f"{view['name']}__learn_vs_cf.png"))
            _plot_lines(x_time, series_map_pp_cf, f"{view['name']}: PP vs CF", view.get("ylabel",""), os.path.join(outdir, f"{view['name']}__pp_vs_cf.png"))
            
    return outdir