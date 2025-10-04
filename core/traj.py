# -*- coding: utf-8 -*-
# core/traj.py — schema-driven, multi-domain trajectory visualization
# - cf/pp/learn 궤적을 같은 그래프에 통합 출력
# - 시각화에 사용된 모든 궤적 데이터를 CSV로 저장
# - SIR 전용 커스텀 플롯 지원
# - ✅ (업데이트) CSV에는 선택된 모든 path를 각각 기록하고, 플롯은 첫 번째 path만 사용(토글 가능)
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
    sample_initial_states, simulate, make_generator,
)

# ----------------------------
# ENV 토글 (플롯/저장 동작 제어)
# ----------------------------
PGDPO_TRAJ_B   = int(os.getenv("PGDPO_TRAJ_B", 1))
PGDPO_CURRENT_MODE = os.getenv("PGDPO_CURRENT_MODE", "unknown")
PP_MODE = os.getenv('PP_MODE','direct').lower()

PGDPO_TRAJ_BMAX       = int(os.getenv("PGDPO_TRAJ_BMAX", "5"))
PGDPO_TRAJ_SAVE_ALL   = int(os.getenv("PGDPO_TRAJ_SAVE_ALL", "1"))
PGDPO_TRAJ_PLOT_FIRST = int(os.getenv("PGDPO_TRAJ_PLOT_FIRST", "1"))

_HAS_RUNNER, _HAS_DIRECT = False, False
try:
    from pgdpo_run import ppgdpo_u_run, REPEATS as REPEATS_RUN, SUBBATCH as SUBBATCH_RUN; _HAS_RUNNER = True
except ImportError:
    pass
try:
    from pgdpo_with_projection import (
        ppgdpo_u_direct, REPEATS as REPEATS_DIR, SUBBATCH as SUBBATCH_DIR,
        estimate_costates, PP_NEEDS
    ); _HAS_DIRECT = True
except ImportError:
    estimate_costates, PP_NEEDS = None, ()

try: from user_pgdpo_base import ref_signals_fn as _REF_FN
except Exception: _REF_FN = None
try: from user_pgdpo_base import R_INFO as _R_INFO
except Exception: _R_INFO = {}
try: from user_pgdpo_base import price as harvesting_income
except ImportError: harvesting_income = None
try: from user_pgdpo_base import calculate_rt
except (ImportError, AttributeError): calculate_rt = None

# ----------------------------
# Schema
# ----------------------------
def _try_load_user_schema() -> Optional[Dict[str, Any]]:
    try:
        from user_pgdpo_base import get_traj_schema
        return get_traj_schema()
    except (ImportError, AttributeError):
        pass
    return None

def _get_base_default_views() -> List[Dict[str, Any]]:
    views: List[Dict[str, Any]] = []
    if DIM_X and int(DIM_X) > 0:
        views.append({"name": "X_first_components", "block": "X",
                      "mode": "indices", "indices": [0], "ylabel": "X[0] Component"})
    if DIM_U and int(DIM_U) > 0:
        views.append({"name": "U_first_components", "block": "U",
                      "mode": "indices", "indices": [0], "ylabel": "U[0] Component"})
    return views

def _load_and_merge_schemas() -> Dict[str, Any]:
    base_views = _get_base_default_views()
    user_schema = _try_load_user_schema()
    if user_schema:
        if "views" not in user_schema:
            user_schema["views"] = []
        existing = {v.get("name") for v in user_schema["views"]}
        for v in base_views:
            if v["name"] not in existing:
                user_schema["views"].append(v)
        return user_schema
    else:
        return {
            "roles": {
                "X": {"dim": int(DIM_X or 0), "labels": [f"X_{i+1}" for i in range(int(DIM_X or 0))]},
                "U": {"dim": int(DIM_U or 0), "labels": [f"u_{i+1}" for i in range(int(DIM_U or 0))]},
            },
            "views": base_views,
            "sampling": {"Bmax": 5}
        }

SCHEMA = _load_and_merge_schemas()

# ----------------------------
# Helpers
# ----------------------------
def _ensure_outdir(outdir: Optional[str]) -> str:
    if outdir is None:
        outdir = os.path.join("plots", "_traj", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def _linspace_time(n_steps: int) -> np.ndarray:
    return np.linspace(0.0, float(T), num=n_steps)

def _as_ts1d(y: np.ndarray | list | float, T: int) -> np.ndarray:
    a = np.asarray(y)
    if a.ndim == 0:
        return np.full(T, float(a))
    a = np.squeeze(a)
    if a.ndim == 1:
        if a.shape[0] == T:
            return a.astype(float)
        if a.shape[0] == 1:
            return np.full(T, float(a[0]))
        xp = np.linspace(0.0, 1.0, num=a.shape[0])
        x  = np.linspace(0.0, 1.0, num=T)
        return np.interp(x, xp, a.astype(float))
    for ax in range(a.ndim):
        if a.shape[ax] == T:
            sl = [0]*a.ndim
            sl[ax] = slice(None)
            return a[tuple(sl)].reshape(T).astype(float)
    flat = a.reshape(-1).astype(float)
    xp = np.linspace(0.0, 1.0, num=flat.shape[0])
    x  = np.linspace(0.0, 1.0, num=T)
    return np.interp(x, xp, flat)

# ----------------------------
# P-PGDPO 래퍼(플롯/기록용)
# ----------------------------
class PPDirectPolicy(nn.Module):
    def __init__(self, stage1_policy: nn.Module, seed: int):
        super().__init__()
        self.stage1_policy = stage1_policy
        self.seed = seed
    
    def forward(self, **states_dict):
        if not _HAS_DIRECT:
            raise RuntimeError("'direct' utilities not available.")
        u_proj = ppgdpo_u_direct(
            self.stage1_policy, states_dict,
            repeats=REPEATS_DIR, sub_batch=SUBBATCH_DIR, seed_eval=self.seed
        )
        with torch.no_grad():
            out0 = self.stage1_policy(**states_dict)
        is_consumption = out0.size(1) > u_proj.size(1)
        if is_consumption:
            c_learned = out0[:, u_proj.size(1):]
            return torch.cat([u_proj, c_learned], dim=1)
        return u_proj

class PPRUNNERPolicy(nn.Module):
    def __init__(self, stage1_policy: nn.Module, seed: int):
        super().__init__()
        self.stage1_policy = stage1_policy
        self.seed = seed
    
    def forward(self, **states_dict):
        if not _HAS_RUNNER:
            raise RuntimeError("'runner' utilities not available.")
        u_proj = ppgdpo_u_run(self.stage1_policy, states_dict, REPEATS_RUN, SUBBATCH_RUN, self.seed)
        with torch.no_grad():
            out0 = self.stage1_policy(**states_dict)
        is_consumption = out0.size(1) > u_proj.size(1)
        if is_consumption:
            c_learned = out0[:, u_proj.size(1):]
            return torch.cat([u_proj, c_learned], dim=1)
        return u_proj

# ----------------------------
# 기록용 래퍼
# ----------------------------
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
            if not hasattr(self, "Y_frames"):
                self.Y_frames = []
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
                        costates = estimate_costates(
                            simulate, self.base_policy,
                            {k: v.detach().clone().requires_grad_(True)
                             for k, v in states_dict.items() if k in ["X", "Y"]},
                            repeats=16, sub_batch=16, seed_eval=123, needs=PP_NEEDS
                        )
                    JX = costates.get("JX")
                    if JX is not None:
                        pI = JX[:, 1::3].sum(dim=1, keepdim=True)
                        self.Costate_I_frames.append(pI.detach())
                except Exception:
                    pass

        return U

    def stacked(self) -> Dict[str, torch.Tensor]:
        data = {"X": torch.stack(self.X_frames, 1),
                "U": torch.stack(self.U_frames, 1)}
        if hasattr(self, "Y_frames"):
            data["Y"] = torch.stack(self.Y_frames, 1)
        if self.HarvestingIncome_frames:
            data["HarvestingIncome"] = torch.stack(self.HarvestingIncome_frames, 1)
        if self.Rt_frames:
            data["Rt"] = torch.stack(self.Rt_frames, 1)
        if self.Costate_I_frames:
            data["Costate_I"] = torch.stack(self.Costate_I_frames, 1)
        return data

def _simulate_and_record(policy: nn.Module, B: int, rng: torch.Generator, sync_time: bool = False) -> Dict[str, torch.Tensor]:
    recorder = RecordingPolicy(policy).to(device)
    init, _ = sample_initial_states(B, rng=rng)
    if sync_time:
        init["TmT"].fill_(T)
    simulate(recorder, B, initial_states_dict=init, m_steps=m, train=False, rng=rng)
    return recorder.stacked()

# ... (이하 _series_for_view_wrapper, _save_series_to_csv, _save_full_trajectories_to_csv, _plot_lines 함수들은 그대로 유지)
def _series_for_view(
    view: Dict[str, Any],
    traj_np: Dict[str, Any],
    Bsel: List[int],
) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    block_names = view.get("block", "X")
    if isinstance(block_names, str):
        block_names = [block_names]

    x_time: Optional[np.ndarray] = None
    all_series: List[np.ndarray] = []
    all_labels: List[str] = []

    for block_name in block_names:
        if block_name not in traj_np:
            continue

        arr = _to_np(traj_np.get(block_name))
        if arr.ndim == 2:
            arr = arr[..., None]

        arr_sel = arr[Bsel]
        steps = arr.shape[1]
        if x_time is None:
            x_time = _linspace_time(steps)

        role_info   = SCHEMA.get("roles", {}).get(block_name, {})
        labels_info = role_info.get("labels", [])

        indices = view.get("indices", [0])
        valid_indices = [i for i in indices if 0 <= i < arr_sel.shape[-1]]
        if not valid_indices:
            continue

        if PGDPO_TRAJ_PLOT_FIRST:
            data = arr_sel[0, :, valid_indices]
        else:
            data = arr_sel[..., valid_indices].mean(axis=0)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        series = [data[:, i] for i in range(data.shape[1])]
        labels = (
            [labels_info[idx] for idx in valid_indices]
            if labels_info else [f"dim_{idx}" for idx in valid_indices]
        )

        all_series.extend(series)
        all_labels.extend(labels)

    if x_time is None:
        x_time = _linspace_time(m)

    return x_time, all_series, all_labels


def _handle_custom_view(
    view: dict,
    traj_np: dict,
    Bsel: list,
    policy_tag: str | None = None
) -> Optional[Tuple[np.ndarray, List[np.ndarray], List[str]]]:
    mode = (view.get("mode", "") or "").lower()
    name = (view.get("name", "") or "").lower()

    if mode == "u_rnorm":
        U = _to_np(traj_np["U"])[Bsel, :, :d]
        x_time = _linspace_time(U.shape[1])
        if "R" in _R_INFO:
            R = _R_INFO["R"].cpu().numpy()
            uRu = np.einsum("bti,ij,btj->bt", U, R, U)
            y = uRu[0] if PGDPO_TRAJ_PLOT_FIRST else uRu.mean(axis=0)
            r_norm_series = np.sqrt(_as_ts1d(y, len(x_time)))
        else:
            alpha = _R_INFO.get("alpha", 0.0)
            if alpha <= 0:
                r_norm_series = np.zeros(U.shape[1])
            else:
                y = (
                    np.linalg.norm(U[0], axis=-1)
                    if PGDPO_TRAJ_PLOT_FIRST else
                    np.linalg.norm(U, axis=-1).mean(axis=0)
                )
                r_norm_series = np.sqrt(alpha) * _as_ts1d(y, len(x_time))
        return x_time, [r_norm_series], [view.get("block", "U")]

    if mode == "tracking_vpp":
        U = _to_np(traj_np["U"])[Bsel]
        x_time = _linspace_time(U.shape[1])
        y = U[0].sum(axis=-1) if PGDPO_TRAJ_PLOT_FIRST else U.sum(axis=-1).mean(axis=0)
        series, labels = [_as_ts1d(y, len(x_time))], ["Output_Power_Sum"]
        if _REF_FN:
            ref = _REF_FN(x_time)
            for key, vals in ref.items():
                series.append(_as_ts1d(vals, len(x_time)))
                labels.append(f"{key}_ref")
        return x_time, series, labels

    if mode == "custom_sir_rt_plot":
        Rt = _to_np(traj_np.get("Rt"))[Bsel]
        x_time = _linspace_time(Rt.shape[1])
        y_rt = Rt[0].flatten() if PGDPO_TRAJ_PLOT_FIRST else Rt.mean(axis=0).flatten()

        X = _to_np(traj_np.get("X"))[Bsel]
        i_idx = [i for i in range(1, X.shape[-1], 3)]
        y_i = (
            X[0, :, i_idx].sum(axis=-1).flatten()
            if PGDPO_TRAJ_PLOT_FIRST else
            X[..., i_idx].sum(axis=-1).mean(axis=0).flatten()
        )
        return x_time, [_as_ts1d(y_rt, len(x_time)), _as_ts1d(y_i, len(x_time))], ["Effective_Rt", "Total_Infected"]

    if mode == "custom_sir_i_plot":
        X = _to_np(traj_np.get("X"))[Bsel]
        x_time = _linspace_time(X.shape[1])
        i_idx = [i for i in range(1, X.shape[-1], 3)]
        y = (
            X[0, :, i_idx].sum(axis=-1).flatten()
            if PGDPO_TRAJ_PLOT_FIRST else
            X[..., i_idx].sum(axis=-1).mean(axis=0).flatten()
        )
        return x_time, [_as_ts1d(y, len(x_time))], ["Total_Infected"]

    if name == "consumption_path" or mode == "consumption":
        C = None
        if "C" in traj_np:
            C = _to_np(traj_np["C"])[Bsel]
            if C.ndim == 2:
                C = C[..., None]
        elif "U" in traj_np:
            U = _to_np(traj_np["U"])[Bsel]
            if U.ndim == 2:
                U = U[..., None]
            if U.shape[2] > d:
                C = U[..., -1:]

        if C is None:
            return None

        x_time = _linspace_time(C.shape[1])
        y = C[0, :, 0] if PGDPO_TRAJ_PLOT_FIRST else C[..., 0].mean(axis=0)
        return x_time, [_as_ts1d(y, len(x_time))], ["Consumption"]

    return None

def _series_for_view_wrapper(view, traj_np, Bsel, policy_tag=None):
    if (view.get("name","").lower().startswith("tracking")):
        view = {**view, "mode": "tracking_vpp"}
    custom_result = _handle_custom_view(view, traj_np, Bsel, policy_tag)
    if custom_result is not None:
        return custom_result
    return _series_for_view(view, traj_np, Bsel)

def _save_series_to_csv(x_time, all_views_data, out_path):
    x_time = np.asarray(x_time, dtype=float).reshape(-1)
    Tlen = int(x_time.shape[0])

    colmap: Dict[str, np.ndarray] = {}
    for view_name, series_map in all_views_data.items():
        for component, policies in sorted(series_map.items()):
            for policy, data in sorted(policies.items()):
                col_name = component if ("ref" in component) else f"{component}_{policy}"
                y = _as_ts1d(data, Tlen)
                if col_name not in colmap:
                    colmap[col_name] = y

    header = ["Time"] + list(colmap.keys())
    cols = [x_time] + [colmap[h] for h in header[1:]]

    rows = np.stack(cols, axis=-1)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
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
    if not trajectories:
        return

    first_valid_traj = next((t for t in trajectories.values() if t is not None), None)
    if first_valid_traj is None:
        return
    n_steps = first_valid_traj["X"].shape[1]
    x_time = _linspace_time(n_steps)

    header = ["Time"]
    all_data_map: Dict[str, np.ndarray] = {}

    roles = schema.get("roles", {})

    for policy_name, traj_data in trajectories.items():
        if traj_data is None:
            continue
        for block_name, block_tensor in traj_data.items():
            if block_name not in roles:
                continue

            arr_sel = _to_np(block_tensor[Bsel])
            if arr_sel.ndim == 2:
                arr_sel = arr_sel[..., None]
            labels = roles[block_name].get(
                "labels",
                [f"{block_name}_{i}" for i in range(arr_sel.shape[-1])]
            )

            if PGDPO_TRAJ_SAVE_ALL:
                for b_idx, b in enumerate(Bsel):
                    for i in range(arr_sel.shape[-1]):
                        col = f"{labels[i]}_{policy_name}_p{b}"
                        if col not in header:
                            header.append(col)
                        all_data_map[col] = arr_sel[b_idx, :, i]
            else:
                block_np_avg = arr_sel.mean(axis=0)
                for i in range(block_np_avg.shape[1]):
                    col = f"{labels[i]}_{policy_name}"
                    if col not in header:
                        header.append(col)
                    all_data_map[col] = block_np_avg[:, i]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t_idx in range(n_steps):
            row = [x_time[t_idx]]
            for col in header[1:]:
                row.append(all_data_map.get(col, np.array([]))[t_idx])
            writer.writerow(row)

    print(f"[traj] Full trajectory data saved to: {os.path.basename(out_path)}")

def _plot_lines(x_time, series_map, title, ylabel, save_path, view_opts: dict = {}):
    is_dual_axis = view_opts.get("mode") == "dual_axis"

    POLICY_STYLES = {
        "cf":    {"color": "royalblue",  "ls": "-",  "lw": 2.5, "zorder": 5, "label": "Ref/Myopic"},
        "pp":    {"color": "orangered",  "ls": "--", "lw": 1.8, "zorder": 4, "label": "P-PGDPO (pp)", "marker": "o", "ms": 4, "alpha": 0.7},
        "learn": {"color": "forestgreen","ls": ":",  "lw": 1.8, "zorder": 3, "label": "Learned", "marker": "s", "ms": 4, "alpha": 0.7},
        "ref":   {"color": "gray",       "ls": "-.", "lw": 1.5, "zorder": 2}
    }

    if "legend_labels" in SCHEMA:
        custom_labels = SCHEMA["legend_labels"]
        for key, new_label in custom_labels.items():
            if key in POLICY_STYLES:
                POLICY_STYLES[key]["label"] = new_label

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax2 = ax1.twinx() if is_dual_axis else None

    ax_map = {}
    if is_dual_axis:
        for comp in series_map.keys():
            if "costate" in comp.lower() or "shadow" in comp.lower():
                ax_map[comp] = ax2
            else:
                ax_map[comp] = ax1
    else:
        for comp in series_map.keys():
            ax_map[comp] = ax1

    Tlen = len(x_time)

    for comp_label, policies in sorted(series_map.items()):
        ax = ax_map.get(comp_label, ax1)
        if "ref" in comp_label:
            if "cf" in policies:
                y = _as_ts1d(policies["cf"], Tlen)
                ax.plot(x_time, y, label=POLICY_STYLES["ref"].get("label", comp_label),
                        **{k: v for k, v in POLICY_STYLES["ref"].items() if k != "label"})
            continue
        for pkey in ["cf", "pp", "learn"]:
            if pkey in policies:
                y = _as_ts1d(policies[pkey], Tlen)
                st = POLICY_STYLES.get(pkey, {})
                full_label = f"{comp_label} - {st.get('label')}"
                ax.plot(x_time, y, label=full_label,
                        **{k: v for k, v in st.items() if k != "label"})

    ax1.set(xlabel="Time", title=title)
    if is_dual_axis and "ylabels" in view_opts:
        ax1.set_ylabel(view_opts["ylabels"][0])
        if ax2:
            ax2.set_ylabel(view_opts["ylabels"][1])
    else:
        ax1.set_ylabel(ylabel)

    if "h_lines" in view_opts:
        for line in view_opts["h_lines"]:
            y = float(line["y"])
            ax1.axhline(y=y, color=line.get("color", "gray"),
                        ls=line.get("ls", "--"), label=line.get("label"))

    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = (ax2.get_legend_handles_labels() if ax2 else ([], []))
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ----------------------------
# 엔트리
# ----------------------------
@torch.no_grad()
def generate_and_save_trajectories(
    policy_learn: nn.Module,
    policy_cf: Optional[nn.Module] = None,
    outdir: Optional[str] = None,
    seed_crn: Optional[int] = None
) -> str:
    outdir = _ensure_outdir(outdir)
    B_all = int(PGDPO_TRAJ_B or N_eval_states)
    seed_crn = int(seed_crn or CRN_SEED_EU)
    def _g(): return make_generator(seed_crn)

    print("[traj] Simulating trajectories for 'learn' policy...")
    traj_learn = _simulate_and_record(policy_learn.to(device), B_all, _g(), sync_time=True)

    traj_cf = _simulate_and_record(policy_cf.to(device), B_all, _g(), sync_time=True) if policy_cf else None

    traj_pp = None
    if PGDPO_CURRENT_MODE in ["run", "projection", "residual"]:
        if PP_MODE == "direct" and _HAS_DIRECT:
            print("[traj] Generating 'pp' trajectories using DIRECT method.")
            traj_pp = _simulate_and_record(PPDirectPolicy(policy_learn.to(device), seed_crn), B_all, _g(), sync_time=True)
        elif _HAS_RUNNER:
            print("[traj] Generating 'pp' trajectories using RUNNER method.")
            traj_pp = _simulate_and_record(PPRUNNERPolicy(policy_learn.to(device), seed_crn), B_all, _g(), sync_time=True)

    trajectories = {"learn": traj_learn, "cf": traj_cf, "pp": traj_pp}

    schema_Bmax = SCHEMA.get("sampling", {}).get("Bmax", 5)
    Bmax = PGDPO_TRAJ_BMAX if PGDPO_TRAJ_BMAX > 0 else schema_Bmax
    Bsel = list(range(min(B_all, Bmax)))

    all_views_data = {}
    master_x_time = None

    for view in SCHEMA.get("views", []):
        series_map = {}
        x_time = None
        for name, traj in trajectories.items():
            if traj is None:
                continue
            curr_x, series, labels = _series_for_view_wrapper(view, traj, Bsel, policy_tag=name)
            if curr_x is None or not series:
                continue
            if x_time is None:
                x_time = curr_x
                if master_x_time is None:
                    master_x_time = x_time
            for s, lab in zip(series, labels):
                if lab not in series_map:
                    series_map[lab] = {}
                series_map[lab][name] = s

        if x_time is not None and series_map:
            _plot_lines(
                x_time, series_map,
                f"Trajectory Comparison: {view['name']}",
                view.get("ylabel", "Value"),
                os.path.join(outdir, f"traj_{view['name']}_comparison.png"),
                view_opts=view
            )
            all_views_data[view["name"]] = series_map

    if master_x_time is not None and all_views_data:
        _save_series_to_csv(master_x_time, all_views_data, os.path.join(outdir, "traj_comparison_data.csv"))

    _save_full_trajectories_to_csv(
        trajectories=trajectories,
        schema=SCHEMA,
        Bsel=Bsel,
        out_path=os.path.join(outdir, "traj_full_data.csv")
    )

    return outdir