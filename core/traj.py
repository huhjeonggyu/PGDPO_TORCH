# -*- coding: utf-8 -*-
# core/traj.py — Fast JEDC-style trajectory visualization with PP overlay
#  - Robust series extraction (no fancy-index pitfalls)
#  - One-shot simulation per policy (cache) → big speed-up for --projection
#  - PP (Stage-2 projection) overlay via PPDirectPolicy / PPRUNNERPolicy
#  - Custom views:
#      * Consumption_Stairs: C and C̄ (stairs)
#      * Consumption_Amount: C·dt
#      * Consumption_Cumulative: ∑ C·dt
#      * Wealth_with_thresholds: X with b(t,C̄) and BW boundary X̄
#      * Risky_Share: π/X (preview = first asset share)
#      * Discretionary_Wealth: DW = X - b(t,C̄)
#      * X_log_returns: Δlog X
from __future__ import annotations

import os, time, csv, math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Imports from the training/runtime core
# ---------------------------------------------------------------------
from pgdpo_base import (
    device, T, m, d, k, DIM_X, DIM_U, CRN_SEED_EU, N_eval_states,
    sample_initial_states, simulate, make_generator,
)

# Optional helpers exposed by the user's module
try:
    from user_pgdpo_base import ref_signals_fn as _REF_FN     # noqa: F401
except Exception:
    _REF_FN = None

try:
    from user_pgdpo_base import R_INFO as _R_INFO             # may contain theta_eff, etc.
except Exception:
    _R_INFO = {}

# ---------------------------------------------------------------------
# Env toggles
# ---------------------------------------------------------------------
PGDPO_CURRENT_MODE     = os.getenv("PGDPO_CURRENT_MODE", "unknown").lower()
PGDPO_TRAJ_B           = int(os.getenv("PGDPO_TRAJ_B", "1"))
PGDPO_TRAJ_BMAX        = int(os.getenv("PGDPO_TRAJ_BMAX", "5"))
PGDPO_TRAJ_SAVE_ALL    = int(os.getenv("PGDPO_TRAJ_SAVE_ALL", "1"))
PGDPO_TRAJ_PLOT_FIRST  = int(os.getenv("PGDPO_TRAJ_PLOT_FIRST", "1"))

# Speed/UX knobs
PGDPO_TRAJ_CACHE       = int(os.getenv("PGDPO_TRAJ_CACHE", "1"))     # cache trajectories once/policy
PGDPO_TRAJ_DECIMATE    = int(os.getenv("PGDPO_TRAJ_DECIMATE", "1"))  # keep every k-th step
PGDPO_TRAJ_NO_CSV      = int(os.getenv("PGDPO_TRAJ_NO_CSV", "0"))    # skip CSV
PGDPO_TRAJ_ONLY        = os.getenv("PGDPO_TRAJ_ONLY", "")            # comma-separated view names
PGDPO_TRAJ_SKIP        = os.getenv("PGDPO_TRAJ_SKIP", "")            # comma-separated view names

# PP overlay
PGDPO_TRAJ_PLOT_PP     = int(os.getenv("PGDPO_TRAJ_PLOT_PP", "1"))
PGDPO_TRAJ_FORCE_PP    = int(os.getenv("PGDPO_TRAJ_FORCE_PP", "0"))
PP_MODE                = os.getenv("PP_MODE", "direct").lower()
PP_REPEATS_OVERRIDE    = os.getenv("PGDPO_PP_REPEATS", "")            # e.g., "4"
PP_SUBBATCH_OVERRIDE   = os.getenv("PGDPO_PP_SUBBATCH", "")           # e.g., "32"

# ---------------------------------------------------------------------
# Optional Stage-2 projection wrappers (pp)
# ---------------------------------------------------------------------
_HAS_RUNNER, _HAS_DIRECT = False, False
_HAS_UC_DIRECT, _HAS_UC_RUN = False, False

# Optional RUNNER interface (if available in your codebase)
try:
    from pgdpo_run import ppgdpo_u_run, REPEATS as REPEATS_RUN, SUBBATCH as SUBBATCH_RUN
    _HAS_RUNNER = True
    try:
        from pgdpo_run import ppgdpo_uc_run as _PP_UC_RUN
        _HAS_UC_RUN = True
    except Exception:
        _HAS_UC_RUN = False
except Exception:
    pass

# DIRECT interface is mandatory for pp overlay (we added UC-direct)
try:
    from pgdpo_with_projection import (
        ppgdpo_u_direct, REPEATS as REPEATS_DIR, SUBBATCH as SUBBATCH_DIR
    )
    _HAS_DIRECT = True
    try:
        from pgdpo_with_projection import ppgdpo_uc_direct as _PP_UC_DIRECT
        _HAS_UC_DIRECT = True
    except Exception:
        _HAS_UC_DIRECT = False
except Exception:
    pass


def _pp_rs(default_r: int, default_s: int) -> Tuple[int, int]:
    r = os.getenv("PGDPO_PP_REPEATS", "")
    s = os.getenv("PGDPO_PP_SUBBATCH", "")
    rr = int(r) if r.isdigit() else default_r
    ss = int(s) if s.isdigit() else default_s
    return max(1, rr), max(1, ss)


class PPDirectPolicy(nn.Module):
    """pp를 policy처럼 호출. 가능하면 (u,C) 풀벡터 API 사용."""
    def __init__(self, stage1_policy: nn.Module, seed: int):
        super().__init__()
        self.stage1_policy = stage1_policy
        self.seed = seed
        self.rpt, self.sub = _pp_rs(REPEATS_DIR if _HAS_DIRECT else 1,
                                    SUBBATCH_DIR if _HAS_DIRECT else 1)

    def forward(self, **states_dict):
        if _HAS_UC_DIRECT:
            # (u_pp, C_pp) 한 번에
            return _PP_UC_DIRECT(self.stage1_policy, states_dict,
                                 repeats=self.rpt, sub_batch=self.sub, seed_eval=self.seed)
        # 폴백: u만 pp, C는 Stage-1
        if not _HAS_DIRECT:
            raise RuntimeError("ppgdpo_u_direct unavailable and no UC-direct fallback.")
        u_proj = ppgdpo_u_direct(self.stage1_policy, states_dict,
                                 repeats=self.rpt, sub_batch=self.sub, seed_eval=self.seed)
        with torch.no_grad():
            out0 = self.stage1_policy(**states_dict)
        return torch.cat([u_proj, out0[:, u_proj.size(1):]], dim=1) if out0.size(1) > u_proj.size(1) else u_proj


class PPRUNNERPolicy(nn.Module):
    def __init__(self, stage1_policy: nn.Module, seed: int):
        super().__init__()
        self.stage1_policy = stage1_policy
        self.seed = seed
        self.rpt, self.sub = _pp_rs(REPEATS_RUN if _HAS_RUNNER else 1,
                                    SUBBATCH_RUN if _HAS_RUNNER else 1)

    def forward(self, **states_dict):
        if _HAS_UC_RUN:
            return _PP_UC_RUN(self.stage1_policy, states_dict,
                              repeats=self.rpt, sub_batch=self.sub, seed_eval=self.seed)
        if not _HAS_RUNNER:
            raise RuntimeError("ppgdpo_u_run unavailable and no UC-run fallback.")
        u_proj = ppgdpo_u_run(self.stage1_policy, states_dict, self.rpt, self.sub, self.seed)
        with torch.no_grad():
            out0 = self.stage1_policy(**states_dict)
        return torch.cat([u_proj, out0[:, u_proj.size(1):]], dim=1) if out0.size(1) > u_proj.size(1) else u_proj


# ---------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------
def _try_load_user_schema() -> Optional[Dict[str, Any]]:
    try:
        from user_pgdpo_base import get_traj_schema
        return get_traj_schema()
    except Exception:
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
        user_schema.setdefault("views", [])
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

# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def _ensure_outdir(outdir: Optional[str]) -> str:
    if outdir is None:
        outdir = os.path.join("plots", "_traj", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _linspace_time(n_steps: int) -> np.ndarray:
    return np.linspace(0.0, float(T), num=n_steps)


def _as_ts1d(y: np.ndarray | list | float, Tlen: int) -> np.ndarray:
    """Return a 1D array of length Tlen (robust up/down sampling)."""
    a = np.asarray(y)
    if a.ndim == 0:
        return np.full(Tlen, float(a))
    a = np.squeeze(a)
    if a.ndim == 1:
        if a.shape[0] == Tlen:
            return a.astype(float)
        if a.shape[0] == 1:
            return np.full(Tlen, float(a[0]))
        xp = np.linspace(0.0, 1.0, num=a.shape[0]); x = np.linspace(0.0, 1.0, num=Tlen)
        return np.interp(x, xp, a.astype(float))
    for ax in range(a.ndim):
        if a.shape[ax] == Tlen:
            sl = [0]*a.ndim; sl[ax] = slice(None)
            return a[tuple(sl)].reshape(Tlen).astype(float)
    flat = a.reshape(-1).astype(float)
    xp = np.linspace(0.0, 1.0, num=flat.shape[0]); x = np.linspace(0.0, 1.0, num=Tlen)
    return np.interp(x, xp, flat)


def _stairs_1d(x: np.ndarray) -> np.ndarray:
    """Monotone non-decreasing staircase (cumulative max)."""
    return np.maximum.accumulate(np.asarray(x).reshape(-1))


def _bw_rstar_tau(theta: float, rho: float, r: float, tau: float) -> float:
    """
    Barone–Adesi–Whaley style boundary factor:
    0.5 θ^2 n^2 + (ρ - r - 0.5 θ^2) n - ρ (1 - e^{-ρ τ}) = 0 ;  R*_τ = 1/(1 - n_-)
    """
    if tau <= 0:
        return 0.0
    a = 0.5 * theta**2
    b = (rho - r - 0.5 * theta**2)
    c = - rho * (1.0 - math.exp(-rho * tau))
    disc = max(b*b - 4*a*c, 0.0)
    if a == 0.0:
        return 0.0
    n_minus = (-b - math.sqrt(disc)) / (2*a)
    return 1.0 / max(1.0 - n_minus, 1e-8)


def _decimate_traj(tr: Dict[str, torch.Tensor], every: int) -> Dict[str, torch.Tensor]:
    if every <= 1:
        return tr
    out = {}
    for k, v in tr.items():
        if v.ndim >= 2:
            out[k] = v[:, ::every, ...]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------
# Recording policy (captures X, U along the trajectory)
# ---------------------------------------------------------------------
class RecordingPolicy(nn.Module):
    def __init__(self, base_policy: nn.Module):
        super().__init__()
        self.base_policy = base_policy
        self.X_frames: List[torch.Tensor] = []
        self.U_frames: List[torch.Tensor] = []

    def forward(self, **states_dict):
        self.X_frames.append(states_dict["X"].detach().clone())
        U = self.base_policy(**states_dict)
        self.U_frames.append(U.detach())
        return U

    def stacked(self) -> Dict[str, torch.Tensor]:
        return {
            "X": torch.stack(self.X_frames, 1),   # (B, steps, dimX)
            "U": torch.stack(self.U_frames, 1),   # (B, steps, dimU)
        }


def _simulate_and_record(policy: nn.Module, B: int, rng: torch.Generator, sync_time: bool = False) -> Dict[str, torch.Tensor]:
    rec = RecordingPolicy(policy).to(device)
    init, _ = sample_initial_states(B, rng=rng)
    if sync_time:
        init["TmT"].fill_(T)
    simulate(rec, B, initial_states_dict=init, m_steps=m, train=False, rng=rng)
    return rec.stacked()


# ---------------------------------------------------------------------
# Series builders
# ---------------------------------------------------------------------
def _series_for_view(
    view: Dict[str, Any],
    traj_np: Dict[str, Any],
    Bsel: List[int],
) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    Extract (x_time, [series...], [labels...]) for a basic index view.
    Always returns 2D series with shape (steps, n_comp).
    """
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
        if arr is None:
            continue

        # normalize to (B, steps, dims)
        if arr.ndim == 1:
            arr = arr[None, :, None]
        elif arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.ndim > 3:
            Bn, steps = arr.shape[0], arr.shape[1]
            arr = arr.reshape(Bn, steps, -1)

        arr_sel = arr[Bsel]  # (B', steps, dims)
        steps = arr_sel.shape[1]
        if x_time is None:
            x_time = _linspace_time(steps)

        role_info   = SCHEMA.get("roles", {}).get(block_name, {})
        labels_info = role_info.get("labels", [])

        indices = view.get("indices", [0])
        valid_indices = [int(i) for i in indices if 0 <= int(i) < arr_sel.shape[-1]]
        if not valid_indices:
            continue

        base = arr_sel[0] if PGDPO_TRAJ_PLOT_FIRST else arr_sel.mean(axis=0)
        data = np.take(base, indices=valid_indices, axis=-1)  # (steps, ncomp) or (steps,)
        if data.ndim == 1:
            data = data.reshape(steps, 1)
        elif data.shape[0] != steps and data.shape[-1] == steps:
            data = data.T

        series = [data[:, i].astype(float) for i in range(data.shape[1])]
        if labels_info and max(valid_indices) < len(labels_info):
            labels = [labels_info[idx] for idx in valid_indices]
        else:
            labels = [f"dim_{idx}" for idx in valid_indices]

        all_series.extend(series)
        all_labels.extend(labels)

    if x_time is None:
        x_time = _linspace_time(m)
    return x_time, all_series, all_labels


def _handle_custom_view(
    view: Dict[str, Any],
    traj_np: Dict[str, Any],
    Bsel: List[int],
    policy_tag: Optional[str] = None
) -> Optional[Tuple[np.ndarray, List[np.ndarray], List[str]]]:
    mode = (view.get("mode", "") or "").lower()
    name = (view.get("name", "") or "").lower()

    # Consumption + stairs (C, C̄)
    if name == "consumption_stairs" or mode == "stairs":
        U = _to_np(traj_np.get("U"))[Bsel]
        if U.ndim == 2:
            U = U[..., None]
        if U.shape[2] <= d:
            return None
        C = U[..., -1]  # (B, steps)
        x_time = _linspace_time(C.shape[1])
        c_line = C[0] if PGDPO_TRAJ_PLOT_FIRST else C.mean(axis=0)
        cbar = _stairs_1d(c_line)
        return x_time, [c_line, cbar], ["Consumption", "Cbar_ref"]

    # Per-step consumption AMOUNT: C·dt
    if name in ("consumption_amount", "consumption_amt") or mode == "c_amount":
        U = _to_np(traj_np.get("U"))[Bsel]
        if U.ndim == 2:
            U = U[..., None]
        if U.shape[2] <= d:
            return None
        C = U[..., -1]
        steps = C.shape[1]
        x_time = _linspace_time(steps)
        dt = (x_time[1] - x_time[0]) if steps > 1 else float(T) / max(steps, 1)
        c_line = C[0] if PGDPO_TRAJ_PLOT_FIRST else C.mean(axis=0)
        amount = c_line * dt
        return x_time, [amount], ["C_amount (C·dt)"]

    # Cumulative consumption: ∑ C·dt
    if name in ("consumption_cumulative", "consumption_cum", "cumc") or mode == "c_cum":
        U = _to_np(traj_np.get("U"))[Bsel]
        if U.ndim == 2:
            U = U[..., None]
        if U.shape[2] <= d:
            return None
        C = U[..., -1]
        steps = C.shape[1]
        x_time = _linspace_time(steps)
        dt = (x_time[1] - x_time[0]) if steps > 1 else float(T) / max(steps, 1)
        c_line = C[0] if PGDPO_TRAJ_PLOT_FIRST else C.mean(axis=0)
        cum_amount = np.cumsum(c_line * dt)
        return x_time, [cum_amount], ["CumC (∑ C·dt)"]

    # Risky share π/X (single-asset: u/X; multi-asset: first component share for preview)
    if name == "risky_share" or mode == "risky_share":
        X = _to_np(traj_np.get("X"))[Bsel]
        U = _to_np(traj_np.get("U"))[Bsel]
        if X.ndim == 2:
            X = X[..., None]
        if U.ndim == 2:
            U = U[..., None]
        x_time = _linspace_time(X.shape[1])
        x = X[0, :, 0] if PGDPO_TRAJ_PLOT_FIRST else X.mean(axis=0)[:, 0]
        if d >= 1:
            u1 = U[0, :, 0] if PGDPO_TRAJ_PLOT_FIRST else U.mean(axis=0)[:, 0]
            share = u1 / np.clip(x, 1e-12, None)
            return x_time, [share], ["Risky Share"]
        return None

    # Discretionary wealth: DW = X - b(t, C̄)
    if name in ("discretionary_wealth", "dw") or mode == "dw":
        X = _to_np(traj_np["X"])[Bsel]
        U = _to_np(traj_np.get("U"))[Bsel]
        if U.ndim == 2:
            U = U[..., None]
        x = X[0, :, 0] if PGDPO_TRAJ_PLOT_FIRST else X.mean(axis=0)[:, 0]
        C = U[0, :, -1] if PGDPO_TRAJ_PLOT_FIRST else U.mean(axis=0)[:, -1]
        from user_pgdpo_base import r as Rf, T as TT
        t = _linspace_time(x.shape[0])
        tau = TT - t
        cbar = _stairs_1d(C)
        b = cbar * (1.0 - np.exp(-Rf * tau)) / max(Rf, 1e-12)
        dw = x - b
        return t, [dw], ["DW = X - b(t,C̄)"]

    # Log-returns Δ log X
    if name in ("x_log_returns", "x_logret") or mode == "logret":
        X = _to_np(traj_np["X"])[Bsel]
        x = X[0, :, 0] if PGDPO_TRAJ_PLOT_FIRST else X.mean(axis=0)[:, 0]
        logx = np.log(np.clip(x, 1e-12, None))
        ret = np.diff(logx)
        t = _linspace_time(len(ret))
        return t, [ret], ["Δlog X"]

    # Wealth with b(t,C̄) and BW boundary X̄(t,C̄)
    if name == "wealth_with_thresholds" or mode == "jedc_wealth":
        X = _to_np(traj_np["X"])[Bsel]
        U = _to_np(traj_np.get("U"))[Bsel] if "U" in traj_np else None
        if U is None or U.shape[-1] <= d:
            return None
        x = X[0, :, 0] if PGDPO_TRAJ_PLOT_FIRST else X.mean(axis=0)[:, 0]
        C = U[0, :, -1] if PGDPO_TRAJ_PLOT_FIRST else U.mean(axis=0)[:, -1]
        from user_pgdpo_base import r as Rf, rho as Rho, gamma as Gamma, T as TT
        t = _linspace_time(x.shape[0])
        tau = TT - t
        cbar = _stairs_1d(C)
        b = cbar * (1.0 - np.exp(-Rf * tau)) / max(Rf, 1e-12)
        theta_eff = float(_R_INFO.get("theta_eff", 0.0) or _R_INFO.get("theta", 0.0) or 0.0)
        Rstar = np.array([_bw_rstar_tau(theta_eff, float(Rho), float(Rf), float(tt)) for tt in tau])
        Xbar = (Gamma / np.maximum(Gamma - Rstar, 1e-8)) * b
        return t, [x, b, Xbar], ["Wealth (X)", "b(t,C)_ref", "Xbar_ref"]

    return None


def _series_for_view_wrapper(view: Dict[str, Any], traj_np: Dict[str, Any], Bsel: List[int], policy_tag: Optional[str] = None):
    res = _handle_custom_view(view, traj_np, Bsel, policy_tag)
    if res is not None:
        return res
    return _series_for_view(view, traj_np, Bsel)


# ---------------------------------------------------------------------
# CSV saver and plotting
# ---------------------------------------------------------------------
def _save_series_to_csv(x_time: np.ndarray, all_views_data: Dict[str, Dict[str, Dict[str, np.ndarray]]], out_path: str) -> None:
    x_time = np.asarray(x_time, dtype=float).reshape(-1)
    Tlen = int(x_time.shape[0])

    colmap: Dict[str, np.ndarray] = {}
    for view_name, series_map in all_views_data.items():
        for component, policies in sorted(series_map.items()):
            for policy, data in sorted(policies.items()):
                col_name = component if ("ref" in component.lower()) else f"{component}_{policy}"
                colmap[col_name] = _as_ts1d(data, Tlen)

    header = ["Time"] + list(colmap.keys())
    rows = np.stack([x_time] + [colmap[h] for h in header[1:]], axis=-1)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[traj] Trajectory data saved to: {os.path.basename(out_path)}")


def _plot_lines(x_time: np.ndarray, series_map: Dict[str, Dict[str, np.ndarray]],
                title: str, ylabel: str, save_path: str, view_opts: dict = {}) -> None:
    POLICY_STYLES = {
        "cf":    {"color": "royalblue",  "ls": "-",  "lw": 2.2, "zorder": 5, "label": "Ref/Myopic"},
        "pp":    {"color": "orangered",  "ls": "--", "lw": 1.8, "zorder": 4, "label": "P-PGDPO (pp)", "marker": "o", "ms": 4, "alpha": 0.85},
        "learn": {"color": "forestgreen","ls": ":",  "lw": 1.9, "zorder": 3, "label": "Learned", "marker": "s", "ms": 3.5, "alpha": 0.9},
        "ref":   {"color": "gray",       "ls": "-.", "lw": 1.3, "zorder": 2}
    }
    if "legend_labels" in SCHEMA:
        for key, new_label in SCHEMA["legend_labels"].items():
            if key in POLICY_STYLES:
                POLICY_STYLES[key]["label"] = new_label

    fig, ax1 = plt.subplots(figsize=(8.8, 4.6))
    Tlen = len(x_time)

    for comp_label, policies in sorted(series_map.items()):
        # treat *_ref components as reference overlays (no policy suffix)
        is_reference = ("ref" in comp_label.lower())
        if is_reference:
            key = "cf" if "cf" in policies else (next(iter(policies)) if policies else None)
            if key is None:
                continue
            y = _as_ts1d(policies[key], Tlen)
            ax1.plot(x_time, y, **{k: v for k, v in POLICY_STYLES["ref"].items()})
            continue

        for pkey in ["cf", "pp", "learn"]:
            if pkey in policies:
                y = _as_ts1d(policies[pkey], Tlen)
                st = {k: v for k, v in POLICY_STYLES.get(pkey, {}).items() if k != "label"}
                label = f"{comp_label} - {POLICY_STYLES.get(pkey, {}).get('label', pkey)}"
                ax1.plot(x_time, y, label=label, **st)

    ax1.set(xlabel="Time", title=title)
    ax1.set_ylabel(ylabel)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    h1, l1 = ax1.get_legend_handles_labels()
    if h1:
        ax1.legend(h1, l1, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
@torch.no_grad()
def generate_and_save_trajectories(
    policy_learn: nn.Module,
    policy_cf: Optional[nn.Module] = None,
    outdir: Optional[str] = None,
    seed_crn: Optional[int] = None
) -> str:
    """Simulate trajectories for (learn[, cf[, pp]]) under shared CRN and save plots/CSV."""
    outdir = _ensure_outdir(outdir)
    B_all = int(PGDPO_TRAJ_B or N_eval_states)
    seed_crn = int(seed_crn or CRN_SEED_EU)

    # Prepare policies
    policies_to_sim: Dict[str, nn.Module] = {"learn": policy_learn.to(device)}
    if policy_cf is not None:
        policies_to_sim["cf"] = policy_cf.to(device)

    # Add pp if requested/available
    pp_available = (_HAS_DIRECT or _HAS_RUNNER)
    want_pp = PGDPO_TRAJ_PLOT_PP and (PGDPO_CURRENT_MODE in ["projection", "run"] or PGDPO_TRAJ_FORCE_PP)
    if want_pp and pp_available:
        if PP_MODE == "run" and _HAS_RUNNER:
            print("[traj] Generating 'pp' using RUNNER (throttle via PGDPO_PP_REPEATS/SUBBATCH).")
            policies_to_sim["pp"] = PPRUNNERPolicy(policy_learn, seed_crn).to(device)
        else:
            print("[traj] Generating 'pp' using DIRECT (throttle via PGDPO_PP_REPEATS/SUBBATCH).")
            policies_to_sim["pp"] = PPDirectPolicy(policy_learn, seed_crn).to(device)

    # -------- Cache trajectories once per policy --------
    traj_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    if PGDPO_TRAJ_CACHE:
        print("[traj] Caching trajectories once per policy...")
        for name, policy_obj in policies_to_sim.items():
            gen = make_generator(seed_crn)  # CRN sharing
            tr = _simulate_and_record(policy_obj, B_all, gen, sync_time=True)
            if PGDPO_TRAJ_DECIMATE > 1:
                tr = _decimate_traj(tr, PGDPO_TRAJ_DECIMATE)
            traj_cache[name] = tr

    # Optional view filtering
    only = set([s.strip() for s in PGDPO_TRAJ_ONLY.split(",") if s.strip()]) if PGDPO_TRAJ_ONLY else None
    skip = set([s.strip() for s in PGDPO_TRAJ_SKIP.split(",") if s.strip()]) if PGDPO_TRAJ_SKIP else set()

    all_views_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    master_x_time: Optional[np.ndarray] = None
    Bsel = list(range(min(B_all, SCHEMA.get("sampling", {}).get("Bmax", PGDPO_TRAJ_BMAX))))

    for view in SCHEMA.get("views", []):
        vname = view.get("name", "")
        if only and vname not in only:
            continue
        if vname in skip:
            continue

        series_map: Dict[str, Dict[str, np.ndarray]] = {}
        x_time: Optional[np.ndarray] = None

        for name, policy_obj in policies_to_sim.items():
            if PGDPO_TRAJ_CACHE:
                traj_data = traj_cache[name]
            else:
                gen = make_generator(seed_crn)  # CRN sharing, per-view (slower)
                traj_data = _simulate_and_record(policy_obj, B_all, gen, sync_time=True)
                if PGDPO_TRAJ_DECIMATE > 1:
                    traj_data = _decimate_traj(traj_data, PGDPO_TRAJ_DECIMATE)

            curr_x, series, labels = _series_for_view_wrapper(view, traj_data, Bsel, policy_tag=name)
            if curr_x is None or not series:
                continue

            if x_time is None:
                x_time = curr_x
                if master_x_time is None:
                    master_x_time = x_time

            for s, lab in zip(series, labels):
                series_map.setdefault(lab, {})[name] = s

        if x_time is not None and series_map:
            _plot_lines(
                x_time, series_map,
                f"Trajectory Comparison: {vname}",
                view.get("ylabel", "Value"),
                os.path.join(outdir, f"traj_{vname}_comparison.png"),
                view_opts=view
            )
            all_views_data[vname] = series_map

    # Save combined CSV
    if not PGDPO_TRAJ_NO_CSV and master_x_time is not None and all_views_data:
        _save_series_to_csv(master_x_time, all_views_data, os.path.join(outdir, "traj_comparison_data.csv"))

    return outdir
