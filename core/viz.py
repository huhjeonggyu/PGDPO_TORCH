# core/viz.py
# 역할: 결과 시각화/로그 유틸
# - [수정] u_learn, u_pp, u_cf를 하나의 산점도에 그리는 기능 추가
# - ✨ [SIR] (S, I) 상태 공간에 대한 정책 히트맵 생성 기능 추가

from __future__ import annotations

import os
import csv
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ✨ [SIR 추가] T (Horizon)를 user_pgdpo_base에서 가져옵니다.
from pgdpo_base import T

# ------------------------------------------------------------
# 내부 유틸
# ------------------------------------------------------------
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _as_numpy_1d(x: torch.Tensor | np.ndarray, coord: int = 0) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu()
        if x.ndim == 1: arr = x.numpy()
        elif x.ndim == 2: arr = x[:, coord].numpy()
        else: arr = x.reshape(x.shape[0], -1)[:, coord].numpy()
    else:
        x = np.asarray(x)
        if x.ndim == 1: arr = x
        elif x.ndim == 2: arr = x[:, coord]
        else: arr = x.reshape(x.shape[0], -1)[:, coord]
    return arr

def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    sse = np.sum((y_pred - y_true) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    if sst <= 1e-12:
        return 1.0 if sse <= 1e-12 else 0.0
    return 1.0 - (sse / sst)

# ------------------------------------------------------------
# 통합 산점도 (R^2 주석 포함)
# ------------------------------------------------------------
def save_combined_scatter(
    *,
    u_ref: torch.Tensor | np.ndarray,
    u_learn: torch.Tensor | np.ndarray,
    u_pp: Optional[torch.Tensor | np.ndarray],
    outdir: str,
    fname: str,
    coord: int = 0,
    xlabel: str = "u_closed-form",
    title: Optional[str] = None,
) -> str:
    _ensure_dir(outdir)
    x_ref = _as_numpy_1d(u_ref, coord=coord)
    y_learn = _as_numpy_1d(u_learn, coord=coord)
    
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    r2_learn = _compute_r2(x_ref, y_learn)
    ax.scatter(x_ref, y_learn, s=15, alpha=0.7, label=f"Learned (R²={r2_learn:.4f})", zorder=2)

    if u_pp is not None:
        y_pp = _as_numpy_1d(u_pp, coord=coord)
        r2_pp = _compute_r2(x_ref, y_pp)
        ax.scatter(x_ref, y_pp, s=15, alpha=0.7, label=f"P-PGDPO (R²={r2_pp:.4f})", zorder=3)

    all_vals = np.concatenate([x_ref, y_learn] + ([_as_numpy_1d(u_pp, coord=coord)] if u_pp is not None else []))
    lo, hi = np.min(all_vals), np.max(all_vals)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label="y=x (Perfect Match)", zorder=4)

    ax.set(xlabel=xlabel, ylabel="Policy Output", title=title or f"Policy Comparison (dim {coord})")
    ax.legend(loc="best")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    out_path = os.path.join(outdir, fname)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path

# ------------------------------------------------------------
# 오버레이 히스토그램
# ------------------------------------------------------------
def save_overlaid_delta_hists(
    *,
    u_learn: torch.Tensor | np.ndarray,
    u_pp: Optional[torch.Tensor | np.ndarray],
    u_cf: Optional[torch.Tensor | np.ndarray],
    outdir: str,
    coord: int = 0,
    fname: str = "delta_overlaid_hist.png",
    bins: int = 60,
) -> str:
    _ensure_dir(outdir)
    series: list[tuple[str, np.ndarray]] = []
    
    if u_cf is not None:
        series.append(("Learned vs Ref.", _as_numpy_1d(u_learn, coord) - _as_numpy_1d(u_cf, coord)))
    if u_pp is not None and u_cf is not None:
        series.append(("P-PGDPO vs Ref.", _as_numpy_1d(u_pp, coord) - _as_numpy_1d(u_cf, coord)))

    if not series: return ""

    all_vals = np.concatenate([s[1] for s in series])
    lo, hi = np.min(all_vals), np.max(all_vals)
    span = max(hi - lo, 1e-8)
    edges = np.histogram_bin_edges(all_vals, bins=bins, range=(lo - 0.05 * span, hi + 0.05 * span))

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for label, vals in series:
        mu, sd = np.mean(vals), np.std(vals)
        ax.hist(vals, bins=edges, density=True, histtype="step", linewidth=1.5, label=f"{label} (μ={mu:.3f}, σ={sd:.3f})")
    
    ax.axvline(0.0, color='gray', linestyle="--", linewidth=1.0)
    ax.set(xlabel="Policy Difference (Delta)", ylabel="Density", title=f"Overlaid Policy Deltas (dim {coord})")
    ax.legend(loc="best")
    out_path = os.path.join(outdir, fname)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path

# ------------------------------------------------------------
# ✨ [SIR 추가] 정책 히트맵 생성 함수
# ------------------------------------------------------------
def save_policy_heatmap(
    policy: torch.nn.Module,
    t: float,
    s_range: tuple = (0.0, 1.5),
    i_range: tuple = (0.0, 0.5),
    n_grid: int = 50,
    outdir: str = "plots",
    fname: str = "policy_heatmap.png"
) -> str:
    """(S, I) 상태 공간에서 정책의 등고선 히트맵을 생성합니다."""
    _ensure_dir(outdir)
    s_grid = torch.linspace(s_range[0], s_range[1], n_grid)
    i_grid = torch.linspace(i_range[0], i_range[1], n_grid)
    S, I = torch.meshgrid(s_grid, i_grid, indexing='ij')

    S_flat, I_flat = S.reshape(-1, 1), I.reshape(-1, 1)
    R_flat = torch.zeros_like(S_flat) # R은 0으로 가정
    
    # 정책 입력에 맞는 상태 텐서 X를 생성합니다 (지역이 1개라고 가정).
    X_flat = torch.cat([S_flat, I_flat, R_flat], dim=1).to(next(policy.parameters()).device)
    TmT_flat = torch.full_like(S_flat, T - t).to(X_flat.device)

    with torch.no_grad():
        u_flat = policy(X=X_flat, TmT=TmT_flat).sum(dim=1)
    U = u_flat.reshape(n_grid, n_grid).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    c = ax.contourf(S.numpy(), I.numpy(), U, levels=20, cmap='viridis')
    ax.set(xlabel="Susceptible Population (S)", ylabel="Infected Population (I)",
           title=f"Policy u(S, I) at t = {t:.2f}")
    fig.colorbar(c, ax=ax, label="Control Intensity (u)")
    
    out_path = os.path.join(outdir, fname)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[viz] Policy heatmap saved to: {os.path.basename(out_path)}")
    return out_path

# ------------------------------------------------------------
# 손실 곡선/CSV, 메트릭 CSV
# ------------------------------------------------------------
def save_loss_curve(loss_hist: list[float], outdir: str, fname: str = "loss_curve.png") -> str:
    _ensure_dir(outdir)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(np.arange(1, len(loss_hist) + 1), loss_hist)
    ax.set(xlabel="Epoch", ylabel="Loss")
    fig.tight_layout(); out_path = os.path.join(outdir, fname); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path

def save_loss_csv(loss_hist: list[float], outdir: str, fname: str = "loss_history.csv") -> str:
    _ensure_dir(outdir)
    out_path = os.path.join(outdir, fname)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        writer.writerows([[i, float(v)] for i, v in enumerate(loss_hist, 1)])
    return out_path

def append_metrics_csv(metrics: Dict[str, Any], outdir: str, fname: str = "metrics.csv") -> str:
    _ensure_dir(outdir)
    path = os.path.join(outdir, fname)
    file_exists = os.path.exists(path)
    header, rows = [], []
    if file_exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            rows = list(reader)
    
    new_keys = [k for k in metrics if k not in header]
    header.extend(new_keys)
    
    new_row = {k: metrics.get(k, "") for k in header}
    rows.append(new_row)
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path

__all__ = [
    "save_combined_scatter", "save_overlaid_delta_hists", "save_policy_heatmap",
    "save_loss_curve", "save_loss_csv", "append_metrics_csv", "_ensure_dir",
]