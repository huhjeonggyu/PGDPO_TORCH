# core/viz.py
# 역할: 결과 시각화/로그 유틸
# - [수정] u_learn, u_pp, u_cf를 하나의 산점도에 그리는 기능 추가

from __future__ import annotations

import os
import csv
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 내부 유틸 (변경 없음)
# ------------------------------------------------------------
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _as_numpy_1d(x: torch.Tensor | np.ndarray, coord: int = 0) -> np.ndarray:
    """
    (B,D) 텐서/배열에서 coord 차원을 1D로 꺼내거나,
    (B,)인 경우 그대로 반환.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu()
        if x.ndim == 1:
            arr = x.numpy()
        elif x.ndim == 2:
            arr = x[:, coord].numpy()
        else:
            arr = x.reshape(x.shape[0], -1)[:, coord].numpy()
    else:
        x = np.asarray(x)
        if x.ndim == 1:
            arr = x
        elif x.ndim == 2:
            arr = x[:, coord]
        else:
            arr = x.reshape(x.shape[0], -1)[:, coord]
    return arr


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R^2 = 1 - SSE/SST
    SST=0 (상수목표)일 때는 SSE==0이면 1.0, 아니면 0.0 처리.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = np.sum((y_pred - y_true) ** 2)
    y_mean = np.mean(y_true)
    sst = np.sum((y_true - y_mean) ** 2)
    if sst <= 1e-12:
        return 1.0 if sse <= 1e-12 else 0.0
    return 1.0 - (sse / sst)


# ------------------------------------------------------------
# ✨ [업데이트] 통합 산점도 (R^2 주석 포함)
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
    """
    u_learn vs u_ref와 u_pp vs u_ref를 하나의 산점도에 겹쳐 그립니다.
    각각의 R^2 값을 계산하여 범례(legend)에 함께 표시합니다.
    """
    _ensure_dir(outdir)

    x_ref = _as_numpy_1d(u_ref, coord=coord)
    y_learn = _as_numpy_1d(u_learn, coord=coord)
    
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111)

    # u_learn vs u_cf 플롯
    r2_learn = _compute_r2(x_ref, y_learn)
    ax.scatter(x_ref, y_learn, s=15, alpha=0.7, label=f"Learned (R²={r2_learn:.4f})", zorder=2)

    # u_pp vs u_cf 플롯 (u_pp가 있을 경우)
    if u_pp is not None:
        y_pp = _as_numpy_1d(u_pp, coord=coord)
        r2_pp = _compute_r2(x_ref, y_pp)
        ax.scatter(x_ref, y_pp, s=15, alpha=0.7, label=f"P-PGDPO (R²={r2_pp:.4f})", zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Policy Output")
    ax.set_title(title or f"Policy Comparison (dim {coord})")

    # y=x 기준선
    all_vals_list = [x_ref, y_learn]
    if u_pp is not None:
        all_vals_list.append(_as_numpy_1d(u_pp, coord=coord))
    all_vals = np.concatenate(all_vals_list)
    lo, hi = np.min(all_vals), np.max(all_vals)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label="y=x (Perfect Match)", zorder=4)

    ax.legend(loc="best")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    out_path = os.path.join(outdir, fname)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------
# 오버레이 히스토그램 (기존과 동일)
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
    density: bool = True,
    include_learn_pp: bool = False,
) -> str:
    _ensure_dir(outdir)
    series: list[tuple[str, np.ndarray]] = []
    def _as1(x):
        x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        return x[:, coord] if x.ndim == 2 else x

    if u_cf is not None:
        series.append(("Learned vs Closed-Form", _as1(u_learn) - _as1(u_cf)))
    if (u_pp is not None) and (u_cf is not None):
        series.append(("P-PGDPO vs Closed-Form", _as1(u_pp) - _as1(u_cf)))

    if not series:
        return ""

    all_vals = np.concatenate([s[1] for s in series], axis=0)
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    span = max(hi - lo, 1e-8)
    edges = np.histogram_bin_edges(all_vals, bins=bins, range=(lo - 0.05 * span, hi + 0.05 * span))

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for label, vals in series:
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        ax.hist(vals, bins=edges, density=density, histtype="step", linewidth=1.5, label=f"{label} (μ={mu:.3f}, σ={sd:.3f})")

    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set(xlabel="Policy Difference (Delta)", ylabel="Density" if density else "Count", title=f"Overlaid Policy Deltas (dim {coord})")
    ax.legend(loc="best")
    out_path = os.path.join(outdir, fname)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------
# 손실 곡선/CSV, 메트릭 CSV (기존과 동일)
# ------------------------------------------------------------
def save_loss_curve(loss_hist: list[float], outdir: str, fname: str = "loss_curve.png") -> str:
    _ensure_dir(outdir)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(np.arange(1, len(loss_hist) + 1), loss_hist)
    ax.set(xlabel="epoch", ylabel="loss")
    fig.tight_layout()
    out_path = os.path.join(outdir, fname)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
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
    
    # metrics 딕셔너리를 rows의 마지막에 추가하거나 업데이트
    # (tag가 같으면 업데이트, 없으면 추가하는 로직이 필요할 수 있으나 여기서는 단순 추가)
    new_row = {k: metrics.get(k, "") for k in header}
    rows.append(new_row)
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path


__all__ = [
    "save_combined_scatter", "save_overlaid_delta_hists",
    "save_loss_curve", "save_loss_csv", "append_metrics_csv", "_ensure_dir",
]