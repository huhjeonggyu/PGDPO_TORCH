# core/viz.py
# 역할: 결과 시각화/로그 유틸
# - matplotlib만 사용, 플롯당 하나의 차트
# - 색상 지정하지 않음(기본값 사용)
# - 서버 환경(headless)에서도 파일 저장 가능하도록 Agg 사용

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
# 내부 유틸
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
# 산점도 (R^2 주석 포함)
# ------------------------------------------------------------
def save_policy_scatter(
    *,
    u_ref: torch.Tensor | np.ndarray,
    u_pred: torch.Tensor | np.ndarray,
    outdir: str,
    fname: str,
    coord: int = 0,
    xlabel: str = "reference",
    ylabel: str = "prediction",
    title: Optional[str] = None,
) -> str:
    """
    u_ref vs u_pred 산점도 저장. 그림에 R^2를 주석으로 표기.
    """
    _ensure_dir(outdir)

    x = _as_numpy_1d(u_ref, coord=coord)
    y = _as_numpy_1d(u_pred, coord=coord)
    r2 = _compute_r2(x, y)

    fig = plt.figure(figsize=(5.0, 5.0))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    ax.plot([lo, hi], [lo, hi])

    # R^2 주석 - 소수점 8자리까지 표시하도록 수정
    ax.text(
        0.02, 0.98,
        f"R² = {r2:.8f}",
        transform=ax.transAxes,
        ha="left", va="top",
    )

    out_path = os.path.join(outdir, fname)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------
# 히스토그램 (단일 쌍)
# ------------------------------------------------------------
def save_delta_hist(
    *,
    u_a: torch.Tensor | np.ndarray,
    u_b: torch.Tensor | np.ndarray,
    outdir: str,
    fname: str,
    label: Optional[str] = None,
    coord: int = 0,
    bins: int = 50,
) -> str:
    _ensure_dir(outdir)
    a = _as_numpy_1d(u_a, coord=coord)
    b = _as_numpy_1d(u_b, coord=coord)
    delta = a - b
    fig = plt.figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    ax.hist(delta, bins=bins)
    ax.set_xlabel(label or "delta")
    ax.set_ylabel("count")
    mu = float(np.mean(delta))
    sd = float(np.std(delta))
    ax.set_title(f"mean={mu:.4f}, std={sd:.4f}")
    out_path = os.path.join(outdir, fname)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------
# 히스토그램 (closed-form 있을 때 3쌍 자동 저장)
# ------------------------------------------------------------
def save_pairwise_hists(
    *,
    u_learn: torch.Tensor | np.ndarray,
    u_pp: Optional[torch.Tensor | np.ndarray],
    u_cf: Optional[torch.Tensor | np.ndarray],
    outdir: str,
    coord: int = 0,
    prefix: str = "delta",
    bins: int = 50,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    p1 = p2 = p3 = None
    if u_pp is not None:
        p1 = save_delta_hist(u_a=u_learn, u_b=u_pp, outdir=outdir, coord=coord, bins=bins, fname=f"{prefix}_learn_minus_pp_hist.png", label="u_learn - u_pp")
    if u_cf is not None:
        p2 = save_delta_hist(u_a=u_learn, u_b=u_cf, outdir=outdir, coord=coord, bins=bins, fname=f"{prefix}_learn_minus_cf_hist.png", label="u_learn - u_cf")
    if (u_pp is not None) and (u_cf is not None):
        p3 = save_delta_hist(u_a=u_pp, u_b=u_cf, outdir=outdir, coord=coord, bins=bins, fname=f"{prefix}_pp_minus_cf_hist.png", label="u_pp - u_cf")
    return p1, p2, p3


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
    density: bool = True,
    include_learn_pp: bool = False,
) -> str:
    _ensure_dir(outdir)
    series: list[tuple[str, np.ndarray]] = []
    def _as1(x):
        x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        return x[:, coord] if x.ndim == 2 else x

    if include_learn_pp and (u_pp is not None):
        series.append(("learn - pp", _as1(u_learn) - _as1(u_pp)))
    if u_cf is not None:
        series.append(("learn - cf", _as1(u_learn) - _as1(u_cf)))
    if (u_pp is not None) and (u_cf is not None):
        series.append(("pp - cf", _as1(u_pp) - _as1(u_cf)))

    if not series:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.set_title("No deltas available")
        out_path = os.path.join(outdir, fname)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    all_vals = np.concatenate([s[1] for s in series], axis=0)
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    span = max(hi - lo, 1e-8)
    edges = np.histogram_bin_edges(all_vals, bins=bins, range=(lo - 0.05 * span, hi + 0.05 * span))

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for label, vals in series:
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        ax.hist(vals, bins=edges, density=density, histtype="step", linewidth=1.5, label=f"{label} (μ={mu:.3f}, σ={sd:.3f})")

    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set(xlabel="delta", ylabel="density" if density else "count", title=f"Overlaid policy deltas (dim {coord})")
    ax.legend(loc="best")
    out_path = os.path.join(outdir, fname)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_pairwise_scatters(
    *,
    u_learn: torch.Tensor | np.ndarray,
    u_pp: Optional[torch.Tensor | np.ndarray],
    u_cf: Optional[torch.Tensor | np.ndarray],
    outdir: str,
    coord: int = 0,
    prefix: str = "scatter",
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    p1, p2, p3 = None, None, None
    if u_cf is not None:
        p2 = save_policy_scatter(u_ref=u_cf, u_pred=u_learn, outdir=outdir, coord=coord, fname=f"{prefix}_learn_vs_cf_dim{coord}.png", xlabel="u_cf", ylabel="u_learn")
    if (u_pp is not None) and (u_cf is not None):
        p3 = save_policy_scatter(u_ref=u_cf, u_pred=u_pp, outdir=outdir, coord=coord, fname=f"{prefix}_pp_vs_cf_dim{coord}.png", xlabel="u_cf", ylabel="u_pp")
    return p1, p2, p3


# ------------------------------------------------------------
# 손실 곡선/CSV, 메트릭 CSV
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
    
    new_row = {k: metrics.get(k, "") for k in header}
    rows.append(new_row)
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path


__all__ = [
    "save_policy_scatter", "save_delta_hist", "save_pairwise_hists", "save_overlaid_delta_hists",
    "save_loss_curve", "save_loss_csv", "save_pairwise_scatters", "append_metrics_csv", "_ensure_dir",
]