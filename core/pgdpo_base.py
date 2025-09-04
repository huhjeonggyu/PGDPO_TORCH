# core/pgdpo_base.py
# 공통 러너/헬퍼: 전 모드에서 공통으로 쓰는 유틸과 기본 학습/비교 함수

from __future__ import annotations

import random
import importlib
from typing import Callable, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# 사용자 모델 심볼 로드 (tests/<model>/user_pgdpo_base.py가 제공)
# -----------------------------------------------------------------------------
try:
    from user_pgdpo_base import (
        # 장치/격자/차원
        device,  # torch.device
        T, m, d, k,
        DIM_X, DIM_Y, DIM_U,            # ✅ 추가: 사용자 정의 차원 그대로 재노출
        # 하이퍼파라미터/시드
        epochs, batch_size, lr, seed,
        # 평가용 공통 시드
        CRN_SEED_EU,
        # 데이터/시뮬레이터/정책
        sample_initial_states,  # (B, rng=Generator) -> (states_dict, aux?)
        simulate,               # simulate(policy, B, initial_states_dict=..., random_draws=(ZX, ZY), m_steps=m) -> U
        DirectPolicy,           # nn.Module
        # 평가 배치 크기
        N_eval_states,
    )
except Exception as e:
    raise RuntimeError(f"[pgdpo_base] Failed to import symbols from user_pgdpo_base: {e}")

# -----------------------------------------------------------------------------
# RNG 유틸
# -----------------------------------------------------------------------------
def make_generator(seed_local: Optional[int] = None) -> torch.Generator:
    """
    Torch RNG 생성기. 가능하면 사용자 device에 올리고, 실패 시 CPU 생성.
    """
    if seed_local is None:
        seed_local = 0
    try:
        gen = torch.Generator(device=device)
    except Exception:
        gen = torch.Generator()
    gen.manual_seed(int(seed_local))
    return gen


def set_global_seeds(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# -----------------------------------------------------------------------------
# 공통 노이즈 생성 (X: d, Y: k)
# -----------------------------------------------------------------------------
def _draw_base_normals(B: int, steps: int, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    시뮬레이션용 표준정규 노이즈를 (ZX, ZY)로 분리 생성.
    Shapes:
      ZX: (B, steps, d), ZY: (B, steps, k)
    """
    Z = torch.randn(B, steps, d + k, device=device, generator=gen)
    ZX, ZY = Z[:, :, :d], Z[:, :, d:]
    return ZX, ZY

# -----------------------------------------------------------------------------
# 공통 실행기
# -----------------------------------------------------------------------------
def run_common(
    *,
    train_fn: Callable[..., nn.Module],
    rmse_fn: Callable[..., Any],
    seed_train: Optional[int] = None,
    train_kwargs: Optional[Dict[str, Any]] = None,
    rmse_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    공통 실행기(간소화 버전):
      1) 전역 시드 고정
      2) 학습 함수 호출 (train_kwargs 전달)
      3) 폐형해는 오직 user_pgdpo_base.build_closed_form_policy()만 사용
         - 반환이 (module, extra) 형태면 module만 사용
         - 반환이 module이면 그대로 사용
         - 반환이 callable이면 간단 래퍼(nn.Module)로 감싸 사용
      4) RMSE/플롯 함수 호출 (rmse_kwargs 전달)
    """
    set_global_seeds(int(seed_train if seed_train is not None else seed))
    train_kwargs = train_kwargs or {}
    rmse_kwargs = rmse_kwargs or {}

    # 1) 학습
    policy = train_fn(**train_kwargs)

    # 2) 폐형해: build_closed_form_policy만 사용
    cf_policy = None
    try:
        upb = importlib.import_module("user_pgdpo_base")
        if hasattr(upb, "build_closed_form_policy") and callable(upb.build_closed_form_policy):
            res = upb.build_closed_form_policy()

            # (module, extra) 형태 지원
            if isinstance(res, tuple) and len(res) >= 1:
                res0 = res[0]
            else:
                res0 = res

            if isinstance(res0, nn.Module):
                cf_policy = res0.to(device)
                print("✅ Closed-form policy loaded via build_closed_form_policy()")
            elif callable(res0):
                class _Wrap(nn.Module):
                    def __init__(self, f):
                        super().__init__(); self.f = f
                    def forward(self, **states):
                        return self.f(**states)
                cf_policy = _Wrap(res0).to(device)
                print("✅ Closed-form callable wrapped as nn.Module via build_closed_form_policy()")
            else:
                print("ℹ️ build_closed_form_policy() returned unsupported type; ignoring closed-form.")
        else:
            print("ℹ️ build_closed_form_policy() not found; proceeding without closed-form.")
    except Exception as e:
        print(f"[WARN] build_closed_form_policy() failed: {e}")
        cf_policy = None

    # 3) 평가/시각화
    rmse_fn(policy, cf_policy, **rmse_kwargs)

# -----------------------------------------------------------------------------
# 기본 학습 루프(Base)
# -----------------------------------------------------------------------------
def train_stage1_base(
    epochs_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    seed_train: Optional[int] = None,
    outdir: Optional[str] = None,
) -> nn.Module:
    """
    기본 시뮬레이터 기반 Stage-1 학습 (분산감소/프로젝션 없음).
    - seed_local을 직접 넘기지 않고, 초기상태/노이즈를 생성해 simulate(...)에 전달.
    """
    _epochs = int(epochs_override if epochs_override is not None else epochs)
    _lr = float(lr_override if lr_override is not None else lr)

    policy = DirectPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=_lr)

    loss_hist: list[float] = []
    for ep in range(1, _epochs + 1):
        opt.zero_grad()

        # 재현성 있는 초기상태 + 노이즈
        gen = make_generator((seed_train or 0) + ep)
        states, _ = sample_initial_states(batch_size, rng=gen)
        ZX, ZY = _draw_base_normals(batch_size, m, gen)

        # simulate는 다음 형태를 가정:
        #   U = simulate(policy, B, initial_states_dict=states, random_draws=(ZX, ZY), m_steps=m)
        U = simulate(
            policy,
            batch_size,
            initial_states_dict=states,
            random_draws=(ZX, ZY),
            m_steps=m,
        )

        loss = -U.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            print(f"[{ep:04d}] loss={loss.item():.6f}")
        loss_hist.append(float(loss.item()))

    # (옵션) 손실 저장
    if outdir is not None:
        try:
            from viz import save_loss_curve, save_loss_csv
            save_loss_csv(loss_hist, outdir, "loss_history.csv")
            save_loss_curve(loss_hist, outdir, "loss_curve.png")
        except Exception as e:
            print(f"[WARN] base: could not save loss plots: {e}")

    return policy

# -----------------------------------------------------------------------------
# base 모드 평가기: learn↔cf만 비교/플롯
# -----------------------------------------------------------------------------
@torch.no_grad()
def print_policy_rmse_and_samples_base(
    pol_s1: nn.Module,
    pol_cf: Optional[nn.Module],
    *,
    repeats: int = 0,            # run.py 호환을 위해 받지만 사용하지 않음
    sub_batch: int = 0,          # run.py 호환을 위해 받지만 사용하지 않음
    seed_eval: Optional[int] = None,
    tile: Optional[int] = None,  # 호환 인자 (미사용)
    outdir: Optional[str] = None,
) -> None:
    """
    Base 모드 평가기.
    - u_learn = pol_s1(**states)
    - (있으면) u_cf = pol_cf(**states)
    - RMSE 출력 + 겹쳐 히스토그램/산점도 저장 (u_pp는 없음)
    """
    gen = make_generator(seed_eval or CRN_SEED_EU)
    states_dict, _ = sample_initial_states(N_eval_states, rng=gen)

    u_learn = pol_s1(**states_dict)

    if pol_cf is None:
        print("[INFO] No closed-form policy available; skipping base comparison.")
        return

    u_cf = pol_cf(**states_dict)
    rmse = torch.sqrt(((u_learn - u_cf) ** 2).mean()).item()
    print(f"[Policy RMSE] ||u_learn - u_closed-form||_RMSE: {rmse:.6f}")

    if outdir is not None:
        try:
            from viz import (
                save_policy_scatter,
                save_overlaid_delta_hists,
                append_metrics_csv,
            )
            # 산점도/차이 분포 저장 (대표 좌표 0)
            save_policy_scatter(
                u_ref=u_cf, u_pred=u_learn, outdir=outdir, coord=0,
                fname="scatter_base_learn_vs_cf_dim0.png", xlabel="u_cf", ylabel="u_learn"
            )
            save_overlaid_delta_hists(
                u_learn=u_learn, u_pp=None, u_cf=u_cf,
                outdir=outdir, coord=0, fname="delta_base_overlaid_hist.png", bins=60, density=True
            )
            append_metrics_csv({"rmse_learn_cf_base": rmse}, outdir)
        except Exception as e:
            print(f"[WARN] base: could not save compare plots: {e}")

    # 대표 샘플 3개 출력
    B = next(iter(states_dict.values())).size(0)
    for i in [0, B // 2, B - 1]:
        parts, vec = [], False
        for k_, v in states_dict.items():
            ts = v[i]
            if ts.numel() > 1:
                parts.append(f"{k_}[0]={ts[0].item():.3f}"); vec = True
            else:
                parts.append(f"{k_}={ts.item():.3f}")
        if vec: parts.append("...")
        sstr = ", ".join(parts)
        print(
            f"  ({sstr}) -> (u_learn[0]={u_learn[i,0].item():.4f}, "
            f"u_cf[0]={u_cf[i,0].item():.4f}, ...)"
        )

# -----------------------------------------------------------------------------
# (선택) 간단 비교기 유지
# -----------------------------------------------------------------------------
@torch.no_grad()
def compare_policy_functions(
    policy: nn.Module,
    cf_policy: Optional[nn.Module],
    *,
    seed_eval: Optional[int] = None,
    outdir: Optional[str] = None,
) -> None:
    """
    학습 정책 vs 폐형해 비교. 폐형해가 없으면 스킵.
    (간단 버전; base 모드 평가기가 있으므로 보조 용도)
    """
    gen = make_generator(seed_eval or CRN_SEED_EU)
    states_dict, _ = sample_initial_states(N_eval_states, rng=gen)

    u_learn = policy(**states_dict)

    if cf_policy is None:
        print("[INFO] No closed-form policy available; skipping base comparison.")
        return

    u_cf = cf_policy(**states_dict)
    rmse = torch.sqrt(((u_learn - u_cf) ** 2).mean()).item()
    print(f"[Policy RMSE] ||u_learn - u_closed-form||_RMSE: {rmse:.6f}")

    if outdir is not None:
        try:
            from viz import (
                save_policy_scatter,
                save_overlaid_delta_hists,  # 겹침 히스토그램 사용
                append_metrics_csv,
            )
            save_policy_scatter(
                u_ref=u_cf, u_pred=u_learn, outdir=outdir, coord=0,
                fname="scatter_base_learn_vs_cf_dim0.png", xlabel="u_cf", ylabel="u_learn"
            )
            save_overlaid_delta_hists(
                u_learn=u_learn, u_pp=None, u_cf=u_cf,
                outdir=outdir, coord=0, fname="delta_base_overlaid_hist.png", bins=60, density=True
            )
            append_metrics_csv({"rmse_learn_cf_base": rmse}, outdir)
        except Exception as e:
            print(f"[WARN] base: could not save compare plots: {e}")

# -----------------------------------------------------------------------------
# 독립 실행 테스트
# -----------------------------------------------------------------------------
def main():
    """
    Quick standalone test for base mode:
    - Train Stage-1 (base)
    - Evaluate with base evaluator (learn vs cf only)
    """
    run_common(
        train_fn=train_stage1_base,
        rmse_fn=print_policy_rmse_and_samples_base,
        train_kwargs={},  # 필요 시 epochs/lr/outdir override 가능
        rmse_kwargs={
            "seed_eval": CRN_SEED_EU,
            # "outdir": "<원하면 경로 지정>"
        },
    )

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# __all__
# -----------------------------------------------------------------------------
__all__ = [
    # 환경/격자/차원/하이퍼
    "device", "T", "m", "d", "k", "epochs", "batch_size", "lr", "seed",
    "CRN_SEED_EU", "N_eval_states",
    "DIM_X", "DIM_Y", "DIM_U",              # ✅ 추가 재노출
    # 유틸
    "make_generator", "set_global_seeds",
    # 사용자 훅/심볼 재노출
    "sample_initial_states", "simulate", "DirectPolicy",
    # 공통 실행기/루프/비교
    "run_common",
    "train_stage1_base",
    "print_policy_rmse_and_samples_base",
    "compare_policy_functions",
]