# core/pgdpo_run.py
# 역할: Variance-Reduced 러너 (Antithetic)와
#       P-PGDPO 사영 평가(런타임 버전) + 시각화/로그 저장(outdir)
# 규칙:
#   - 학습/평가용 simulate_run: antithetic 
#   - 코스테이트 추정: antithetic (±Z 평균)

from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from pgdpo_base import (
    device, T, m, d, k, N_eval_states,
    make_generator, run_common,
    sample_initial_states, simulate,          # 베이스 simulate (VR 조합에 사용)
    _draw_base_normals,                       # 베이스 노이즈 생성
    # 학습 하이퍼파라미터
    batch_size, lr, epochs, seed,
    # 평가 공통 seed
    CRN_SEED_EU,
)

# with_projection에서는 반복 관련 상수만 사용 (estimate_costates는 로컬 구현)
from pgdpo_with_projection import (
    REPEATS, SUBBATCH, project_pmp
)

from viz import (
    save_pairwise_hists, save_pairwise_scatters, append_metrics_csv, save_overlaid_delta_hists
)


# ------------------------------------------------------------
# Variance-Reduced 시뮬레이터 (학습/평가 공용)
#   - antithetic 평균
# ------------------------------------------------------------
def simulate_run(
    policy: nn.Module,
    B: int,
    *,
    initial_states_dict: Optional[Dict[str, torch.Tensor]] = None,
    seed_local: Optional[int] = None,
) -> torch.Tensor:
    """
    Variance-reduced estimator for U via antithetic.
    항상 pgdpo_base.simulate(...)와 pgdpo_base._draw_base_normals(...)를 사용한다.
    """
    # 상태 샘플링
    rng = make_generator(seed_local)
    states = (
        sample_initial_states(B, rng=rng)[0]
        if initial_states_dict is None
        else initial_states_dict
    )

    ZX_c, ZY_c = _draw_base_normals(B, m, rng)
    U_c_p = simulate(policy, B, initial_states_dict=states, random_draws=(+ZX_c, +ZY_c), m_steps=m)
    U_c_m = simulate(policy, B, initial_states_dict=states, random_draws=(-ZX_c, -ZY_c), m_steps=m)
    U_c = 0.5 * (U_c_p + U_c_m)

    return U_c

# ------------------------------------------------------------
# 코스테이트 추정: antithetic-only (±Z 평균)
# ------------------------------------------------------------
def estimate_costates_anti(
    policy_net: nn.Module,
    initial_states: Dict[str, torch.Tensor],
    repeats: int,
    sub_batch: int,
    seed_eval: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    코스테이트 추정 (JX, JXX, JXY):
      - antithetic (±Z 평균)
    """
    B = next(iter(initial_states.values())).size(0)

    # 미분 대상 leaf 상태 (X, Y만)
    leaf_states = {
        k: v.detach().clone().requires_grad_(True)
        for k, v in initial_states.items()
        if k in ["X", "Y"]
    }
    TmT_val = initial_states["TmT"].detach().clone()

    sums = {
        "JX":  torch.zeros_like(leaf_states["X"]),
        "JXX": torch.zeros_like(leaf_states["X"]),
        "JXY": torch.zeros_like(leaf_states.get("Y")) if "Y" in leaf_states else None,
    }

    # 정책 파라미터는 고정
    params = list(policy_net.parameters())
    req_bak = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)

    try:
        done = 0
        while done < repeats:
            rpts = min(sub_batch, repeats - done)

            # 동일 상태 rpts번 복제 → 총 B*rpts
            batch_states = {k: v.repeat(rpts, 1) for k, v in leaf_states.items()}
            batch_states["TmT"] = TmT_val.repeat(rpts, 1)

            # antithetic 노이즈(±Z) 생성: m 스텝
            gen = make_generator((seed_eval or 0) + done)
            ZX, ZY = _draw_base_normals(B * rpts, m, gen)

            # U(+Z), U(-Z) 평균 → antithetic U
            U_p = simulate(policy_net, B * rpts,
                           initial_states_dict=batch_states,
                           random_draws=(+ZX, +ZY), m_steps=m)
            U_m = simulate(policy_net, B * rpts,
                           initial_states_dict=batch_states,
                           random_draws=(-ZX, -ZY), m_steps=m)
            U = 0.5 * (U_p + U_m)                      # (B*rpts, ...)

            # (rpts, B, ·) 평균 → (B, ·) → 전체 평균 스칼라
            U_bar = U.view(rpts, B, -1).mean(dim=0)
            J_scalar = U_bar.mean()

            # 1차: JX
            JX_b = torch.autograd.grad(
                J_scalar, leaf_states["X"], retain_graph=True, create_graph=True
            )[0]

            # 2차: JXX, JXY
            targets = [leaf_states["X"]]
            if "Y" in leaf_states:
                targets.append(leaf_states["Y"])
            JXX_b, *JXY_b_tuple = torch.autograd.grad(
                JX_b.sum(), targets, retain_graph=False, create_graph=False, allow_unused=True
            )
            JXY_b = JXY_b_tuple[0] if JXY_b_tuple else None

            # 누적(서브배치 가중치 rpts)
            sums["JX"]  += JX_b.detach() * rpts
            sums["JXX"] += (JXX_b.detach() if JXX_b is not None else torch.zeros_like(leaf_states["X"])) * rpts
            if sums["JXY"] is not None and JXY_b is not None:
                sums["JXY"] += JXY_b.detach() * rpts

            done += rpts
    finally:
        for p, r in zip(params, req_bak):
            p.requires_grad_(r)

    inv = 1.0 / float(repeats)
    return {k: v * inv for k, v in sums.items() if v is not None}


# ------------------------------------------------------------
# P-PGDPO 사영 (런타임 버전): 코스테이트 추정 → PMP 사영
# ------------------------------------------------------------
def ppgdpo_u_run(
    policy_s1: nn.Module,
    states: Dict[str, torch.Tensor],
    repeats: int,
    sub_batch: int,
    seed_eval: Optional[int] = None,
) -> torch.Tensor:
    with torch.enable_grad():
        # ✅ antithetic-only costate estimator (로컬)
        costates = estimate_costates_anti(policy_s1, states, repeats, sub_batch, seed_eval=seed_eval)
        u = project_pmp(costates, states)
        return u.detach()

def _divisors_desc(n: int):
    return sorted([d for d in range(1, n + 1) if n % d == 0], reverse=True)


# ------------------------------------------------------------
# 평가/시각화: 학습정책 vs P-PGDPO 사영, (있으면) 폐형해와 비교
# ------------------------------------------------------------
@torch.no_grad()
def print_policy_rmse_and_samples_run(
    pol_s1: nn.Module,
    pol_cf: nn.Module | None,
    *,
    repeats: int,
    sub_batch: int,
    seed_eval: int | None = None,
    outdir: str | None = None,
    tile: int | None = None,
    enable_pp: bool = True,         # pp(P-PGDPO) 계산 여부
    prefix: str = "run",            # 파일명 접두어 ("run"/"base" 등)
) -> None:
    """
    Variance-reduced 'run' 모드 평가기.
    - u_learn = pol_s1(**states)
    - (옵션) u_pp = ppgdpo_u_run(...)  (타일링으로 OOM 방지; enable_pp=False면 생략)
    - (있으면) u_cf = pol_cf(**states)
    - RMSE 출력 + (겹침) 히스토그램/산점도 저장
    """
    # RNG & states
    gen = make_generator(seed_eval or CRN_SEED_EU)
    states_dict, _ = sample_initial_states(N_eval_states, rng=gen)

    # 학습 정책 출력
    u_learn = pol_s1(**states_dict)

    # (옵션) P-PGDPO 사영 정책 계산
    u_pp_run = None
    if enable_pp:
        B = next(iter(states_dict.values())).size(0)
        divisors = _divisors_desc(B)
        start_idx = next((i for i, d in enumerate(divisors) if d <= (tile or B)), 0)
        u_pp_run = torch.empty_like(u_learn)
        for idx in range(start_idx, len(divisors)):
            cur_tile = divisors[idx]
            try:
                for s in range(0, B, cur_tile):
                    e = min(B, s + cur_tile)
                    tile_states = {k: v[s:e] for k, v in states_dict.items()}
                    u_pp_run[s:e] = ppgdpo_u_run(
                        pol_s1, tile_states, repeats, sub_batch,
                        seed_eval=(seed_eval if seed_eval is not None else 0)
                    )
                break  # 성공했으면 종료
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if idx + 1 < len(divisors):
                        print(f"[Eval] OOM; reducing tile -> {divisors[idx+1]}")
                        continue
                    else:
                        print("[Eval] OOM; could not reduce tile further.")
                        raise
                else:
                    raise

    # 폐형해가 있으면 로드
    u_cf = pol_cf(**states_dict) if pol_cf is not None else None

    # RMSE 출력 및 메트릭 저장
    if u_cf is not None:
        rmse_learn = torch.sqrt(((u_learn - u_cf) ** 2).mean()).item()
        print(f"[Policy RMSE] ||u_learn - u_closed-form||_RMSE: {rmse_learn:.6f}")
        if enable_pp and (u_pp_run is not None):
            rmse_pp = torch.sqrt(((u_pp_run - u_cf) ** 2).mean()).item()
            print(f"[Policy RMSE-PP({prefix})] ||u_pp({prefix}) - u_closed-form||_RMSE: {rmse_pp:.6f}")
        if outdir is not None:
            metrics = {f"rmse_learn_cf_{prefix}": rmse_learn}
            if enable_pp and (u_pp_run is not None):
                metrics[f"rmse_pp_cf_{prefix}"] = rmse_pp
            append_metrics_csv(metrics, outdir)
    else:
        print("[INFO] No closed-form policy provided for comparison.")

    # 그림 저장 (겹침 히스토그램 + 산점도)
    if outdir is not None:
        save_overlaid_delta_hists(
            u_learn=u_learn, u_pp=u_pp_run, u_cf=u_cf,
            outdir=outdir, coord=0,
            fname=f"delta_{prefix}_overlaid_hist.png", bins=60, density=True
        )
        save_pairwise_scatters(
            u_learn=u_learn, u_pp=u_pp_run, u_cf=u_cf,
            outdir=outdir, coord=0, prefix=f"scatter_{prefix}"
        )

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
        if u_cf is not None and enable_pp and (u_pp_run is not None):
            print(f"  ({sstr}) -> (u_learn[0]={u_learn[i,0].item():.4f}, u_pp({prefix})[0]={u_pp_run[i,0].item():.4f}, u_cf[0]={u_cf[i,0].item():.4f}, ...)")
        elif u_cf is not None:
            print(f"  ({sstr}) -> (u_learn[0]={u_learn[i,0].item():.4f}, u_cf[0]={u_cf[i,0].item():.4f}, ...)")
        else:
            if enable_pp and (u_pp_run is not None):
                print(f"  ({sstr}) -> (u_learn[0]={u_learn[i,0].item():.4f}, u_pp({prefix})[0]={u_pp_run[i,0].item():.4f}, ...)")
            else:
                print(f"  ({sstr}) -> (u_learn[0]={u_learn[i,0].item():.4f}, ...)")


# ------------------------------------------------------------
# 학습 루프(Variance-Reduced) + 손실 저장(outdir)
# ------------------------------------------------------------
def train_stage1_run(
    epochs_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    seed_train: Optional[int] = None,
    outdir: Optional[str] = None,
) -> nn.Module:
    """
    Variance-Reduced 시뮬레이터로 Stage-1 정책 학습.
    """
    _epochs = int(epochs_override if epochs_override is not None else epochs)
    _lr = float(lr_override if lr_override is not None else lr)

    # 정책/옵티마이저
    try:
        from user_pgdpo_base import DirectPolicy  # 일부 모델이 커스텀 DirectPolicy를 둘 수 있음
    except Exception as e:
        raise RuntimeError(f"[pgdpo_run] DirectPolicy not found in user_pgdpo_base: {e}")

    policy = DirectPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=_lr)

    loss_hist = []
    for ep in range(1, _epochs + 1):
        opt.zero_grad()
        U = simulate_run(policy, batch_size, seed_local=(seed_train or 0) + ep)
        loss = -U.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            print(f"[{ep:04d}] loss={loss.item():.6f}")
        loss_hist.append(float(loss.item()))

    if outdir is not None:
        try:
            from viz import save_loss_curve, save_loss_csv
            save_loss_csv(loss_hist, outdir, "loss_history_run.csv")
            save_loss_curve(loss_hist, outdir, "loss_curve_run.png")
        except Exception as e:
            print(f"[WARN] run: could not save loss plots: {e}")

    return policy


# ------------------------------------------------------------
# 독립 실행용 (테스트)
# ------------------------------------------------------------
def main():
    # base 학습 → run 평가 (독립 테스트 실행)
    run_common(
        train_fn=train_stage1_run,
        rmse_fn=print_policy_rmse_and_samples_run,
        seed_train=seed,
        train_kwargs={},  # 필요시 {"epochs_override":..., "lr_override":..., "outdir":...}
        rmse_kwargs={"repeats": REPEATS, "sub_batch": SUBBATCH, "seed_eval": CRN_SEED_EU},
    )


if __name__ == "__main__":
    main()


__all__ = [
    "simulate_run",
    "ppgdpo_u_run",
    "print_policy_rmse_and_samples_run",
    "train_stage1_run",
]