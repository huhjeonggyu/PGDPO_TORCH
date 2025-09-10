# core/pgdpo_with_projection.py
# 역할: "Direct" P-PGDPO (코스테이트 추정 → PMP 사영) 평가/유틸
# - 호환 래퍼 없이 random_draws를 직접 생성하여 simulate(...)에 전달
# - 타일/서브배치 재현성: seed_eval + 진행 오프셋 기반 RNG

from __future__ import annotations

from typing import Optional, Dict

import torch
import torch.nn as nn

from pgdpo_base import (
    device, N_eval_states,
    make_generator, run_common, train_stage1_base,
    sample_initial_states, simulate,
    CRN_SEED_EU, m, d, k,
)

try:
    from user_pgdpo_with_projection import project_pmp
except Exception as e:
    raise RuntimeError(f"[pgdpo_with_projection] Failed to import symbols from user_pgdpo_with_projection: {e}")    

from viz import (
    save_pairwise_scatters, append_metrics_csv, save_overlaid_delta_hists
)


# 코스테이트 추정 반복 수와 서브배치 크기 (기본값)
REPEATS = 2048
SUBBATCH = 2048 // 16


# -----------------------------------------------------------------------------
# 공통 노이즈 생성 (X용 d, Y용 k 채널 분리)
# -----------------------------------------------------------------------------
def _draw_base_normals(B: int, steps: int, gen: torch.Generator):
    Z = torch.randn(B, steps, d + k, device=device, generator=gen)
    ZX, ZY = Z[:, :, :d], Z[:, :, d:]
    return ZX, ZY


# -----------------------------------------------------------------------------
# 코스테이트 추정 (자동미분, random_draws 직접 생성)
# -----------------------------------------------------------------------------
def estimate_costates(
    policy_net: nn.Module,
    initial_states: Dict[str, torch.Tensor],
    repeats: int,
    sub_batch: int,
    seed_eval: Optional[int] = None,
):
    """
    자동미분으로 on-policy 코스테이트 추정.
    시드는 random_draws를 직접 생성해서 넘김(호환 래퍼/seed_local 불필요).
    Returns: {'JX': (B,·), 'JXX': (B,·), 'JXY': (B,·)?}
    """
    B = next(iter(initial_states.values())).size(0)

    # 미분 대상 leaf 상태 (X, Y만)
    leaf_states = {
        k: v.detach().clone().requires_grad_(True)
        for k, v in initial_states.items()
        if k in ["X", "Y"]
    }
    TmT_val = initial_states["TmT"].detach().clone()  # 시간 텐서는 미분 대상 아님

    # 누적 버퍼
    costate_sums = {
        "JX": torch.zeros_like(leaf_states["X"]),
        "JXX": torch.zeros_like(leaf_states["X"]),
        "JXY": torch.zeros_like(leaf_states.get("Y")) if "Y" in leaf_states else None,
    }

    # 정책 파라미터는 고정(코스테이트 추정 시 업데이트 없음)
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

            # 타일/서브배치 재현성: seed_eval + done
            gen = make_generator((seed_eval or 0) + done)
            ZX, ZY = _draw_base_normals(B * rpts, m, gen)

            # simulate는 U=-J를 반환한다고 가정
            U = simulate(
                policy_net,
                B * rpts,
                train=True,
                initial_states_dict=batch_states,
                random_draws=(ZX, ZY),  # ← 시드 대신 명시 노이즈 전달
                m_steps=m,
            )

            # (rpts, B, ·) 평균 → (B, ·), 다시 전체 평균 스칼라
            U_bar = U.view(rpts, B, -1).mean(dim=0)
            J_scalar = U_bar.mean()

            # 1차: JX = ∂U/∂X
            JX_b = torch.autograd.grad(
                J_scalar, leaf_states["X"], retain_graph=True, create_graph=True
            )[0]

            # 2차: JXX, JXY(필요 시)
            grad_targets = [leaf_states["X"]]
            if "Y" in leaf_states:
                grad_targets.append(leaf_states["Y"])

            JXX_b, *JXY_b_tuple = torch.autograd.grad(
                JX_b.sum(),
                grad_targets,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            JXY_b = JXY_b_tuple[0] if JXY_b_tuple else None

            # 누적(서브배치 가중치 rpts)
            costate_sums["JX"] += JX_b.detach() * rpts
            costate_sums["JXX"] += (
                JXX_b.detach() if JXX_b is not None else torch.zeros_like(leaf_states["X"])
            ) * rpts
            if costate_sums["JXY"] is not None and JXY_b is not None:
                costate_sums["JXY"] += JXY_b.detach() * rpts

            done += rpts
    finally:
        for p, r in zip(params, req_bak):
            p.requires_grad_(r)

    inv = 1.0 / float(repeats)
    return {k: v * inv for k, v in costate_sums.items() if v is not None}


# -----------------------------------------------------------------------------
# Direct P-PGDPO: 코스테이트 추정 → PMP 사영
# -----------------------------------------------------------------------------
def ppgdpo_u_direct(
    policy_s1: nn.Module,
    states: Dict[str, torch.Tensor],
    repeats: int,
    sub_batch: int,
    seed_eval: Optional[int] = None,
) -> torch.Tensor:
    with torch.enable_grad():
        costates = estimate_costates(policy_s1, states, repeats, sub_batch, seed_eval=seed_eval)
        u = project_pmp(costates, states)
    return u.detach()


def _divisors_desc(n: int):
    return sorted([d for d in range(1, n + 1) if n % d == 0], reverse=True)


# -----------------------------------------------------------------------------
# 평가/시각화: 학습정책 vs P-PGDPO 사영, (있으면) 폐형해 비교
# -----------------------------------------------------------------------------
@torch.no_grad()
def print_policy_rmse_and_samples_direct(
    pol_s1: nn.Module,
    pol_cf: nn.Module | None,
    *,
    repeats: int,
    sub_batch: int,
    seed_eval: int | None = None,
    tile: int | None = None,
    outdir: str | None = None,
) -> None:
    """
    Projection/direct 모드 평가기.
    - u_learn = pol_s1(**states)
    - u_pp_dir = ppgdpo_u_direct(...) (타일링으로 OOM 방지)
    - (있으면) u_cf = pol_cf(**states)
    - RMSE 출력 + pairwise 히스토그램/산점도 저장
    """
    # RNG & states
    gen = make_generator(seed_eval or CRN_SEED_EU)
    states_dict, _ = sample_initial_states(N_eval_states, rng=gen)

    # 학습 정책 출력
    u_learn = pol_s1(**states_dict)

    # 타일 사이즈 결정(+OOM 대비)
    B = next(iter(states_dict.values())).size(0)
    divisors = _divisors_desc(B)
    start_idx = next((i for i, d in enumerate(divisors) if d <= (tile or B)), 0)

    # P-PGDPO(direct) 액션 계산 (타일 순회)
    u_pp_dir = torch.empty_like(u_learn)
    for idx in range(start_idx, len(divisors)):
        cur_tile = divisors[idx]
        try:
            for s in range(0, B, cur_tile):
                e = min(B, s + cur_tile)
                tile_states = {k: v[s:e] for k, v in states_dict.items()}
                u_pp_dir[s:e] = ppgdpo_u_direct(
                    pol_s1, tile_states, repeats, sub_batch, seed_eval=seed_eval
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
        rmse_pp_dir = torch.sqrt(((u_pp_dir - u_cf) ** 2).mean()).item()
        print(f"[Policy RMSE] ||u_learn - u_closed-form||_RMSE: {rmse_learn:.6f}")
        print(f"[Policy RMSE-PP(direct)] ||u_pp(direct) - u_closed-form||_RMSE: {rmse_pp_dir:.6f}")
        if outdir is not None:
            append_metrics_csv(
                {"rmse_learn_cf_direct": rmse_learn, "rmse_pp_cf_direct": rmse_pp_dir}, outdir
            )
    else:
        print("[INFO] No closed-form policy provided for comparison.")

    # 히스토그램/산점도 저장
    if outdir is not None:
        save_overlaid_delta_hists(
            u_learn=u_learn, u_pp=u_pp_dir, u_cf=u_cf,
            outdir=outdir, coord=0, fname="delta_projection_overlaid_hist.png",
            bins=60, density=True
        )
        save_pairwise_scatters(
            u_learn=u_learn, u_pp=u_pp_dir, u_cf=u_cf,
            outdir=outdir, coord=0, prefix="scatter_projection"
        )

    # 대표 샘플 3개 출력
    idxs = [0, B // 2, B - 1]
    for i in idxs:
        state_parts, vector_found = [], False
        for key, value in states_dict.items():
            ts = value[i]
            if ts.numel() > 1:
                state_parts.append(f"{key}[0]={ts[0].item():.3f}")
                vector_found = True
            else:
                state_parts.append(f"{key}={ts.item():.3f}")
        if vector_found:
            state_parts.append("...")
        state_str = ", ".join(state_parts)

        if u_cf is not None:
            print(
                f"  ({state_str}) -> (u_learn[0]={u_learn[i,0].item():.4f}, "
                f"u_pp_dir[0]={u_pp_dir[i,0].item():.4f}, "
                f"u_cf[0]={u_cf[i,0].item():.4f}, ...)"
            )
        else:
            print(
                f"  ({state_str}) -> (u_learn[0]={u_learn[i,0].item():.4f}, "
                f"u_pp_dir[0]={u_pp_dir[i,0].item():.4f}, ...)"
            )



# -----------------------------------------------------------------------------
# 독립 실행 테스트
# -----------------------------------------------------------------------------
def main():
    # base 학습 → direct 평가 (간이 테스트)
    run_common(
        train_fn=train_stage1_base,
        rmse_fn=print_policy_rmse_and_samples_direct,
        train_kwargs={},  # 필요 시 epochs/lr override 가능
        rmse_kwargs={"repeats": REPEATS, "sub_batch": SUBBATCH, "seed_eval": CRN_SEED_EU},
    )


if __name__ == "__main__":
    main()


__all__ = [
    "REPEATS",
    "SUBBATCH",
    "estimate_costates",
    "project_pmp",
    "ppgdpo_u_direct",
    "print_policy_rmse_and_samples_direct",
]