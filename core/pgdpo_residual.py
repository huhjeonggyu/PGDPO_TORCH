# core/pgdpo_residual.py
# Residual 모드 트레이너 (엄격/최소 변경)
# - tests/<model>/user_pgdpo_residual.py 에 MyopicPolicy 필수
# - 학습: 코어 ResidualPolicy(잔차 래퍼)만 학습, 고성능 simulate_run 사용
# - 폴백 없음(없으면 즉시 에러)
# - epochs 는 pgdpo_base 의 epochs 를 그대로 사용

from __future__ import annotations

from typing import Optional
import importlib

import torch
import torch.nn as nn
import torch.optim as optim

from pgdpo_base import (
    device, run_common,
    d, k, DIM_X, DIM_Y,           # ✅ 기존 변수명 체계를 따라 DIM_X, DIM_Y를 가져옵니다.
    seed,
    epochs,batch_size
)

# 고성능 러너: 분산-축소 시뮬레이터 + 평가기
from pgdpo_run import simulate_run, print_policy_rmse_and_samples_run
from pgdpo_with_projection import REPEATS, SUBBATCH, VERBOSE, SAMPLE_PREVIEW_N

try :
    from user_pgdpo_residual import MyopicPolicy, ResCap
except Exception as e:
    raise RuntimeError(f"[pgdpo_residual] Failed to import symbols from user_pgdpo_residual: {e}")    


# ========================= 잔차 래퍼 =========================
class ResidualPolicy(nn.Module):
    """
    u_final(state) = u_base(state) + δu_NN(state)
    - base_policy는 고정(파라미터 업데이트 X)
    - 입력: X, (optional) Y, TmT
    - 출력: (B, d)
    """
    def __init__(self, base_policy: nn.Module):
        super().__init__()
        self.base = base_policy

        # ✅ 기존 변수명을 활용하여 상태 차원을 동적으로 계산하도록 수정
        state_dim = DIM_X + DIM_Y + 1    # X(DIM_X) + Y(DIM_Y) + TmT(1)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, d)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.4)
                nn.init.zeros_(m.bias)

    def forward(self, **states_dict):
        with torch.no_grad():
            base_u = self.base(**states_dict)  # (B, d)

        X   = states_dict.get('X')
        TmT = states_dict.get('TmT')
        Y   = states_dict.get('Y')
        if X is None or TmT is None:
            raise ValueError("ResidualPolicy requires states: X and TmT (and optional Y).")

        feats = [X, TmT]
        if Y is not None:
            feats.append(Y)
        # ✅ 주석도 실제 차원을 반영하도록 수정
        x_in = torch.cat(feats, dim=1)  # (B, DIM_X + DIM_Y + 1)

        delta_u = ResCap*torch.sigmoid( self.net(x_in) )
        u = base_u + delta_u
            
        return u
        

# ========================= 학습 루프 =========================
def train_residual_stage1(
    *,
    epochs: int = epochs,       # ✅ pgdpo_base.epochs 기본 사용
    lr: float = 1e-3,
    seed_train: Optional[int] = seed,
    outdir: Optional[str] = None,
) -> nn.Module:
    """
    MyopicPolicy(사용자) 고정 + ResidualPolicy(코어)만 학습.
    simulate_run으로 분산-축소 U 추정.
    """
    # 필수 모듈/클래스 강제
    try:
        upres = importlib.import_module("user_pgdpo_residual")
        MyopicPolicy = getattr(upres, "MyopicPolicy")
    except Exception as e:
        raise RuntimeError(
            "[Residual mode] 'tests/<model>/user_pgdpo_residual.py' import 실패 또는 "
            "'MyopicPolicy' 클래스가 없음. 원인: " + repr(e)
        ) from e

    base_policy = MyopicPolicy().to(device)
    policy = ResidualPolicy(base_policy).to(device)

    # 잔차 네트워크만 학습
    opt = optim.Adam(policy.net.parameters(), lr=lr)
    # 안전장치: 잔차 이외 파라미터는 업데이트 대상이 아님을 보장
    assert all(not p.requires_grad for p in base_policy.parameters()), "[cv] base_policy 파라미터가 requires_grad=True 입니다."
    assert any(p.requires_grad for p in policy.net.parameters()), "[cv] policy.net 파라미터가 requires_grad=False 입니다."
    print("[residual] using ResidualPolicy over MyopicPolicy (net-only optimization)")

    loss_hist = []
    policy.train()
    for ep in range(1, int(epochs) + 1):
        opt.zero_grad()
        pair_seed = None if seed_train is None else int(seed_train) + ep
        U_pol = simulate_run(policy, B=batch_size, seed_local=pair_seed)  # (B,) or (B,1)
        loss = -U_pol.mean()

        loss.backward()
        nn.utils.clip_grad_norm_(policy.net.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            print(f"[{ep:04d}] loss={loss.item():.6f}")
        loss_hist.append(float(loss.item()))

    policy.eval()

    # (옵션) 손실 저장
    if outdir is not None:
        try:
            from viz import save_loss_curve, save_loss_csv
            save_loss_csv(loss_hist, outdir, "loss_history_residual.csv")
            save_loss_curve(loss_hist, outdir, "loss_curve_residual.png")
        except Exception as e:
            print(f"[WARN] residual: could not save loss plots: {e}")

    return policy


# ========================= 실행기 =========================
def main():
    """파일 단독 실행용: 잔차 학습 + 평가"""
    from user_pgdpo_base import CRN_SEED_EU
    run_common(
        train_fn=lambda seed_train=seed: train_residual_stage1(seed_train=seed_train),
        rmse_fn=print_policy_rmse_and_samples_run,
        seed_train=seed,
        rmse_kwargs={"seed_eval": CRN_SEED_EU, "repeats": int(REPEATS), "sub_batch": int(SUBBATCH)},
    )

if __name__ == "__main__":
    main()