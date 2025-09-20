# 파일: core/pgdpo_residual.py

from __future__ import annotations
from typing import Optional
import importlib
import torch
import torch.nn as nn
import torch.optim as optim

# d, k, DIM_X, DIM_Y, DIM_U를 모두 import하여 유연성을 확보합니다.
from pgdpo_base import (
    device, run_common,
    d, k, DIM_X, DIM_Y, DIM_U,
    seed,
    epochs, batch_size
)

# 고성능 러너 import는 그대로 유지합니다.
from pgdpo_run import simulate_run, print_policy_rmse_and_samples_run
from pgdpo_with_projection import REPEATS, SUBBATCH, VERBOSE, SAMPLE_PREVIEW_N

try:
    from user_pgdpo_residual import MyopicPolicy, ResCap
except Exception as e:
    raise RuntimeError(f"[pgdpo_residual] Failed to import symbols from user_pgdpo_residual: {e}")

# ========================= 잔차 래퍼 (수정됨) =========================
class ResidualPolicy(nn.Module):
    """
    u_final(state) = [u_base(state) + δu_NN(state), c_base(state)]
    - 잔차(residual)는 포트폴리오(u)에만 적용됩니다.
    - 소비(c)는 베이스 정책의 값을 그대로 사용합니다.
    """
    def __init__(self, base_policy: nn.Module):
        super().__init__()
        self.base = base_policy

        state_dim = DIM_X + DIM_Y + 1
        # 잔차 네트워크의 출력은 d (투자 차원)로 유지합니다.
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, d) # 출력은 d차원
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.4)
                nn.init.zeros_(m.bias)

    def forward(self, **states_dict):
        with torch.no_grad():
            # base_output은 (B, d+1) 또는 (B, d) 형태
            base_output = self.base(**states_dict)

        X   = states_dict.get('X')
        TmT = states_dict.get('TmT')
        Y   = states_dict.get('Y')
        if X is None or TmT is None:
            raise ValueError("ResidualPolicy requires states: X and TmT (and optional Y).")

        feats = [X, TmT]
        if Y is not None:
            feats.append(Y)
        x_in = torch.cat(feats, dim=1)

        # 잔차는 d차원으로 계산됩니다.
        delta_u = ResCap * self.net(x_in)

        # --- 핵심 수정 로직 ---
        is_consumption_model = base_output.size(1) > d

        if is_consumption_model:
            # 소비 모델인 경우, 출력을 분리하고 투자에만 잔차를 더한 후 다시 결합합니다.
            base_u = base_output[:, :d]
            base_c = base_output[:, d:]
            
            final_u = base_u + delta_u
            final_output = torch.cat([final_u, base_c], dim=1)
        else:
            # 소비가 없는 모델은 기존과 동일하게 동작합니다.
            final_output = base_output + delta_u
            
        return final_output

# ========================= 학습 루프 (변경 없음) =========================
def train_residual_stage1(
    *,
    epochs: int = epochs,
    lr: float = 1e-3,
    seed_train: Optional[int] = seed,
    outdir: Optional[str] = None,
) -> nn.Module:
    """
    MyopicPolicy(사용자) 고정 + ResidualPolicy(코어)만 학습.
    simulate_run으로 분산-축소 U 추정.
    """
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

    opt = optim.Adam(policy.net.parameters(), lr=lr)
    assert all(not p.requires_grad for p in base_policy.parameters()), "[cv] base_policy 파라미터가 requires_grad=True 입니다."
    assert any(p.requires_grad for p in policy.net.parameters()), "[cv] policy.net 파라미터가 requires_grad=False 입니다."
    print("[residual] using ResidualPolicy over MyopicPolicy (net-only optimization)")

    loss_hist = []
    policy.train()
    for ep in range(1, int(epochs) + 1):
        opt.zero_grad()
        pair_seed = None if seed_train is None else int(seed_train) + ep
        U_pol = simulate_run(policy, B=batch_size, seed_local=pair_seed)
        loss = -U_pol.mean()

        loss.backward()
        nn.utils.clip_grad_norm_(policy.net.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            print(f"[{ep:04d}] loss={loss.item():.6f}")
        loss_hist.append(float(loss.item()))

    policy.eval()

    if outdir is not None:
        try:
            from viz import save_loss_curve, save_loss_csv
            save_loss_csv(loss_hist, outdir, "loss_history_residual.csv")
            save_loss_curve(loss_hist, outdir, "loss_curve_residual.png")
        except Exception as e:
            print(f"[WARN] residual: could not save loss plots: {e}")

    return policy

# ========================= 실행기 (변경 없음) =========================
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