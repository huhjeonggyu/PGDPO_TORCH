# tests/vpp/closed_form_ref.py
import torch
from math import sqrt, tanh
from user_pgdpo_base import T, device, R_diag, Q_diag, x_target_vec, price_fn

@torch.no_grad()
def S_i_t(Qi, Ri, tau):
    # S_i(t) = sqrt(Q_i R_i) * tanh( sqrt(Q_i/R_i)*(T-t) )  (tau = T - t)
    return sqrt(Qi*Ri) * tanh(sqrt(Qi/Ri)*tau)

@torch.no_grad()
def psi_i_t(Qi, Ri, tau, price_grid, time_grid):
    # -dot(psi) = -Qi x_tar + (S/Ri)(psi - P), psi(T)=0
    # 수치 적분(뒤에서 앞으로) - 간단한 Euler로 충분
    dt = time_grid[1]-time_grid[0]
    psi = 0.0
    tpsi = 0.0
    for j in range(len(time_grid)-1, -1, -1):
        s_over_R = S_i_t(Qi, Ri, T - time_grid[j]) / Ri
        rhs = -Qi*x_target_vec[0,0] + s_over_R*(psi - float(price_grid[j]))
        psi = psi - rhs * dt
    return psi

class ClosedFormPolicy(torch.nn.Module):
    def __init__(self, n_grid=2001):
        super().__init__()
        self.register_buffer("R_diag", R_diag.view(1, -1))
        self.register_buffer("Q_diag", Q_diag.view(1, -1))
        self.n_grid = n_grid
        # 미리 시간/가격 그리드 만들고 채널별 S, ψ 테이블도 만들어 캐싱해도 됨 (간단화를 위해 런타임 계산 가능)

    def forward(self, **states_dict):
        X = states_dict["X"]          # (B,d)
        t = T - states_dict["TmT"]    # (B,1)
        price = price_fn(t)           # (B,1)

        # 채널별 S_i(t), ψ_i(t) 계산 (실전에서는 사전 보간 테이블 사용 권장)
        # 여기서는 근사로 S만 즉시 평가, ψ는 0으로 두어도 비교는 가능하지만 논문 1:1엔 ψ 필요
        # 간단 버전(ψ=0 근사):
        tau = t                        # (B,1)
        S = torch.sqrt(self.Q_diag*self.R_diag) * torch.tanh(torch.sqrt(self.Q_diag/self.R_diag)*tau)
        u = -(S/self.R_diag) * (X - x_target_vec) + price/self.R_diag  # ψ/R 항 생략 버전

        return u

def build_closed_form_policy():
    return ClosedFormPolicy().to(device)