# tests/vpp/closed_form_ref.py
import torch
from user_pgdpo_base import T, device, R_diag, x_target_vec, price_fn # Q_diag는 여기서 직접 0으로 설정

class ClosedFormPolicy(torch.nn.Module):
    """
    VPP LQ pure arbitrage의 폐형해 (상태 추적 없음, Q=0):
      u*(t,x) = (P(t) - psi(t)) / R
    with
      -psi_dot = -(S/R)(psi - P).  S=0이므로 -psi_dot=0, psi(t)=const.
      psi(T)=0 이므로 psi(t)=0 for all t.
    결론:
      u*(t,x) = P(t)/R
    """
    def __init__(self, n_grid: int = 2001):
        super().__init__()
        self.register_buffer("R_diag", R_diag.view(1, -1))
        
        # ✨✨✨ 수정된 부분: Q=0이므로 S=0, psi=0 이 되어 로직이 극도로 단순화됩니다. ✨✨✨
        # 기존의 복잡한 S, psi 계산이 필요 없어집니다.
        
    @torch.no_grad()
    def forward(self, **states_dict):
        TmT = states_dict["TmT"] # (B,1) = τ
        t   = T - TmT            # (B,1)
        price = price_fn(t)      # (B,1)

        # u*(t) = P(t) / R
        u = price / self.R_diag
        return u


def build_closed_form_policy():
    return ClosedFormPolicy().to(device)