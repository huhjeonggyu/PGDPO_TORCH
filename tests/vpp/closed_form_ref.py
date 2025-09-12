# tests/vpp/closed_form_ref.py
import torch
from user_pgdpo_base import T, device, R_diag, Q_diag, x_target_vec, price_fn

class ClosedFormPolicy(torch.nn.Module):
    """
    VPP LQ tracking의 폐형해:
      u*(t,x) = -(S(t)/R) (x - x_tar) - psi(t)/R + P(t)/R
    with
      S(t)   = sqrt(Q R) * tanh( sqrt(Q/R) * (T - t) ),
      psi(T) = 0,  -psi_dot = -Q x_tar + (S/R)(psi - P).
    구현:
      - 시간격자에서 S/R, P를 사용해 psi를 뒤→앞(Euler)로 적분하여 테이블화
      - forward에서 t에 대해 선형보간으로 psi(t)를 읽어 사용
    """
    def __init__(self, n_grid: int = 2001):
        super().__init__()
        # (1,d) 모양의 대각 벡터들을 버퍼로 보관
        self.register_buffer("R_diag", R_diag.view(1, -1))
        self.register_buffer("Q_diag", Q_diag.view(1, -1))
        self.register_buffer("x_tar",  x_target_vec.clone())   # (1,d)
        self.n_grid = int(n_grid)

        # 시간격자 및 외생가격 P(t) 테이블
        grid = torch.linspace(0.0, T, steps=self.n_grid, device=device)        # (G,)
        self.register_buffer("grid", grid)
        price_grid = price_fn(grid.view(-1, 1)).view(-1)                       # (G,)
        self.register_buffer("price_grid", price_grid)

        # S_over_R(t) = S(t)/R = sqrt(Q/R) * tanh( sqrt(Q/R) * (T - t) )
        sqrt_Q_over_R = torch.sqrt(self.Q_diag / self.R_diag)                  # (1,d)
        tau_grid = (T - grid).view(-1, 1)                                      # (G,1) = 잔여시간
        S_over_R_grid = sqrt_Q_over_R * torch.tanh(sqrt_Q_over_R * tau_grid)   # (G,d)
        self.register_buffer("S_over_R_grid", S_over_R_grid)

        # psi_tab[j] ≈ psi(grid[j]) (G,d) — Backward Euler 적분
        G = self.n_grid
        psi_tab = torch.zeros(G, self.R_diag.size(1), device=device)           # (G,d), psi(T)=0
        dt_grid = grid[1:] - grid[:-1]                                         # (G-1,)

        # j = G-2 ... 0 (뒤→앞)
        for j in range(G - 2, -1, -1):
            dt = dt_grid[j]
            s_over_R = S_over_R_grid[j + 1]            # (d,)
            psi_next = psi_tab[j + 1]                  # (d,)
            # -psi_dot = -Q x_tar + (S/R)(psi - P)  →  psi_j = psi_{j+1} - rhs_{j+1} * dt
            rhs = - self.Q_diag.squeeze(0) * self.x_tar.squeeze(0) \
                  + s_over_R * (psi_next - price_grid[j + 1])
            psi_tab[j] = psi_next - rhs * dt

        self.register_buffer("psi_tab", psi_tab)       # (G,d)

    @torch.no_grad()
    def forward(self, **states_dict):
        X   = states_dict["X"]          # (B,d)
        TmT = states_dict["TmT"]        # (B,1) = τ
        t   = T - TmT                   # (B,1)
        price = price_fn(t)             # (B,1)

        # S/R(t) = sqrt(Q/R) * tanh( sqrt(Q/R) * τ )
        sqrt_Q_over_R = torch.sqrt(self.Q_diag / self.R_diag)   # (1,d)
        S_over_R = sqrt_Q_over_R * torch.tanh(sqrt_Q_over_R * TmT)  # (B,d)

        # psi(t): 시간격자 선형보간
        t_flat = t.view(-1)                                     # (B,)
        idx = torch.searchsorted(self.grid, t_flat, right=True) # ∈[0,G]
        idx = idx.clamp(min=1, max=self.n_grid - 1)             # 보간용 안전한 인덱스
        t0 = self.grid[idx - 1]                                 # (B,)
        t1 = self.grid[idx]                                     # (B,)
        w  = ((t_flat - t0) / (t1 - t0 + 1e-12)).view(-1, 1)    # (B,1), 0-division 보호

        psi0 = self.psi_tab[idx - 1]                            # (B,d)
        psi1 = self.psi_tab[idx]                                # (B,d)
        psi_t = (1.0 - w) * psi0 + w * psi1                     # (B,d)

        # u*(t,x) = -(S/R)(x - x_tar) - psi/R + P/R
        u = - S_over_R * (X - self.x_tar) - psi_t / self.R_diag + price / self.R_diag
        return u


def build_closed_form_policy():
    return ClosedFormPolicy().to(device)
