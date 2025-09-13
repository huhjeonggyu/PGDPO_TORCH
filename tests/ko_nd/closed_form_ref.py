# closed_form_ref.py
# 역할: 다차원 Riccati ODE를 풀어 분석적 벤치마크 해를 계산 (utility-scaled u^2 패널티, full 버전)

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

__all__ = ["precompute_ABC", "ClosedFormPolicy"]

# ==============================================================================
# ===== 1. Riccati ODE 시스템 정의 =====
# ==============================================================================

def _flatten_C_B(C, B):
    return np.concatenate([C.ravel(), B])

def _unflatten_C_B(z, k):
    C = z[:k * k].reshape((k, k))
    B = z[k * k:]
    return C, B

def _compute_beta_matrices(alpha, sigma, Sigma_reg_inv, rho_Y):
    """
    beta0 = (σA)^T Γ (σA)
    beta1 = (σA)^T Γ (σρ)      # 이후 σ_Y와 곱해져 (σA)^T Γ M C 항으로 사용
    beta2 = (σρ)^T Γ (σρ)      # 이후 σ_Y와 곱해져 C M^T Γ M C 항으로 사용
    """
    sigma_diag = np.diag(sigma)
    sigmaA = sigma_diag @ alpha          # (n x k) = diag(σ) A
    sigma_rho = sigma_diag @ rho_Y       # (n x r) = diag(σ) ρ_Y

    beta0 = sigmaA.T @ Sigma_reg_inv @ sigmaA
    beta1 = sigmaA.T @ Sigma_reg_inv @ sigma_rho
    beta2 = sigma_rho.T @ Sigma_reg_inv @ sigma_rho
    return beta0, beta1, beta2

def _ode_multifactor_ABC_backward(t, z, k, kappa_Y, sigma_Y, Phi_Y, theta_Y, betas):
    """
    Backward(terminal→initial) 적분을 위해 반환값에 음수를 붙여 solve_ivp를 [0,T] 전방 적분으로 변환.
    -dot C = C Q_z C + (kappa^T C + C kappa) + (σA + M C)^T Γ (σA + M C)
    -dot B = kappa^T B + C Q_z θ + (σA + M C)^T Γ M B
    """
    C, B = _unflatten_C_B(z, k)
    beta0, beta1, beta2 = betas

    # Q_z = σ_Y Φ_Y σ_Y^T  (k x k)
    Qz = sigma_Y @ Phi_Y @ sigma_Y.T

    # (σA + M C)^T Γ (σA + M C) 전개 항
    #  = beta0 + beta1 σ_Y C + C σ_Y^T beta1^T + C σ_Y^T beta2 σ_Y C
    cross_quad = (
        beta0
        + beta1 @ sigma_Y @ C
        + C @ sigma_Y.T @ beta1.T
        + C @ sigma_Y.T @ beta2 @ sigma_Y @ C
    )

    # dC_raw = C Qz C + (kappa^T C + C kappa) + cross_quad
    dC = (C @ Qz @ C) + (kappa_Y.T @ C + C @ kappa_Y) + cross_quad

    # dB_raw = kappa^T B + C Qz θ + [(σA + M C)^T Γ M] B
    #        = kappa^T B + C Qz θ + (beta1 σ_Y + C σ_Y^T beta2 σ_Y) B
    dB = (kappa_Y.T @ B) + (C @ Qz @ theta_Y) + (beta1 @ sigma_Y @ B) + (C @ sigma_Y.T @ beta2 @ sigma_Y @ B)

    # Backward integration trick: return negative to integrate from T→0 with t_span=[0,T]
    return -_flatten_C_B(dC, dB)

# ==============================================================================
# ===== 2. ODE 솔버 실행 =====
# ==============================================================================

def precompute_ABC(params: dict, T: float, gamma: float, epsilon: float = 1e-3):
    """
    Utility-scaled quadratic penalty를 반영한 Riccati ODE 해를 풉니다.
    Γ = (γ·Σ + ε·I)^{-1} 를 상수로 사용합니다.  # [FIX] docstring 업데이트
    """
    k = params['kappa_Y'].shape[0]
    kappa_Y = params['kappa_Y'].cpu().numpy()
    sigma_Y = params['sigma_Y'].cpu().numpy()
    Phi_Y   = params['Phi_Y'].cpu().numpy()
    alpha   = params['alpha'].cpu().numpy()    # A (n x k)
    sigma   = params['sigma'].cpu().numpy()    # σ (n,)
    rho_Y   = params['rho_Y'].cpu().numpy()    # ρ_Y (r x r) 또는 (r x r_eff)
    theta_Y = params['theta_Y'].cpu().numpy()
    Sigma   = params['Sigma'].cpu().numpy()    # Σ (n x n)

    # Γ = (γ Σ + ε I)^{-1}  # [FIX] 그대로 사용 (상수)
    Sigma_reg = gamma * Sigma + epsilon * np.eye(Sigma.shape[0])
    Sigma_reg_inv = np.linalg.inv(Sigma_reg)

    betas = _compute_beta_matrices(alpha, sigma, Sigma_reg_inv, rho_Y)

    C_T = np.zeros((k, k))
    B_T = np.zeros(k)
    zT = _flatten_C_B(C_T, B_T)

    sol = solve_ivp(
        fun=_ode_multifactor_ABC_backward,
        t_span=[0, T],            # backward를 위해 fun에서 부호 반전
        y0=zT,                    # terminal condition at T, but with backward trick
        method='Radau',
        dense_output=True,
        args=(k, kappa_Y, sigma_Y, Phi_Y, theta_Y, betas),
        rtol=1e-6,
        atol=1e-8
    )
    print("✅ Analytical solver (utility-scaled u^2 penalty) computed the benchmark solution.")
    return sol

# ==============================================================================
# ===== 3. 분석적 정책 모듈 =====
# ==============================================================================

class ClosedFormPolicy(nn.Module):
    """
    Riccati ODE 해로부터 최적 정책을 계산하는 모듈.
    Utility-scaled quadratic penalty: Γ = (γ·Σ + ε·I)^{-1}.
    """
    def __init__(self, ode_solution, params: dict, T: float, gamma: float, epsilon: float = 1e-3):
        super().__init__()
        self.sol = ode_solution
        self.T = T
        self.gamma = gamma
        self.epsilon = epsilon

        # 파라미터 등록
        self.register_buffer("alpha",   params['alpha'])     # A (n x k)
        self.register_buffer("sigma",   params['sigma'])     # σ (n,)
        self.register_buffer("rho_Y",   params['rho_Y'])     # ρ_Y
        self.register_buffer("sigma_Y", params['sigma_Y'])   # σ_Y (k x r)

        self.d = self.sigma.shape[0]
        self.k = self.alpha.shape[1] if self.alpha.ndim > 1 else 0

        # Γ = (γ Σ + ε I)^{-1}
        Sigma = params['Sigma']
        Sigma_reg = gamma * Sigma + epsilon * torch.eye(Sigma.shape[0], device=Sigma.device, dtype=Sigma.dtype)
        self.register_buffer("Sigma_reg_inv", torch.linalg.inv(Sigma_reg))

    def forward(self, **states_dict):
        """
        입력:
          - TmT: time-to-maturity (tensor), solver의 backward trick과 정합
          - Y:   factor state z_t (tensor of shape [..., k])
        출력:
          - u*:  [..., n]
        """
        TmT = states_dict['TmT']
        Y = states_dict.get('Y')
        if Y is None:
            raise ValueError("ClosedFormPolicy requires factor state 'Y' (z_t).")  # [FIX] Y 없이 사용 금지

        # backward trick 정합: solver는 0→T로 적분하되, 반환은 T-τ를 의미.
        tau = TmT.detach().cpu().numpy().flatten()
        tau = np.clip(tau, 0.0, float(self.T))
        z = self.sol.sol(tau).T  # shape: [batch, k^2 + k]

        C_flat, B_flat = z[:, :self.k**2], z[:, self.k**2:]
        C = torch.from_numpy(C_flat).view(-1, self.k, self.k).to(Y.device, dtype=Y.dtype)   # [..., k, k]
        B = torch.from_numpy(B_flat).view(-1, self.k, 1).to(Y.device, dtype=Y.dtype)       # [..., k, 1]

        # myopic: σ A z
        myopic_term = self.alpha @ Y.unsqueeze(-1)                                         # [..., n, 1] if alpha is (n x k)
        myopic_vec  = torch.diag(self.sigma) @ myopic_term                                 # diag(σ) (A z)

        # hedging: M (b + C z), M = diag(σ) ρ_Y σ_Y
        Sigma_XY = torch.diag(self.sigma) @ self.rho_Y @ self.sigma_Y                      # (n x k) if σ_Y: (k x r)
        hedging_inner = B + C @ Y.unsqueeze(-1)                                            # [..., k, 1]
        hedging_vec   = Sigma_XY @ hedging_inner                                           # [..., n, 1]

        # u* = Γ [ σ A z + M (b + C z) ]  # [FIX] 1/γ 스케일 제거
        u_star = (self.Sigma_reg_inv @ (myopic_vec + hedging_vec)).squeeze(-1)
        return u_star
