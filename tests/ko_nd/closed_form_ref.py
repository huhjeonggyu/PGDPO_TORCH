# closed_form_ref.py
# 역할: 다차원 Riccati ODE를 풀어 분석적 벤치마크 해를 계산 (ver2 형식)
# 제공해주신 analytical_solver.py의 로직을 ver2 프레임워크에 맞게 재구성했습니다.

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

__all__ = ["precompute_ABC", "ClosedFormPolicy"]

# ==============================================================================
# ===== 1. Riccati ODE 시스템 정의 (Helper Functions) =====
# ==============================================================================

def _flatten_C_B(C, B):
    """행렬 C(k,k)와 벡터 B(k,)를 1차원 배열 z(k²+k,)로 변환합니다."""
    return np.concatenate([C.ravel(), B])

def _unflatten_C_B(z, k):
    """1차원 배열 z를 행렬 C와 벡터 B로 복원합니다."""
    C = z[:k * k].reshape((k, k))
    B = z[k * k:]
    return C, B

def _compute_beta_matrices(alpha, sigma, Sigma_inv, rho_Y):
    """
    β₀, β₁, β₂ 행렬을 계산합니다.
    """
    sigma_diag = np.diag(sigma)
    sigmaA = sigma_diag @ alpha
    sigma_rho = sigma_diag @ rho_Y

    beta0 = sigmaA.T @ Sigma_inv @ sigmaA
    beta1 = sigmaA.T @ Sigma_inv @ sigma_rho
    beta2 = sigma_rho.T @ Sigma_inv @ sigma_rho
    return beta0, beta1, beta2

def _ode_multifactor_ABC_backward(t, z, k, gamma, kappa_Y, sigma_Y, Phi_Y, theta_Y, betas):
    """
    Riccati ODE 시스템을 정의합니다. scipy.solve_ivp가 호출할 함수입니다.
    """
    C, B = _unflatten_C_B(z, k)
    beta0, beta1, beta2 = betas
    factor = (1.0 - gamma) / gamma

    # dC/dt 방정식
    dC = (
        kappa_Y.T @ C + C @ kappa_Y
        - C @ sigma_Y @ Phi_Y @ sigma_Y.T @ C
        - factor * (
            beta0
            + beta1 @ sigma_Y @ C
            + C @ sigma_Y.T @ beta1.T
            + C @ sigma_Y.T @ beta2 @ sigma_Y @ C
        )
    )

    # dB/dt 방정식
    dB = (
        kappa_Y.T @ B
        - C @ sigma_Y @ Phi_Y @ sigma_Y.T @ B
        - factor * (
            C @ sigma_Y.T @ beta2 @ sigma_Y @ B
            + beta1 @ sigma_Y @ B
        )
    )
    if theta_Y is not None:
        dB -= C @ kappa_Y @ theta_Y # 부호 및 전치 수정

    # solve_ivp는 순방향 적분기이므로, 역방향 ODE를 위해 -를 붙여 반환
    return -_flatten_C_B(dC, dB)


# ==============================================================================
# ===== 2. ODE 솔버 실행 (Main API) =====
# ==============================================================================

def precompute_ABC(params: dict, T: float, gamma: float):
    """
    scipy.integrate.solve_ivp를 사용하여 다차원 Riccati ODE를 수치적으로 풉니다.
    기존 precompute_BC를 대체하며, 행렬 A, B, C 중 B와 C를 계산합니다.

    Args:
        params (dict): user_pgdpo_base.py에서 생성된 모든 시장 파라미터 딕셔너리.
        T (float): 만기.
        gamma (float): 위험 회피 계수.

    Returns:
        scipy.integrate.OdeSolution: ODE의 해를 담고 있는 솔루션 객체.
    """
    # NumPy 배열로 변환 (scipy는 NumPy 기반)
    k = params['kappa_Y'].shape[0]
    kappa_Y = params['kappa_Y'].cpu().numpy()
    sigma_Y = params['sigma_Y'].cpu().numpy()
    Phi_Y = params['Phi_Y'].cpu().numpy()
    alpha = params['alpha'].cpu().numpy()
    sigma = params['sigma'].cpu().numpy()
    rho_Y = params['rho_Y'].cpu().numpy()
    theta_Y = params['theta_Y'].cpu().numpy()
    Sigma = params['Sigma'].cpu().numpy()
    Sigma_inv = np.linalg.inv(Sigma)

    # ODE에 필요한 beta 행렬 미리 계산
    betas = _compute_beta_matrices(alpha, sigma, Sigma_inv, rho_Y)

    # 터미널 조건 (t=T에서 B와 C는 0)
    C_T = np.zeros((k, k))
    B_T = np.zeros(k)
    zT = _flatten_C_B(C_T, B_T)

    # ODE 풀이 (t=T에서 시작하여 t=0 방향으로)
    sol = solve_ivp(
        fun=_ode_multifactor_ABC_backward,
        t_span=[0, T],
        y0=zT,
        method='Radau',
        dense_output=True,
        args=(k, gamma, kappa_Y, sigma_Y, Phi_Y, theta_Y, betas),
        rtol=1e-6,
        atol=1e-8
    ) #
    
    print("✅ Analytical solver has successfully computed the benchmark solution.")
    return sol


# ==============================================================================
# ===== 3. 분석적 정책 모듈 (PyTorch nn.Module) =====
# ==============================================================================

class ClosedFormPolicy(nn.Module):
    """
    ODE 해로부터 특정 상태(t, Y)에서의 최적 정책을 계산하는 nn.Module.
    ver2의 일반화된 `states_dict` 입력을 처리합니다.
    """
    def __init__(self, ode_solution, params: dict, T: float, gamma: float, u_cap: float):
        super().__init__()
        self.sol = ode_solution
        self.T = T
        self.gamma = gamma
        self.u_cap = u_cap

        # 필요한 파라미터들을 텐서로 변환하여 버퍼에 등록
        self.register_buffer("alpha", params['alpha'])
        self.register_buffer("sigma", params['sigma'])
        self.register_buffer("Sigma_inv", params['Sigma_inv'])
        self.register_buffer("rho_Y", params['rho_Y'])
        self.register_buffer("sigma_Y", params['sigma_Y'])
        
        self.d = self.sigma.shape[0]
        self.k = self.alpha.shape[1] if self.alpha.ndim > 1 else 0

    def forward(self, **states_dict):
        """
        주어진 상태 딕셔너리에 대해 최적 제어(u)를 계산합니다.

        Args:
            **states_dict (dict): 'X', 'TmT', 'Y'를 포함하는 상태 딕셔너리.

        Returns:
            torch.Tensor: 계산된 최적 제어(u).
        """
        TmT = states_dict['TmT']
        Y = states_dict.get('Y')

        if Y is None: # Merton 문제 (Y가 없는 경우)
            mu_minus_r = self.alpha # user_pgdpo_base에서 이렇게 정의했다고 가정
            myopic = self.Sigma_inv @ mu_minus_r.unsqueeze(-1)
            return torch.clamp((1.0 / self.gamma) * myopic.T, -self.u_cap, self.u_cap)

        # 1. 현재 시간에 맞는 B, C 값 추출 (NumPy)
        tau = self.T - TmT.cpu().numpy().flatten()
        z = self.sol.sol(tau).T # (batch, k*k + k)
        
        # 2. NumPy 결과를 PyTorch 텐서로 변환
        C_flat, B_flat = z[:, :self.k**2], z[:, self.k**2:]
        C = torch.from_numpy(C_flat).view(-1, self.k, self.k).to(Y.device, dtype=Y.dtype)
        B = torch.from_numpy(B_flat).view(-1, self.k, 1).to(Y.device, dtype=Y.dtype)

        # 3. Myopic 수요 계산
        # (d, k) @ (b, k, 1) -> (b, d, 1)
        myopic_term = self.alpha @ Y.unsqueeze(-1) 
        myopic_demand = (self.Sigma_inv @ (self.sigma.unsqueeze(-1) * myopic_term))
        
        # 4. Hedging 수요 계산
        # Sigma_XY = diag(sigma) @ rho_Y @ sigma_Y
        Sigma_XY = torch.diag(self.sigma) @ self.rho_Y @ self.sigma_Y
        
        # (b,k,1) + (b,k,k) @ (b,k,1) -> (b,k,1)
        hedging_inner = B + C @ Y.unsqueeze(-1)
        hedging_demand = self.Sigma_inv @ (Sigma_XY @ hedging_inner)

        # 5. 최종 정책 계산
        u = (1.0 / self.gamma) * (myopic_demand + hedging_demand).squeeze(-1)
        return torch.clamp(u, -self.u_cap, self.u_cap)