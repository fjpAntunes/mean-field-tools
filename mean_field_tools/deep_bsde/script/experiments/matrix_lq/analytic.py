"""Analytical benchmark for the general (matrix-coefficient) LQ MV-FBSDE.

The system is:
    dX_t = [(A+Q)(m_t - X_t) - Y_t] dt + Sigma dB_t
    dY_t = [(A+Q) Y_t + (E - Q^2)(m_t - X_t)] dt + Z_t dW_t + Z0_t dW0_t
    Y_T  = C (X_T - m_T)

Decoupling field: Y_t = N(t)(X_t - m_t), where N solves the matrix Riccati ODE:
    N'(t) = (A+Q) N + N (A+Q) + N^2 - (E - Q^2),   N(T) = C

Analytical results:
    Z(t)  = sqrt(1 - rho^2) * N(t) * Sigma
    Z0(t) = 0
    m_t   = E[xi] + rho * Sigma * W0_t
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp


def _scalar_niu(t, T, alpha, beta, c):
    """Scalar Riccati solution: n' = 2*alpha*n + n^2 - beta, n(T)=c.

    alpha = eigenvalue of (A+Q), beta = eigenvalue of (E-Q^2).
    """
    disc = (alpha**2 + beta) ** 0.5
    dp = -alpha + disc
    dm = -alpha - disc
    exp = np.exp((dp - dm) * (T - t))
    A_ = exp - 1
    Bp = dp * exp - dm
    Bm = dm * exp - dp
    return (-beta * A_ - c * Bp) / (Bm - c * A_)


def integrate_riccati(AQ, EQ2, C, time_domain):
    """Integrate the matrix Riccati ODE backward from T to 0.

    N'(t) = (A+Q) N + N (A+Q) + N^2 - (E - Q^2),   N(T) = C

    Args:
        AQ:          numpy array (d, d) — the matrix A+Q
        EQ2:         numpy array (d, d) — the matrix E-Q^2
        C:           numpy array (d, d) — terminal condition N(T) = C
        time_domain: 1-D array of time points in [0, T], increasing

    Returns:
        N_t: numpy array (len(time_domain), d, d)
    """
    d = AQ.shape[0]
    T = float(time_domain[-1])

    def rhs(s, n_flat):
        # tau = T - t, so d(N)/d(tau) = -N'(t) = -(AQ N + N AQ + N^2 - EQ2)
        N = n_flat.reshape(d, d)
        dN_dt = AQ @ N + N @ AQ + N @ N - EQ2
        return (-dN_dt).ravel()

    t_np = np.asarray(time_domain)
    tau_eval = T - t_np[::-1]  # tau = T - t, increasing from 0

    sol = solve_ivp(
        rhs,
        [0.0, T],
        C.ravel(),
        t_eval=tau_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
    )
    # sol.y shape: (d*d, len(tau_eval)); each column is N at tau
    N_tau = sol.y.T.reshape(-1, d, d)  # (len(tau_eval), d, d)
    # reverse back so index 0 = t=0
    return N_tau[::-1].copy()


def commuting_riccati(AQ, EQ2, C, time_domain):
    """Compute N(t) via per-eigenvalue scalar Riccati (requires commuting matrices).

    Valid when AQ, EQ2, C are simultaneously diagonalizable (e.g., they all commute).
    Returns numpy array (len(time_domain), d, d).
    """
    # Diagonalize AQ (symmetric assumed)
    eigvals, O = np.linalg.eigh(AQ)
    # Project EQ2 and C into eigenbasis
    EQ2_diag = np.diag(O.T @ EQ2 @ O)
    C_diag = np.diag(O.T @ C @ O)

    T = float(time_domain[-1])
    t_np = np.asarray(time_domain)

    N_diag = np.zeros((len(t_np), len(eigvals)))
    for k, (alpha, beta, ck) in enumerate(zip(eigvals, EQ2_diag, C_diag)):
        N_diag[:, k] = np.array([_scalar_niu(t, T, alpha, beta, ck) for t in t_np])

    # Reconstruct N(t) = O diag(n_k(t)) O^T
    N_t = np.einsum("ij,tj,kj->tik", O, N_diag, O)
    return N_t


def compute_Z(N_t, rho, Sigma):
    """Compute Z(t) = sqrt(1 - rho^2) * N(t) @ Sigma.

    Args:
        N_t:   numpy (T, d, d)
        rho:   scalar
        Sigma: numpy (d, d)

    Returns:
        numpy (T, d, d)
    """
    coeff = np.sqrt(1.0 - rho**2)
    return coeff * np.einsum("tij,jk->tik", N_t, Sigma)


def compute_mean_field(xi_mean, rho, Sigma, time_domain, common_noise):
    """m_t = E[xi] + rho * Sigma @ W0_t (broadcast over paths).

    Args:
        xi_mean:      torch (1, 1, d) or (d,)  — mean of initial condition
        rho:          scalar
        Sigma:        torch (d, d)
        time_domain:  1-D tensor (L,)
        common_noise: torch (N, L, d) — common Brownian path W0

    Returns:
        torch (N, L, d)
    """
    return xi_mean + rho * torch.einsum("ij,nlj->nli", Sigma, common_noise)


def interpolate_N(N_t, time_domain, t_query):
    """Linear interpolation of N_t at a batch of query times.

    Args:
        N_t:         numpy (L, d, d)
        time_domain: numpy or tensor (L,)
        t_query:     torch (N, L', 1) — query times

    Returns:
        torch (N, L', d, d)
    """
    t_np = np.asarray(time_domain)
    t_q = t_query.detach().cpu().numpy().squeeze(-1)  # (N, L')
    d = N_t.shape[1]

    idx = np.searchsorted(t_np, t_q, side="right") - 1
    idx = np.clip(idx, 0, len(t_np) - 2)

    t0 = t_np[idx]  # (N, L')
    t1 = t_np[idx + 1]
    dt = np.where(t1 - t0 > 1e-15, t1 - t0, 1.0)
    alpha = (t_q - t0) / dt  # (N, L')

    N0 = N_t[idx]  # (N, L', d, d)
    N1 = N_t[idx + 1]
    N_interp = N0 + alpha[..., None, None] * (N1 - N0)

    return torch.tensor(N_interp, dtype=torch.float32)


def simulate_X(N_t, m_t, xi, rho, Sigma, filtration):
    """High-resolution Euler–Maruyama simulation of X using Y = N(X-m).

    Args:
        N_t:        numpy (L, d, d)
        m_t:        torch (N, L, d)
        xi:         torch (N, 1, d)  — initial condition X_0
        rho:        scalar
        Sigma:      torch (d, d)
        filtration: CommonNoiseFiltration with idiosyncratic_noise_increments

    Returns:
        torch (N, L, d)  — simulated X paths
    """
    time_domain = filtration.time_domain
    dt = filtration.dt
    idio_inc = filtration.idiosyncratic_noise_increments  # (N, L-1, d)
    common_inc = filtration.common_noise_increments  # (N, L-1, d)
    dB = rho * common_inc + (1 - rho**2) ** 0.5 * idio_inc  # (N, L-1, d)

    N_torch = torch.tensor(N_t, dtype=torch.float32)  # (L, d, d)
    Sigma_t = (
        Sigma
        if isinstance(Sigma, torch.Tensor)
        else torch.tensor(Sigma, dtype=torch.float32)
    )

    N_paths = xi.shape[0]
    L = len(time_domain)
    d = xi.shape[-1]

    X = torch.zeros(N_paths, L, d)
    X[:, 0, :] = xi[:, 0, :]

    for i in range(L - 1):
        Ni = N_torch[i]  # (d, d)
        Xi = X[:, i, :]  # (N, d)
        mi = m_t[:, i, :]  # (N, d)
        Yi = torch.einsum("ij,nj->ni", Ni, Xi - mi)  # (N, d)
        drift = -Yi  # simplified drift contribution from Y
        diffusion = torch.einsum("ij,nj->ni", Sigma_t, dB[:, i, :])
        X[:, i + 1, :] = Xi + drift * dt + diffusion

    return X
