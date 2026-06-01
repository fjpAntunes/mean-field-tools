"""Integration tests for the general (matrix-coefficient) LQ MV-FBSDE.

Three tests:
1. Reduction to isotropic — matrix path reproduces scalar results.
2. Riccati correctness  — ODE residual ≈ 0, terminal condition exact,
                          commuting closed-form matches numerical N.
3. Coupled benchmark    — end-to-end Picard solver with non-commuting matrices.
"""

import numpy as np
import pytest
import torch

from mean_field_tools.deep_bsde.filtration import CommonNoiseFiltration, Filtration
from mean_field_tools.deep_bsde.forward_backward_sde import (
    CommonNoiseBackwardSDE,
    ForwardBackwardSDE,
    NumericalForwardSDE,
)
from mean_field_tools.deep_bsde.measure_flow import CommonNoiseMeasureFlow
from mean_field_tools.deep_bsde.script.experiments.matrix_lq.analytic import (
    commuting_riccati,
    compute_Z,
    compute_mean_field,
    integrate_riccati,
    interpolate_N,
)
from mean_field_tools.deep_bsde.utils import L_2_norm

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Shared scalar parameters (isotropic baseline)
# ---------------------------------------------------------------------------
a, q, sigma_scalar, epsilon, c = 2.0, 1.0, 1.0, 10.0, 0.1
RHO = 0.3
T_END = 1.0


# ---------------------------------------------------------------------------
# Helper: scalar niu (closed-form)
# ---------------------------------------------------------------------------
def scalar_niu(t_arr, T=T_END):
    alpha = a + q
    beta = epsilon - q**2
    disc = (alpha**2 + beta) ** 0.5
    dp = -alpha + disc
    dm = -alpha - disc
    exp = np.exp((dp - dm) * (T - t_arr))
    A_ = exp - 1
    Bp = dp * exp - dm
    Bm = dm * exp - dp
    return (-beta * A_ - c * Bp) / (Bm - c * A_)


# ===========================================================================
# Test 1 — Reduction to isotropic
# ===========================================================================
class TestReductionToIsotropic:
    D = 2
    T_STEPS = 51

    @pytest.fixture(scope="class")
    def riccati_isotropic(self):
        D = self.D
        I = np.eye(D)
        AQ = (a + q) * I
        EQ2 = (epsilon - q**2) * I
        C_mat = c * I
        t_np = np.linspace(0.0, T_END, self.T_STEPS)
        N_t = integrate_riccati(AQ, EQ2, C_mat, t_np)
        return N_t, t_np

    def test_N_equals_nu_times_identity(self, riccati_isotropic):
        D = self.D
        N_t, t_np = riccati_isotropic
        nu_t = scalar_niu(t_np)
        for i, t in enumerate(t_np):
            expected = nu_t[i] * np.eye(D)
            np.testing.assert_allclose(N_t[i], expected, atol=1e-4)

    def test_Z_is_diagonal(self, riccati_isotropic):
        D = self.D
        N_t, _ = riccati_isotropic
        I = np.eye(D)
        Sigma = sigma_scalar * I
        Z_t = compute_Z(N_t, RHO, Sigma)
        for i in range(len(N_t)):
            off_diag = Z_t[i] - np.diag(np.diag(Z_t[i]))
            np.testing.assert_allclose(off_diag, np.zeros((D, D)), atol=1e-4)


# ===========================================================================
# Test 2 — Riccati correctness (no learning)
# ===========================================================================
class TestRiccatiCorrectness:
    D = 2
    T_STEPS = 201

    # Non-commuting symmetric matrices
    AQ = np.array([[3.0, 1.0], [1.0, 2.0]])
    EQ2 = np.array([[5.0, 0.5], [0.5, 8.0]])
    C_mat = np.array([[0.2, 0.05], [0.05, 0.15]])

    @pytest.fixture(scope="class")
    def N_t_and_time(self):
        t_np = np.linspace(0.0, T_END, self.T_STEPS)
        N_t = integrate_riccati(self.AQ, self.EQ2, self.C_mat, t_np)
        return N_t, t_np

    def test_terminal_condition(self, N_t_and_time):
        N_t, _ = N_t_and_time
        np.testing.assert_allclose(N_t[-1], self.C_mat, atol=1e-4)

    def test_ode_residual(self, N_t_and_time):
        N_t, t_np = N_t_and_time
        dt = t_np[1] - t_np[0]
        AQ, EQ2 = self.AQ, self.EQ2
        # finite-difference N'(t) ≈ (N(t+dt) - N(t-dt)) / (2 dt)
        residuals = []
        for i in range(1, len(t_np) - 1):
            Ni = N_t[i]
            dN_dt_fd = (N_t[i + 1] - N_t[i - 1]) / (2 * dt)
            rhs = AQ @ Ni + Ni @ AQ + Ni @ Ni - EQ2
            residuals.append(np.linalg.norm(dN_dt_fd - rhs, "fro"))
        assert max(residuals) < 0.01, f"Max ODE residual: {max(residuals):.6f}"

    def test_commuting_case_matches_numerical(self):
        """Diagonal matrices commute; per-eigenvalue formula must match integrate_riccati."""
        D = self.D
        I = np.eye(D)
        AQ_comm = np.diag([3.0, 2.0])
        EQ2_comm = np.diag([5.0, 8.0])
        C_comm = np.diag([0.2, 0.15])
        t_np = np.linspace(0.0, T_END, self.T_STEPS)
        N_num = integrate_riccati(AQ_comm, EQ2_comm, C_comm, t_np)
        N_closed = commuting_riccati(AQ_comm, EQ2_comm, C_comm, t_np)
        np.testing.assert_allclose(N_num, N_closed, atol=1e-4)


# ===========================================================================
# Test 3 — Coupled benchmark (end-to-end)
# ===========================================================================
@pytest.mark.slow
class TestCoupledBenchmark:
    """Non-commuting matrix LQ system; asserts deep solver matches analytic N."""

    D = 2
    N_PATHS = 5_000
    T_STEPS = 51
    RHO = RHO

    # Non-commuting symmetric matrices
    AQ = np.array([[3.0, 1.0], [1.0, 2.0]])
    EQ2 = np.array([[5.0, 0.5], [0.5, 8.0]])
    C_mat = np.array([[0.2, 0.05], [0.05, 0.15]])
    Sigma_np = np.array([[1.0, 0.3], [0.2, 0.8]])

    @pytest.fixture(scope="class")
    def solved_system(self):
        D = self.D
        N_PATHS = self.N_PATHS
        T_STEPS = self.T_STEPS
        rho = self.RHO

        t_np = np.linspace(0.0, T_END, T_STEPS)
        TIME_DOMAIN = torch.tensor(t_np, dtype=torch.float32)

        AQ = self.AQ
        EQ2 = self.EQ2
        C_mat = self.C_mat
        Sigma_np = self.Sigma_np

        # Analytic Riccati
        N_t = integrate_riccati(AQ, EQ2, C_mat, t_np)
        N_t_torch = torch.tensor(N_t, dtype=torch.float32)
        Sigma_t = torch.tensor(Sigma_np, dtype=torch.float32)
        AQ_t = torch.tensor(AQ, dtype=torch.float32)

        Z_t = compute_Z(N_t, rho, Sigma_np)  # (T, d, d)

        torch.manual_seed(0)
        FILTRATION = CommonNoiseFiltration(
            spatial_dimensions=D,
            time_domain=TIME_DOMAIN,
            number_of_paths=N_PATHS,
            common_noise_coefficient=rho,
            seed=0,
        )

        XI = torch.distributions.Normal(loc=0.0, scale=1.0).sample((N_PATHS, 1, D))

        xi_mean = XI.mean(dim=0, keepdim=True)  # (1, 1, D)

        def FORWARD_DRIFT(filtration: Filtration):
            X_t = filtration.forward_process
            m_t = filtration.forward_mean_field
            Y_t = filtration.backward_process
            return torch.einsum("ij,nlj->nli", AQ_t, m_t - X_t) - Y_t

        def FORWARD_VOL(filtration: Filtration):
            N, L, _ = filtration.brownian_process.shape
            return Sigma_t.unsqueeze(0).unsqueeze(0).expand(N, L, D, D)

        def BACKWARD_DRIFT(filtration: Filtration):
            X_t = filtration.forward_process
            m_t = filtration.forward_mean_field
            Y_t = filtration.backward_process
            EQ2_t = torch.tensor(EQ2, dtype=torch.float32)
            term1 = torch.einsum("ij,nlj->nli", AQ_t, Y_t)
            term2 = torch.einsum("ij,nlj->nli", EQ2_t, m_t - X_t)
            return -(term1 + term2)

        def TERMINAL_CONDITION(filtration: Filtration):
            X_T = filtration.forward_process[:, -1, :]
            m_T = filtration.forward_mean_field[:, -1, :]
            C_t = torch.tensor(C_mat, dtype=torch.float32)
            return torch.einsum("ij,nj->ni", C_t, X_T - m_T)

        forward_sde = NumericalForwardSDE(
            filtration=FILTRATION,
            initial_value=XI,
            drift=FORWARD_DRIFT,
            volatility=FORWARD_VOL,
        )

        backward_sde = CommonNoiseBackwardSDE(
            drift=BACKWARD_DRIFT,
            terminal_condition_function=TERMINAL_CONDITION,
            filtration=FILTRATION,
            exogenous_process=["time_process", "forward_process", "forward_mean_field"],
            z_is_matrix=True,
        )

        nn_args_y = {
            "number_of_layers": 2,
            "number_of_nodes": 24,
            "optimizer": torch.optim.Adam,
            "optimizer_params": {"lr": 0.005},
        }
        nn_args_z = {
            "number_of_layers": 1,
            "number_of_nodes": 8,
            "optimizer": torch.optim.Adam,
            "optimizer_params": {"lr": 0.005},
        }

        backward_sde.initialize_approximator(nn_args=nn_args_y)
        backward_sde.initialize_z_approximator(nn_args=nn_args_z)

        measure_flow = CommonNoiseMeasureFlow(filtration=FILTRATION)
        measure_flow.initialize_approximator(
            nn_args={
                "number_of_layers": 2,
                "number_of_nodes": 18,
                "optimizer": torch.optim.Adam,
            },
            training_args={
                "training_strategy_args": {
                    "batch_size": 512,
                    "number_of_iterations": 200,
                    "number_of_batches": 5,
                }
            },
        )

        fbsde = ForwardBackwardSDE(
            filtration=FILTRATION,
            forward_sde=forward_sde,
            backward_sde=backward_sde,
            measure_flow=measure_flow,
            damping=lambda i: 0.5,
        )

        picard_args = {
            "training_strategy_args": {
                "batch_size": 512,
                "number_of_iterations": 200,
                "number_of_batches": 5,
            }
        }

        fbsde.backward_solve(
            number_of_iterations=5,
            approximator_args=picard_args,
        )

        return (
            FILTRATION,
            N_t_torch,
            Z_t,
            Sigma_t,
            backward_sde,
            xi_mean,
            rho,
            TIME_DOMAIN,
        )

    def test_Y_matches_analytic(self, solved_system):
        FILTRATION, N_t_torch, Z_t, Sigma_t, backward_sde, xi_mean, rho, TIME_DOMAIN = (
            solved_system
        )
        X_hat = FILTRATION.forward_process
        m_hat = FILTRATION.forward_mean_field
        Y_hat = FILTRATION.backward_process

        N_interp = interpolate_N(
            N_t_torch.numpy(), TIME_DOMAIN, FILTRATION.time_process
        )
        Y_analytic = torch.einsum("nlij,nlj->nli", N_interp, X_hat - m_hat)

        err = L_2_norm(Y_hat - Y_analytic)
        assert err < 2.0, f"Y L2 error: {err:.4f}"

    def test_Z_offdiagonal_nonzero(self, solved_system):
        """Off-diagonal entries of the analytic Z must be non-negligible."""
        _, _, Z_t, _, _, _, _, _ = solved_system
        off_diag_rms = np.sqrt(np.mean(Z_t[:, 0, 1] ** 2 + Z_t[:, 1, 0] ** 2))
        assert off_diag_rms > 0.01, "Analytic Z off-diagonal should be non-zero"

    def test_Z0_approximator_near_zero(self, solved_system):
        FILTRATION, _, _, _, backward_sde, _, _, _ = solved_system
        Z0_hat = backward_sde.generate_common_noise_volatility()
        err = L_2_norm(Z0_hat)
        assert err < 2.0, f"Z0 L2 norm: {err:.4f}"
