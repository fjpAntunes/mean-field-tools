"""Unit tests for rank-4 (matrix) volatility support in NumericalForwardSDE."""

import torch
from mean_field_tools.deep_bsde.filtration import Filtration
from mean_field_tools.deep_bsde.forward_backward_sde import NumericalForwardSDE

torch.manual_seed(0)

TIME_DOMAIN = torch.linspace(0, 1, 11)
N_PATHS = 50
D = 2

FILTRATION = Filtration(
    spatial_dimensions=D,
    time_domain=TIME_DOMAIN,
    number_of_paths=N_PATHS,
    seed=0,
)
FILTRATION.forward_process = FILTRATION.brownian_process


def _matrix_volatility(filtration):
    """Returns rank-4 constant identity volatility (N, L, d, d)."""
    N = filtration.brownian_process.shape[0]
    L = filtration.brownian_process.shape[1]
    eye = torch.eye(D).unsqueeze(0).unsqueeze(0)
    return eye.expand(N, L, D, D)


def _vector_volatility(filtration):
    """Returns rank-3 constant ones volatility (N, L, d)."""
    return torch.ones_like(filtration.brownian_process)


def _zero_drift(filtration):
    return torch.zeros_like(filtration.time_process).expand_as(
        filtration.brownian_process
    )


def test_matrix_vol_output_shape():
    """Rank-4 volatility should produce (N, L, d) output paths."""
    sde = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=torch.zeros(N_PATHS, 1, D),
        drift=_zero_drift,
        volatility=_matrix_volatility,
    )
    paths = sde.generate_paths()
    assert paths.shape == (N_PATHS, len(TIME_DOMAIN), D)


def test_matrix_identity_vol_equals_vector_ones_vol():
    """Identity matrix vol should give same integral as ones vector vol."""
    sde_mat = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=torch.zeros(N_PATHS, 1, D),
        drift=_zero_drift,
        volatility=_matrix_volatility,
    )
    sde_vec = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=torch.zeros(N_PATHS, 1, D),
        drift=_zero_drift,
        volatility=_vector_volatility,
    )
    ito_mat = sde_mat.calculate_ito_integral()
    ito_vec = sde_vec.calculate_ito_integral()
    assert torch.allclose(ito_mat, ito_vec, atol=1e-6)
