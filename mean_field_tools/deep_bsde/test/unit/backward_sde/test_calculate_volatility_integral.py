"""Regression test: CommonNoiseBackwardSDE.calculate_volatility_integral
uses the correct filtration attribute names and returns the right shape.
"""

import torch
from mean_field_tools.deep_bsde.filtration import CommonNoiseFiltration
from mean_field_tools.deep_bsde.forward_backward_sde import CommonNoiseBackwardSDE
from mean_field_tools.deep_bsde.utils import IDENTITY_TERMINAL

torch.manual_seed(42)

TIME_DOMAIN = torch.linspace(0, 1, 21)
N_PATHS = 100
D = 1

FILTRATION = CommonNoiseFiltration(
    spatial_dimensions=D,
    time_domain=TIME_DOMAIN,
    number_of_paths=N_PATHS,
    common_noise_coefficient=0.3,
    seed=42,
)

FILTRATION.forward_process = FILTRATION.brownian_process

BSDE = CommonNoiseBackwardSDE(
    terminal_condition_function=IDENTITY_TERMINAL,
    filtration=FILTRATION,
)
BSDE.initialize_approximator()
BSDE.initialize_z_approximator()


def test_calculate_volatility_integral_no_attribute_error():
    """Should not raise AttributeError due to wrong increment names."""
    BSDE.set_drift_path()
    integral = BSDE.calculate_volatility_integral()
    assert integral is not None


def test_calculate_volatility_integral_shape():
    """Output shape should be (N_PATHS, L, D)."""
    BSDE.set_drift_path()
    integral = BSDE.calculate_volatility_integral()
    L = len(TIME_DOMAIN)
    assert integral.shape == (N_PATHS, L, D)


def test_calculate_volatility_integral_terminal_zero():
    """Terminal value of the integral should be zero."""
    BSDE.set_drift_path()
    integral = BSDE.calculate_volatility_integral()
    assert torch.allclose(integral[:, -1, :], torch.zeros(N_PATHS, D))
