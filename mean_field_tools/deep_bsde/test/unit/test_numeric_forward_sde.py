from mean_field_tools.deep_bsde.forward_backward_sde import (
    Filtration,
    NumericalForwardSDE,
    AnalyticForwardSDE,
    ForwardBackwardSDE,
)
import torch
import numpy as np


torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"

TIME_DOMAIN = torch.linspace(0, 1, 101)
NUMBER_OF_PATHS = 10_000
SPATIAL_DIMENSIONS = 1

FILTRATION = Filtration(
    spatial_dimensions=SPATIAL_DIMENSIONS,
    time_domain=TIME_DOMAIN,
    number_of_paths=NUMBER_OF_PATHS,
    seed=0,
)

"Forward SDE definition"


def ZERO_FUNCTION(filtration: Filtration):
    zero = torch.zeros_like(filtration.time_process)

    return zero


def ONE_FUNCTION(filtration: Filtration):
    one = torch.ones_like(filtration.time_process)

    return one


def GBM_DRIFT(filtration: Filtration):
    X_t = filtration.forward_process

    return X_t


def GBM_VOL(filtration: Filtration):
    X_t = filtration.forward_process
    return X_t


def test_calculate_riemman_integral():
    forward_sde = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=ZERO_FUNCTION,
        drift=ONE_FUNCTION,
        volatility=ZERO_FUNCTION,
    )

    riemman_integral = forward_sde.calculate_riemman_integral()

    deviation = FILTRATION.time_process - riemman_integral
    assert torch.mean(deviation**2) < 1e-3


def test_calculate_ito_integral():
    forward_sde = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=ZERO_FUNCTION,
        drift=ZERO_FUNCTION,
        volatility=lambda f: f.time_process,
    )

    ito_integral = forward_sde.calculate_ito_integral()

    deviation = torch.mean(ito_integral**2 - FILTRATION.time_process**2, axis=0)

    assert torch.mean(deviation**2) < 0.2


def test_initial_value():
    "Paths should match brownian paths + 1"
    numeric_forward_sde = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=ONE_FUNCTION(FILTRATION),
        drift=ZERO_FUNCTION,
        volatility=ONE_FUNCTION,
    )

    paths = numeric_forward_sde.generate_paths()

    brownian = FILTRATION.brownian_process

    deviations = paths - brownian - 1

    assert torch.mean(deviations**2) + deviations.var() < 1e-4


def test_random_initial_value():
    "Paths should evaluate to xi + B_t"
    initial_condition_size = (NUMBER_OF_PATHS, 1, SPATIAL_DIMENSIONS)
    XI = torch.distributions.normal.Normal(loc=1, scale=2).sample(
        sample_shape=initial_condition_size
    )
    numeric_forward_sde = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=XI,
        drift=ZERO_FUNCTION,
        volatility=ONE_FUNCTION,
    )

    paths = numeric_forward_sde.generate_paths()

    brownian = FILTRATION.brownian_process

    deviations = paths - brownian - XI

    assert torch.mean(deviations**2) + deviations.var() < 1e-4


def test_drift():
    "Should evaluate to X_t = t + B_t"
    numeric_forward_sde = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=ZERO_FUNCTION(FILTRATION),
        drift=ONE_FUNCTION,
        volatility=ONE_FUNCTION,
    )

    paths = numeric_forward_sde.generate_paths()

    analytic_path = FILTRATION.brownian_process + FILTRATION.time_process

    deviations = paths - analytic_path

    assert torch.mean(deviations**2) + deviations.var() < 1e-4


def test_vol():
    "Should evaluate to X_t = 2B_t"
    numeric_forward_sde = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=ZERO_FUNCTION(FILTRATION),
        drift=ZERO_FUNCTION,
        volatility=lambda f: 2 * ONE_FUNCTION(f),
    )

    paths = numeric_forward_sde.generate_paths()

    analytic_path = 2 * FILTRATION.brownian_process

    deviations = paths - analytic_path

    assert torch.mean(deviations**2) + deviations.var() < 1e-4


def test_brownian_dependent_dynamics():
    "Should evaluate to X_t = B_t^2"
    numeric_forward_sde = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=ZERO_FUNCTION(FILTRATION),
        drift=ONE_FUNCTION,
        volatility=lambda f: 2 * f.brownian_process,
    )

    paths = numeric_forward_sde.generate_paths()

    analytic_path = FILTRATION.brownian_process**2

    deviations = paths - analytic_path

    assert torch.mean(deviations**2) + deviations.var() < 2 * 1e-2


def test_X_t_dependent_dynamics():
    """Geometric brownian motion. Should evaluate to X_t = e^(t/2 + B_t).
    Tolerance for test assertion can be made arbitraily small by refining time discretization.
    """
    numeric_forward_sde = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=ONE_FUNCTION(FILTRATION),
        drift=GBM_DRIFT,
        volatility=GBM_VOL,
    )

    numeric_forward_sde.solve()
    paths = numeric_forward_sde.generate_paths()

    t = FILTRATION.time_process
    B_t = FILTRATION.brownian_process
    analytic_path = torch.exp(t / 2 + B_t)

    deviations = paths - analytic_path

    assert torch.mean(deviations**2) + deviations.var() < 1 * 1e-1


K = 2


def OU_FUNCTIONAL_FORM(filtration):
    dummy_time = filtration.time_process[:, 1:, 0].unsqueeze(-1)
    integrand = torch.exp(K * dummy_time) * filtration.brownian_increments

    initial = torch.zeros(
        size=(filtration.number_of_paths, 1, filtration.spatial_dimensions)
    )
    integral = torch.cat([initial, torch.cumsum(integrand, dim=1)], dim=1)

    time = filtration.time_process[:, :, 0].unsqueeze(-1)
    path = torch.exp(-K * time) * integral
    return path


def test_OU_solution():
    "Ornstein-Uhlenbeck SDE. Should evaluate to X_t = X_0 e^(- k t) + int_0^t e^(-k)dW_t"

    numeric_forward_sde = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=ZERO_FUNCTION(FILTRATION),
        drift=lambda f: -K * f.forward_process,
        volatility=ONE_FUNCTION,
    )

    numeric_forward_sde.solve()
    paths = numeric_forward_sde.generate_paths()

    analytic_path = OU_FUNCTIONAL_FORM(FILTRATION)

    deviations = paths - analytic_path
    assert torch.mean(deviations**2) + deviations.var() < 1 * 1e-4
