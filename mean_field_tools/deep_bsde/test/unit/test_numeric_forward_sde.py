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


def analytical_X(filtration: Filtration):
    t = filtration.time_process

    return


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


def test_initial_value():
    "Paths should match brownian paths + 1"
    numeric_forward_sde = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=ONE_FUNCTION,
        drift=ZERO_FUNCTION,
        volatility=ONE_FUNCTION,
    )

    paths = numeric_forward_sde.generate_paths()

    brownian = FILTRATION.brownian_process

    deviations = paths - brownian - 1

    assert torch.mean(deviations**2) + deviations.var() < 1e-4


def test_drift():
    "Should evaluate to X_t = t + B_t"
    numeric_forward_sde = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=ZERO_FUNCTION,
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
        initial_value=ZERO_FUNCTION,
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
        initial_value=ZERO_FUNCTION,
        drift=ONE_FUNCTION,
        volatility=lambda f: 2 * f.brownian_process,
    )

    paths = numeric_forward_sde.generate_paths()

    analytic_path = FILTRATION.brownian_process**2

    deviations = paths - analytic_path

    assert torch.mean(deviations**2) + deviations.var() < 2 * 1e-2


def test_X_t_dependent_dynamics():
    "Geometric brownian motion. Should evaluate to X_t = e^(t/2 + B_t)"
    numeric_forward_sde = NumericalForwardSDE(
        filtration=FILTRATION,
        initial_value=ONE_FUNCTION,
        drift=GBM_DRIFT,
        volatility=GBM_VOL,
    )

    paths = numeric_forward_sde.generate_paths()

    t = FILTRATION.time_process
    B_t = FILTRATION.brownian_process
    analytic_path = torch.exp(t / 2 + B_t)

    deviations = paths - analytic_path

    assert torch.mean(deviations**2) + deviations.var() < 2 * 1e-2
