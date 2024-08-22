from mean_field_tools.deep_bsde.forward_backward_sde import (
    Filtration,
    BackwardSDE,
    NumericalForwardSDE,
    AnalyticForwardSDE,
    ForwardBackwardSDE,
)
from mean_field_tools.deep_bsde.artist import (
    FunctionApproximatorArtist,
    PicardIterationsArtist,
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


def DRIFT(filtration: Filtration):
    Y_t = filtration.backward_process

    return -Y_t


VOL = 3


def VOLATILITY(filtration: Filtration):
    one = torch.ones_like(filtration.time_process)
    return VOL * one


analytic_forward_sde = AnalyticForwardSDE(
    filtration=FILTRATION,
    functional_form=analytical_X,
    volatility_functional_form=VOLATILITY,
)

numeric_forward_sde = NumericalForwardSDE(
    filtration=FILTRATION,
    initial_value=ZERO_FUNCTION,
    drift=ONE_FUNCTION,
    volatility=VOLATILITY,
)

paths = numeric_forward_sde.generate_paths()
