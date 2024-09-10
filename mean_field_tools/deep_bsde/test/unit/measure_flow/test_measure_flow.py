from mean_field_tools.deep_bsde.measure_flow import MeasureFlow
from mean_field_tools.deep_bsde.filtration import Filtration
from mean_field_tools.deep_bsde.utils import L_2_norm

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


def mean_variance_parametrization(paths):
    mean = torch.mean(paths, dim=0)
    var = torch.var(paths, dim=0)

    one = torch.ones_like(paths)

    parameters_along_paths = one * torch.cat([mean, var], dim=-1)

    return parameters_along_paths


def test_default_parameters():
    measure_flow = MeasureFlow(filtration=FILTRATION)
    mean_field_parametrization = measure_flow.parameterize(FILTRATION.brownian_process)

    deviation = L_2_norm(mean_field_parametrization)
    assert deviation < 5 * 1e-4


def test_changing_parametrization_method():
    measure_flow = MeasureFlow(
        filtration=FILTRATION, parametrization=mean_variance_parametrization
    )

    mean_field_parametrization = measure_flow.parameterize(FILTRATION.brownian_process)

    t = FILTRATION.time_process
    zero = torch.zeros_like(t)

    benchmark = torch.cat([zero, t], dim=-1)

    deviation = L_2_norm(mean_field_parametrization - benchmark)

    assert deviation < 5 * 1e-4
