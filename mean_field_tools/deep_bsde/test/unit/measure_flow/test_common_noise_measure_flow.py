from mean_field_tools.deep_bsde.measure_flow import CommonNoiseMeasureFlow
from mean_field_tools.deep_bsde.filtration import CommonNoiseFiltration
from mean_field_tools.deep_bsde.utils import L_2_norm

import torch
import numpy as np


torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
NUMBER_OF_TIMESTEPS = 101
NUMBER_OF_PATHS = 10_000
TIME_DOMAIN = torch.linspace(0, 1, NUMBER_OF_TIMESTEPS)
SPATIAL_DIMENSIONS = 1

FILTRATION = CommonNoiseFiltration(
    spatial_dimensions=SPATIAL_DIMENSIONS,
    time_domain=TIME_DOMAIN,
    number_of_paths=NUMBER_OF_PATHS,
    common_noise_coefficient=0.3,
    seed=0,
)

measure_flow = CommonNoiseMeasureFlow(filtration=FILTRATION)
measure_flow.initialize_approximator(
    training_args={
        "training_strategy_args": {
            "batch_size": 512,
            "number_of_iterations": 100,
            "number_of_batches": 100,
        }
    },
)


def test_set_elicitability_input():
    input = measure_flow._set_elicitability_input()

    assert input.shape == (NUMBER_OF_PATHS, NUMBER_OF_TIMESTEPS, 1 + SPATIAL_DIMENSIONS)


def test_parameterize_time_process():
    "Should evaluate to t"
    conditional_mean = measure_flow.parameterize(FILTRATION.time_process)
    t = FILTRATION.time_process

    deviation = conditional_mean - t
    assert L_2_norm(deviation) < 1e-4


def test_parameterize():
    "Should evaluate to rho * common_noise"
    conditional_mean = measure_flow.parameterize(FILTRATION.brownian_process)
    rho = FILTRATION.common_noise_coefficient
    common_noise = FILTRATION.common_noise

    deviation = conditional_mean - rho * common_noise
    assert L_2_norm(deviation) < 1
