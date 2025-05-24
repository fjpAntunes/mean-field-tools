from mean_field_tools.deep_bsde.function_approximator import OperatorApproximator
from mean_field_tools.deep_bsde.filtration import CommonNoiseFiltration, Filtration

from mean_field_tools.deep_bsde.measure_flow import CommonNoiseMeasureFlow
import torch
import numpy as np
import matplotlib.pyplot as plt


torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"

COMMON_NOISE_COEFFICIENT = 0.3
NUMBER_OF_TIMESTEPS = 101
NUMBER_OF_PATHS = 10_000
SPATIAL_DIMENSIONS = 1

TIME_DOMAIN = torch.linspace(0, 1, NUMBER_OF_TIMESTEPS)

FILTRATION = CommonNoiseFiltration(
    spatial_dimensions=SPATIAL_DIMENSIONS,
    time_domain=TIME_DOMAIN,
    number_of_paths=NUMBER_OF_PATHS,
    common_noise_coefficient=COMMON_NOISE_COEFFICIENT,
    seed=0,
)


"measure flow definition"

ALPHA = 0.6

measure_flow = CommonNoiseMeasureFlow(filtration=FILTRATION)


def quantile_scoring(x, y, alpha):
    first_term = torch.where(x >= y, 1, 0) - alpha
    second_term = x - y

    return first_term * second_term


gru = OperatorApproximator(
    input_size=FILTRATION.spatial_dimensions + 1,
    scoring=lambda x, y: quantile_scoring(x, y, ALPHA),
)

measure_flow.initialize_approximator(
    approximator=gru,
    training_args={
        "training_strategy_args": {
            "batch_size": 2048,
            "number_of_iterations": 1000,
            "number_of_batches": 1000,
        }
    },
)


"Evaluation"

normal = torch.distributions.Normal(0, 1)


def probit(alpha):
    a = torch.Tensor([alpha])
    return normal.icdf(a)


def normal_quantile(mean, std, alpha):
    return mean + std * probit(alpha)


quantiles = measure_flow.parameterize(FILTRATION.brownian_process)

expected_quantiles = normal_quantile(
    COMMON_NOISE_COEFFICIENT * FILTRATION.common_noise,
    FILTRATION.time_process**0.5,
    ALPHA,
)

error = quantiles - expected_quantiles

print("error L_2 norm: ", (error.mean() ** 2 + error.var()) ** 0.5)
