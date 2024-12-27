from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator
from mean_field_tools.deep_bsde.filtration import Filtration

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

FILTRATION = Filtration(
    spatial_dimensions=SPATIAL_DIMENSIONS,
    time_domain=TIME_DOMAIN,
    number_of_paths=NUMBER_OF_PATHS,
    seed=0,
)


class OperatorApproximator(FunctionApproximator):
    def forward(self, x):
        out = super().forward(x)
        out = torch.cumsum(out, dim=1)
        return out


dt = FILTRATION.dt

f = lambda x: torch.cumsum((x**2) * dt, dim=1)

B_t = 100 * FILTRATION.brownian_process[:5000, :, :]

fB_t = f(B_t)


W_t = 100 * FILTRATION.brownian_process[5000:, :, :]
fW_t = f(W_t)

u_hat = OperatorApproximator(domain_dimension=1, output_dimension=1)

training_strategy_args = {
    "batch_size": 500,
    "number_of_iterations": 10_000,
    "number_of_batches": 10_000,
}
u_hat.minimize_over_sample(
    sample=B_t, target=fB_t, training_strategy_args=training_strategy_args
)

fW_hat = u_hat(W_t)

import pdb

pdb.set_trace()
