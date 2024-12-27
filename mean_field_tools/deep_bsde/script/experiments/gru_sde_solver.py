from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator
from mean_field_tools.deep_bsde.filtration import Filtration
from mean_field_tools.deep_bsde.forward_backward_sde import NumericalForwardSDE

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"

COMMON_NOISE_COEFFICIENT = 0.3
NUMBER_OF_TIMESTEPS = 101
NUMBER_OF_PATHS = 10_000
SPATIAL_DIMENSIONS = 1

K = 1

TIME_DOMAIN = torch.linspace(0, 1, NUMBER_OF_TIMESTEPS)

FILTRATION = Filtration(
    spatial_dimensions=SPATIAL_DIMENSIONS,
    time_domain=TIME_DOMAIN,
    number_of_paths=NUMBER_OF_PATHS,
    seed=0,
)


class OperatorApproximator(FunctionApproximator):
    def __init__(self):
        super(OperatorApproximator, self).__init__(
            domain_dimension=1, output_dimension=1
        )
        self.hidden_size = 3
        self.num_layers = 1
        self.input_size = 1
        self.gru = nn.GRU(
            self.input_size, self.hidden_size, self.num_layers, batch_first=True
        )

        self.output = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)

        # out = torch.atanh(out)
        out = self.output(out)
        out = out
        # out = super().forward(x)
        # out = torch.cumsum(out, dim=1)
        return out


dt = FILTRATION.dt


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


B_t = FILTRATION.brownian_process

fB_t = OU_FUNCTIONAL_FORM(FILTRATION)

u_hat = OperatorApproximator()  # domain_dimension=1, output_dimension=1)

training_strategy_args = {
    "batch_size": 1024,
    "number_of_iterations": 5000,
    "number_of_batches": 5000,
}


u_hat.minimize_over_sample(
    sample=B_t, target=fB_t, training_strategy_args=training_strategy_args
)

fB_t_hat = u_hat(B_t)

delta = fB_t_hat - fB_t

print((delta**2).mean())

import pdb

pdb.set_trace()
