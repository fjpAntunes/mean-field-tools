from mean_field_tools.deep_bsde.forward_backward_sde import (
    Filtration,
    BackwardSDE,
    NumericalForwardSDE,
    AnalyticForwardSDE,
    ForwardBackwardSDE,
)
from mean_field_tools.deep_bsde.artist import (
    cast_to_np,
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


K = 1


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


def ZERO_FUNCTION(filtration: Filtration):
    zero = torch.zeros_like(filtration.time_process)

    return zero


def ONE_FUNCTION(filtration: Filtration):
    one = torch.ones_like(filtration.time_process)

    return one


analytic_forward_sde = AnalyticForwardSDE(
    filtration=FILTRATION,
    functional_form=OU_FUNCTIONAL_FORM,
)

numeric_forward_sde = NumericalForwardSDE(
    filtration=FILTRATION,
    initial_value=ZERO_FUNCTION(FILTRATION),
    drift=lambda f: -K * f.forward_process,
    volatility=ONE_FUNCTION,
    tolerance=1e-6,
)

FILTRATION.forward_process = FILTRATION.time_process

numeric_forward_sde.solve()

paths = numeric_forward_sde.generate_paths()

analytic = OU_FUNCTIONAL_FORM(FILTRATION)

artist = PicardIterationsArtist(
    filtration=FILTRATION, analytical_forward_solution=OU_FUNCTIONAL_FORM
)


error_x, _, _ = artist.calculate_errors()


t = cast_to_np(FILTRATION.time_domain)

import matplotlib.pyplot as plt

_, axs = plt.subplots(1, 1, figsize=(12, 5))


quantile_values = [0.5, 0.8, 0.95]
for value in quantile_values:

    # Violin plot
    artist.violin_plot(axs, t, error_x, value)

axs.legend()
axs.grid(True)
axs.set_title(f"Quantile of errors along time")
axs.set_ylabel(r"$(\hat X - X)$")
axs.set_xlabel("Time")
plt.savefig(f"./.figures/error_quantiles_along_time_forward_picard.png")
plt.close()
