"""Tests Ornstein-Uhlenbeck as forward process"""

from mean_field_tools.deep_bsde.forward_backward_sde import (
    Filtration,
    BackwardSDE,
    ForwardSDE,
    ForwardBackwardSDE,
)
from mean_field_tools.deep_bsde.function_approximator import FunctionApproximatorArtist
import torch
import numpy as np

TIME_DOMAIN = torch.linspace(0, 1, 101)
NUMBER_OF_PATHS = 1000
SPATIAL_DIMENSIONS = 1

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


def BACKWARD_DRIFT(filtration: Filtration):
    X_t = filtration.forward_process

    return 2 * X_t


def TERMINAL_CONDITION(filtration: Filtration):
    X_T = filtration.forward_process[:, -1, :]

    return X_T**2


def ANALYTICAL_SOLUTION(x, t, T):
    return (
        x**2 * np.exp(-2 * K * (T - t))
        + ((1 - np.exp(-2 * K * (T - t))) / (2 * K))
        + 2 * x * ((1 - np.exp(-K * (T - t))) / K)
    )


FILTRATION = Filtration(
    spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=100, seed=0
)

forward_sde = ForwardSDE(
    filtration=FILTRATION,
    functional_form=OU_FUNCTIONAL_FORM,
)

backward_sde = BackwardSDE(
    terminal_condition_function=TERMINAL_CONDITION,
    filtration=FILTRATION,
    exogenous_process=["time_process", "brownian_process", "forward_process"],
    drift=BACKWARD_DRIFT,
)
backward_sde.initialize_approximator()

forward_backward_sde = ForwardBackwardSDE(
    filtration=FILTRATION, forward_sde=forward_sde, backward_sde=backward_sde
)


artist = FunctionApproximatorArtist(
    save_figures=True, analytical_solution=ANALYTICAL_SOLUTION
)

APPROXIMATOR_ARGS = {
    "batch_size": 100,
    "number_of_iterations": 5000,
    "number_of_epochs": 50,
    "number_of_plots": 5,
    "plotter": artist,
}

forward_backward_sde.backward_solve(approximator_args=APPROXIMATOR_ARGS)

import pdb

pdb.set_trace()
