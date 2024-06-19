"""Tests quadratic with drift"""

from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, BackwardSDE
from mean_field_tools.deep_bsde.function_approximator import FunctionApproximatorArtist
import torch

TIME_DOMAIN = torch.linspace(0, 1, 101)
NUMBER_OF_PATHS = 1000
SPATIAL_DIMENSIONS = 1

TERMINAL_CONDITION = lambda x: x**2


def DRIFT(x):
    return x[:, :, 1]


def ANALYTICAL_SOLUTION(x, t, T):
    return x**2 + (T - t) + x * (T - t)


filtration = Filtration(SPATIAL_DIMENSIONS, TIME_DOMAIN, NUMBER_OF_PATHS)

bsde = BackwardSDE(
    terminal_condition_function=TERMINAL_CONDITION,
    drift=DRIFT,
    filtration=filtration,
)

bsde.initialize_approximator()


artist = FunctionApproximatorArtist(
    save_figures=True, analytical_solution=ANALYTICAL_SOLUTION
)

bsde.solve(
    approximator_args={
        "batch_size": 100,
        "number_of_iterations": 5000,
        "number_of_epochs": 50,
        "number_of_plots": 5,
        "plotter": artist,
    }
)

import pdb

pdb.set_trace()
