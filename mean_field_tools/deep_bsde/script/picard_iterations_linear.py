from mean_field_tools.deep_bsde.forward_backward_sde import (
    Filtration,
    BackwardSDE,
    ForwardSDE,
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

K = 0
VOL = 1


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


def OU_VOLATILITY(filtration: Filtration):
    one = torch.ones_like(filtration.time_process)
    return VOL * one


forward_sde = ForwardSDE(
    filtration=FILTRATION,
    functional_form=OU_FUNCTIONAL_FORM,
    volatility_functional_form=OU_VOLATILITY,
)

"Backward SDE definition"

alpha = 1
beta = 1


def LINEAR_BACKWARD_DRIFT(filtration: Filtration):
    X_t = filtration.forward_process
    Y_t = filtration.backward_process
    Z_t = filtration.backward_volatility
    return alpha * Y_t + beta * Z_t  # + 2 * X_t


def LINEAR_TERMINAL_CONDITION(filtration: Filtration):
    X_T = filtration.forward_process[:, -1, :]

    return X_T


backward_sde = BackwardSDE(
    terminal_condition_function=LINEAR_TERMINAL_CONDITION,
    filtration=FILTRATION,
    exogenous_process=["time_process", "forward_process"],
    drift=LINEAR_BACKWARD_DRIFT,
)
backward_sde.initialize_approximator(nn_args={"device": device})

"FBSDE definition"

forward_backward_sde = ForwardBackwardSDE(
    filtration=FILTRATION, forward_sde=forward_sde, backward_sde=backward_sde
)


def ANALYTIC_SOLUTION(X_t, t, T):
    return np.exp(alpha * (T - t)) * (beta * (T - t) + X_t)


"Solver parameters"

artist = FunctionApproximatorArtist(
    save_figures=True, analytical_solution=ANALYTIC_SOLUTION
)

PICARD_ITERATION_ARGS = {
    "training_strategy_args": {
        "batch_size": 512,
        "number_of_iterations": 100,
        "number_of_batches": 100,
        "plotter": artist,
        "number_of_plots": 1,
    },
}


"Solving"


def analytical_Y(filtration: Filtration):
    X_t = filtration.forward_process
    t = filtration.time_process
    T = t[:, -1].unsqueeze(-1)
    return torch.exp(alpha * (T - t)) * (beta * (T - t) + X_t)


def analytical_Z(filtration: Filtration):
    t = filtration.time_process
    T = t[:, -1].unsqueeze(-1)

    return torch.exp(alpha * (T - t))


iterations_artist = PicardIterationsArtist(
    FILTRATION,
    analytical_backward_solution=analytical_Y,
    analytical_backward_volatility=analytical_Z,
)

forward_backward_sde.backward_solve(
    number_of_iterations=5,
    plotter=iterations_artist,
    approximator_args=PICARD_ITERATION_ARGS,
)
