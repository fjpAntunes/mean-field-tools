"Zero initial condition case"
from mean_field_tools.deep_bsde.forward_backward_sde import (
    Filtration,
    BackwardSDE,
    NumericalForwardSDE,
    ForwardBackwardSDE,
)
from mean_field_tools.deep_bsde.artist import (
    FunctionApproximatorArtist,
    PicardIterationsArtist,
)
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

"Forward SDE definition"


def ZERO_FUNCTION(filtration: Filtration):
    zero = torch.zeros_like(filtration.time_process)

    return zero


XI = ZERO_FUNCTION(FILTRATION)


Q = 1
TAU = 1
SIGMA = 1


def DRIFT(filtration: Filtration):
    Y_t = filtration.backward_process

    return -Y_t


def VOLATILITY(filtration: Filtration):
    one = torch.ones_like(filtration.time_process)
    return SIGMA * one


forward_sde = NumericalForwardSDE(
    filtration=FILTRATION,
    initial_value=XI,
    drift=DRIFT,
    volatility=VOLATILITY,
)

"Backward SDE definition"


def TERMINAL_CONDITION(filtration: Filtration):
    X_T = filtration.forward_process[:, -1, :]

    return Q * X_T + TAU


backward_sde = BackwardSDE(
    terminal_condition_function=TERMINAL_CONDITION,
    filtration=FILTRATION,
    exogenous_process=["time_process", "forward_process"],
    drift=ZERO_FUNCTION,
)
backward_sde.initialize_approximator(
    nn_args={
        "device": device,
        "optimizer": torch.optim.Adam,
    }
)

"FBSDE definition"

forward_backward_sde = ForwardBackwardSDE(
    filtration=FILTRATION, forward_sde=forward_sde, backward_sde=backward_sde
)


def ANALYTIC_SOLUTION(X_t, t, T):
    return (Q / (1 + Q * (T - t))) * X_t + TAU / (1 + Q * (T - t))


"Solver parameters"


PICARD_ITERATION_ARGS = {
    "training_strategy_args": {
        "batch_size": 512,
        "number_of_iterations": 100,
        "number_of_batches": 100,
        "number_of_plots": 1,
    },
}


"Solving"


def analytical_Z(filtration: Filtration):
    t = filtration.time_process
    T = t[:, -1].unsqueeze(-1)

    return (Q / (1 + Q * (T - t))) * SIGMA


def analytical_X(filtration: Filtration):
    t = filtration.time_process
    T = t[:, -1].unsqueeze(-1)

    initial_conidtion_term = XI * (1 + Q * (T - t)) / (1 + Q * T)

    first_term = -TAU * t / (1 + Q * T)
    dummy_time = filtration.time_process
    integrand = (1 / (1 + Q * (T - dummy_time)))[:, :-1, :]
    initial = torch.zeros_like(t[:, 0, :].unsqueeze(1))

    dB_u = filtration.brownian_increments
    integral_term = torch.cat([initial, torch.cumsum(integrand * dB_u, axis=1)], dim=1)
    second_term = SIGMA * (1 + Q * (T - t)) * integral_term

    return initial_conidtion_term + first_term + second_term


def analytical_Y(filtration: Filtration):
    X_t = analytical_X(filtration=filtration)
    t = filtration.time_process
    T = t[:, -1].unsqueeze(-1)
    return (Q * X_t + TAU) / (1 + Q * (T - t))


def test_coupled_fbsde_zero_initial_condition():

    forward_backward_sde.backward_solve(
        number_of_iterations=10,
        approximator_args=PICARD_ITERATION_ARGS,
    )

    X = analytical_X(FILTRATION)
    Y = analytical_Y(FILTRATION)
    Z = analytical_Z(FILTRATION)

    X_hat = FILTRATION.forward_process
    Y_hat = FILTRATION.backward_process
    Z_hat = FILTRATION.backward_volatility

    deviation = L_2_norm(X - X_hat) + L_2_norm(Y - Y_hat) + L_2_norm(Z - Z_hat)

    assert deviation < 0.015
