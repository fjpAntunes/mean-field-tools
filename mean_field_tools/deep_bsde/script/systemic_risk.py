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

from mean_field_tools.deep_bsde.measure_flow import MeasureFlow
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


XI = torch.distributions.normal.Normal(loc=0, scale=2).sample(
    sample_shape=(NUMBER_OF_PATHS, 1, SPATIAL_DIMENSIONS)
)

"Parameters"

a = 1
q = 1
SIGMA = 1
epsilon = 10
c = 1

"Forward SDE definition"


def ZERO_FUNCTION(filtration: Filtration):
    zero = torch.zeros_like(filtration.time_process)

    return zero


def FORWARD_DRIFT(filtration: Filtration):
    X_t = filtration.forward_process
    m_t = torch.mean(X_t, dim=0)
    Y_t = filtration.backward_process

    value = (a + q) * (m_t - X_t) - Y_t
    return value


def VOLATILITY(filtration: Filtration):
    one = torch.ones_like(filtration.time_process)
    return SIGMA * one


forward_sde = NumericalForwardSDE(
    filtration=FILTRATION,
    initial_value=XI,
    drift=FORWARD_DRIFT,
    volatility=VOLATILITY,
)

"Backward SDE definition"


def BACKWARD_DRIFT(filtration: Filtration):
    X_t = filtration.forward_process
    m_t = torch.mean(X_t, dim=0)
    Y_t = filtration.backward_process
    value = (a + q) * Y_t + (epsilon - q**2) * (m_t - X_t)
    return value


def TERMINAL_CONDITION(filtration: Filtration):
    X_T = filtration.forward_process[:, -1, :]
    mu_T = torch.mean(X_T, dim=0)
    value = c * (X_T - mu_T)

    return value


backward_sde = BackwardSDE(
    drift=BACKWARD_DRIFT,
    terminal_condition_function=TERMINAL_CONDITION,
    filtration=FILTRATION,
    exogenous_process=["time_process", "forward_process", "forward_mean_field"],
)
backward_sde.initialize_approximator(
    nn_args={
        "device": device,
        "optimizer": torch.optim.Adam,
    }
)

"FBSDE definition"

measure_flow = MeasureFlow(filtration=FILTRATION)

forward_backward_sde = ForwardBackwardSDE(
    filtration=FILTRATION,
    forward_sde=forward_sde,
    backward_sde=backward_sde,
    damping=lambda i: i / (1 + i),
    measure_flow=measure_flow,
)


# def ANALYTIC_SOLUTION(X_t, t, T):
#    return (Q / (1 + Q * (T - t))) * X_t + TAU / (1 + Q * (T - t))


"Solver parameters"

artist = FunctionApproximatorArtist(
    save_figures=True,
    # analytical_solution=ANALYTIC_SOLUTION
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


# def analytical_Z(filtration: Filtration):
#    t = filtration.time_process
#    T = t[:, -1].unsqueeze(-1)
#
#    return (Q / (1 + Q * (T - t))) * SIGMA
#
#
# def analytical_X(filtration: Filtration):
#    t = filtration.time_process
#    T = t[:, -1].unsqueeze(-1)
#
#    initial_conidtion_term = XI * (1 + Q * (T - t)) / (1 + Q * T)
#
#    first_term = -TAU * t / (1 + Q * T)
#    dummy_time = filtration.time_process
#    integrand = (1 / (1 + Q * (T - dummy_time)))[:, :-1, :]
#    initial = torch.zeros_like(t[:, 0, :].unsqueeze(1))
#
#    dB_u = filtration.brownian_increments
#    integral_term = torch.cat([initial, torch.cumsum(integrand * dB_u, axis=1)], dim=1)
#    second_term = SIGMA * (1 + Q * (T - t)) * integral_term
#
#    return initial_conidtion_term + first_term + second_term
#
#
# def analytical_Y(filtration: Filtration):
#    X_t = analytical_X(filtration=filtration)
#    t = filtration.time_process
#    T = t[:, -1].unsqueeze(-1)
#    return (Q * X_t + TAU) / (1 + Q * (T - t))
#

iterations_artist = PicardIterationsArtist(
    FILTRATION,
    # analytical_backward_solution=analytical_Y,
    # analytical_backward_volatility=analytical_Z,
    # analytical_forward_solution=analytical_X,
)

forward_backward_sde.backward_solve(
    number_of_iterations=10,
    plotter=iterations_artist,
    approximator_args=PICARD_ITERATION_ARGS,
)
