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
    tolerance=1e-6,
)

"Backward SDE definition"


def BACKWARD_DRIFT(filtration: Filtration):
    X_t = filtration.forward_process
    m_t = torch.mean(X_t, dim=0)
    Y_t = filtration.backward_process
    value = (a + q) * Y_t + (epsilon - q**2) * (m_t - X_t)
    return -value


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
    damping=lambda i: 0.5,
    measure_flow=measure_flow,
)


"Plotting Parameters"


def niu_function(t, T, module=np):
    delta_plus = -(a + q) + ((a + q) ** 2 + (epsilon - q**2)) ** 0.5
    delta_minus = -(a + q) - ((a + q) ** 2 + (epsilon - q**2)) ** 0.5
    exp = module.exp((delta_plus - delta_minus) * (T - t))

    A = exp - 1
    B_plus = delta_plus * exp - delta_minus
    B_minus = delta_minus * exp - delta_plus

    niu = (-(epsilon - q**2) * A - c * B_plus) / (B_minus - c * A)
    return niu


def analytic_Y_as_function_of_X(X_t, t, T, module=np):
    niu = niu_function(t, T, module)
    m_t = module.mean(X_t)
    Y_t = -niu * (m_t - X_t)
    return Y_t


def analytical_X(filtration: Filtration):
    X_t = filtration.forward_process
    m_t = torch.mean(filtration.forward_process, dim=0)
    t = filtration.time_process
    T = t[:, -1].unsqueeze(-1)
    niu = niu_function(t, T, module=torch)

    theta = a + q + niu

    dummy_time = filtration.time_process[:, :-1, 0].unsqueeze(-1)
    integrand = (
        torch.exp(theta[:, :-1, :] * dummy_time) * filtration.brownian_increments
    )

    initial = torch.zeros(
        size=(filtration.number_of_paths, 1, filtration.spatial_dimensions)
    )
    integral = torch.cat([initial, torch.cumsum(integrand, dim=1)], dim=1)

    ito_integral = torch.exp(-theta * t) * integral

    value = m_t + (XI - m_t) * torch.exp(-theta * t) + SIGMA * ito_integral
    return value


def analytical_Y(filtration: Filtration):
    t = filtration.time_process
    T = t[:, -1].unsqueeze(-1)
    X_t = filtration.forward_process
    m_t = torch.mean(X_t, dim=0)
    niu = niu_function(t, T, module=torch)
    Y_t = -niu * (m_t - X_t)

    return Y_t


def analytical_Z(filtration: Filtration):
    t = filtration.time_process
    T = t[:, -1].unsqueeze(-1)
    niu = niu_function(t, T, module=torch)
    return niu


artist = FunctionApproximatorArtist(
    save_figures=True, analytical_solution=analytic_Y_as_function_of_X
)

iterations_artist = PicardIterationsArtist(
    FILTRATION,
    analytical_forward_solution=analytical_X,
    analytical_backward_solution=analytical_Y,
    analytical_backward_volatility=analytical_Z,
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

forward_backward_sde.backward_solve(
    number_of_iterations=10,
    plotter=iterations_artist,
    approximator_args=PICARD_ITERATION_ARGS,
)
