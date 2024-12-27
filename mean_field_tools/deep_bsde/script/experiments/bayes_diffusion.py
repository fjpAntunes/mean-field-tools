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
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"

TIME_DOMAIN = torch.linspace(0, 1, 101)
NUMBER_OF_PATHS = 10_000
SPATIAL_DIMENSIONS = 2

FILTRATION = Filtration(
    spatial_dimensions=SPATIAL_DIMENSIONS,
    time_domain=TIME_DOMAIN,
    number_of_paths=NUMBER_OF_PATHS,
    seed=0,
)


loc = torch.zeros(2)
scale = torch.ones(2)
X_0 = MultivariateNormal(loc, scale_tril=torch.diag(scale)).sample(
    sample_shape=(NUMBER_OF_PATHS, 1)
)

beta_0 = 0.1
beta_1 = 0.2


def beta_function(filtration: Filtration):
    T = filtration.time_process[:, -1, :].unsqueeze(1)
    t = filtration.time_process

    value = beta_0 + beta_1 * (T - t)
    return value


mu_1 = torch.Tensor([3, 3])
mu_2 = torch.Tensor([-3, -3])

"Forward SDE definition"


def FORWARD_DRIFT(filtration: Filtration):
    X_t = filtration.forward_process
    Z_t = filtration.backward_volatility
    beta = beta_function(filtration)

    value = (beta * X_t) / 2 + (torch.sqrt(beta)) * Z_t
    return value


def FORWARD_VOLATILITY(filtration: Filtration):
    beta = beta_function(filtration)
    value = torch.sqrt(beta)

    return value


forward_sde = NumericalForwardSDE(
    filtration=FILTRATION,
    initial_value=X_0,
    drift=FORWARD_DRIFT,
    volatility=FORWARD_VOLATILITY,
)

"Backward SDE definition"


def BACKWARD_DRIFT(filtration: Filtration):
    Z_t = filtration.backward_volatility

    value = -0.5 * torch.linalg.norm(Z_t, dim=2)
    return value.unsqueeze(-1)


def TERMINAL_CONDITION(filtration: Filtration):
    X_T = filtration.forward_process[:, -1, :]

    d_1 = -(torch.linalg.norm(X_T - mu_1, dim=-1) ** 2) / 2
    # d_2 = -(torch.linalg.norm(X_T - mu_2, dim=-1) ** 2) / 2

    deltas = d_1  # torch.cat([d_1.unsqueeze(-1), d_2.unsqueeze(-1)], dim=-1)

    value = torch.logsumexp(deltas, dim=-1)
    return value.unsqueeze(-1)


backward_sde = BackwardSDE(
    terminal_condition_function=TERMINAL_CONDITION,
    filtration=FILTRATION,
    exogenous_process=["time_process", "forward_process"],
    drift=BACKWARD_DRIFT,
    number_of_dimensions=1,
)
backward_sde.initialize_approximator(
    nn_args={
        "device": device,
        "optimizer": torch.optim.Adam,
    }
)

"FBSDE definition"

forward_backward_sde = ForwardBackwardSDE(
    filtration=FILTRATION,
    forward_sde=forward_sde,
    backward_sde=backward_sde,
    # damping=lambda i: i / (1 + i),
)


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

forward_backward_sde.backward_solve(
    number_of_iterations=20,
    approximator_args=PICARD_ITERATION_ARGS,
)

import pdb

pdb.set_trace()
