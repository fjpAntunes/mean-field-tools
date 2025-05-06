from mean_field_tools.deep_bsde.forward_backward_sde import (
    ForwardBackwardSDE,
    AnalyticForwardSDE,
    BackwardSDE,
)
from mean_field_tools.deep_bsde.artist import PicardIterationsArtist
from mean_field_tools.deep_bsde.utils import tensors_are_close, L_inf_norm
from mean_field_tools.deep_bsde.filtration import Filtration
import torch

TIME_DOMAIN = torch.linspace(0, 1, 101)

FILTRATION = Filtration(
    spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=10_000, seed=0
)

MU = 1
SIGMA = 2
r = 1
S_0 = 1


def FORWARD_DRIFT(filtration: Filtration):
    S_t = filtration.forward_process
    value = MU * S_t

    return value


def ANALYTICAL_GBM(filtration: Filtration):
    t = filtration.time_process
    W_t = filtration.brownian_process

    exponent = SIGMA * W_t + (MU - 0.5 * SIGMA**2) * t
    value = S_0 * torch.exp(exponent)

    return value


def FORWARD_VOL(filtration: Filtration):
    S_t = ANALYTICAL_GBM(filtration)
    value = SIGMA * S_t

    return value


def BACKWARD_DRIFT(filtration: Filtration):
    Y_t = filtration.backward_process
    Z_t = filtration.backward_volatility
    S_t = filtration.forward_process

    h_t = Z_t / (S_t * SIGMA)

    print(f"Y error: mean {(Y_t - S_t).mean()} , var {(Y_t - S_t).var()}")
    print(f"Z error: mean {(Z_t - SIGMA*S_t).mean()} , var {(Z_t - SIGMA*S_t).var()}")

    value = r * (Y_t - (Z_t / SIGMA)) + MU * (Z_t / SIGMA)
    return -value


def TERMINAL_CONDITION(filtration: Filtration):
    S_T = filtration.forward_process[:, -1, :]

    return S_T


forward_sde = AnalyticForwardSDE(
    filtration=FILTRATION,
    functional_form=ANALYTICAL_GBM,
    volatility_functional_form=FORWARD_VOL,
)

backward_sde = BackwardSDE(
    terminal_condition_function=TERMINAL_CONDITION,
    exogenous_process=["time_process", "forward_process"],
    filtration=FILTRATION,
    drift=BACKWARD_DRIFT,
)
backward_sde.initialize_approximator(
    nn_args={"number_of_layers": 1, "number_of_nodes": 18}
)

forward_backward_sde = ForwardBackwardSDE(
    filtration=FILTRATION,
    forward_sde=forward_sde,
    backward_sde=backward_sde,
    damping=lambda i: (1 - 0.5 / (i + 1) ** 0.5),
)

iterations_artist = PicardIterationsArtist(
    FILTRATION,
)


PICARD_ITERATION_ARGS = {
    "training_strategy_args": {
        "batch_size": 1024,
        "number_of_iterations": 1000,
        "number_of_batches": 1000,
        "number_of_plots": 1,
    },
}


forward_backward_sde.backward_solve(
    number_of_iterations=10,
    approximator_args=PICARD_ITERATION_ARGS,
    # plotter=iterations_artist,
    initial_forward_process=forward_sde.generate_paths(),
)

nn = backward_sde.y_approximator

import matplotlib.pyplot as plt
from mean_field_tools.deep_bsde.artist import cast_to_np


fig, axs = plt.subplots(2, 1)

x_hat = cast_to_np(FILTRATION.forward_process)[0, :, :]
y_hat = cast_to_np(FILTRATION.backward_process)[0, :, :]
z_hat = cast_to_np(FILTRATION.backward_volatility)[0, :, :]

axs[0].scatter(x_hat, y_hat)
axs[1].scatter(x_hat, z_hat)

fig.show()


import pdb

pdb.set_trace()
### Derivada não está funcionando corretamente - pq?
### Incluir regularização no Z_t
### Testar função gradiente
