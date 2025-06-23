from mean_field_tools.deep_bsde.forward_backward_sde import (
    CommonNoiseBackwardSDE,
    NumericalForwardSDE,
    ForwardBackwardSDE,
)
from mean_field_tools.deep_bsde.filtration import CommonNoiseFiltration, Filtration
from mean_field_tools.deep_bsde.script.experiments.systemic_risk_common_noise.artist import (
    FunctionApproximatorArtist,
    PicardIterationsArtist,
    cast_to_np,
)

from mean_field_tools.deep_bsde.measure_flow import CommonNoiseMeasureFlow
import torch
import numpy as np
import matplotlib.pyplot as plt


torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"

COMMON_NOISE_COEFFICIENT = 0.3
NUMBER_OF_TIMESTEPS = 101
NUMBER_OF_PATHS = 10_000
SPATIAL_DIMENSIONS = 1

TIME_DOMAIN = torch.linspace(0, 1, NUMBER_OF_TIMESTEPS)

FILTRATION = CommonNoiseFiltration(
    spatial_dimensions=SPATIAL_DIMENSIONS,
    time_domain=TIME_DOMAIN,
    number_of_paths=NUMBER_OF_PATHS,
    common_noise_coefficient=COMMON_NOISE_COEFFICIENT,
    seed=0,
)


XI = torch.distributions.normal.Normal(loc=0, scale=2).sample(
    sample_shape=(NUMBER_OF_PATHS, 1, SPATIAL_DIMENSIONS)
)

"Parameters"

a = 2
q = 1
SIGMA = 1
epsilon = 10
c = 0.1

"Forward SDE definition"


def ZERO_FUNCTION(filtration: Filtration):
    zero = torch.zeros_like(filtration.time_process)

    return zero


def FORWARD_DRIFT(filtration: Filtration):
    X_t = filtration.forward_process
    m_t = filtration.forward_mean_field
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
    m_t = filtration.forward_mean_field
    Y_t = filtration.backward_process
    value = (a + q) * Y_t + (epsilon - q**2) * (m_t - X_t)
    return -value


def TERMINAL_CONDITION(filtration: Filtration):
    X_T = filtration.forward_process[:, -1, :]
    mu_T = filtration.forward_mean_field[:, -1, :]
    value = c * (X_T - mu_T)

    return value


backward_sde = CommonNoiseBackwardSDE(
    drift=BACKWARD_DRIFT,
    terminal_condition_function=TERMINAL_CONDITION,
    filtration=FILTRATION,
    exogenous_process=["time_process", "forward_process", "forward_mean_field"],
)
backward_sde.initialize_approximator(
    nn_args={
        "number_of_layers": 2,
        "number_of_nodes": 18,
        "device": device,
        "optimizer": torch.optim.Adam,
    }
)

backward_sde.initialize_z_approximator(
    nn_args={
        "number_of_layers": 1,
        "number_of_nodes": 2,
        "device": device,
        "optimizer": torch.optim.Adam,
    }
)


"measure flow definition"

measure_flow = CommonNoiseMeasureFlow(filtration=FILTRATION)
measure_flow.initialize_approximator(
    nn_args={
        "number_of_layers": 2,
        "number_of_nodes": 18,
        "device": device,
        "optimizer": torch.optim.Adam,
    },
    training_args={
        "training_strategy_args": {
            "batch_size": 2048,
            "number_of_iterations": 1000,
            "number_of_batches": 1000,
        }
    },
)

"FBSDE definition"

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


def analytical_mean_X(filtration: CommonNoiseFiltration):
    W0_t = filtration.common_noise
    rho = filtration.common_noise_coefficient
    mean_xi = torch.mean(XI)
    mean_X = mean_xi + rho * SIGMA * W0_t

    return mean_X


def riemman_integral(process, filtration: Filtration):
    dt = filtration.dt

    initial = torch.zeros(
        size=(filtration.number_of_paths, 1, filtration.spatial_dimensions)
    )
    integral = torch.cat([initial, torch.cumsum(process[:, :-1, :] * dt, dim=1)], dim=1)

    return integral


def ito_integral(process, filtration: Filtration):
    initial = torch.zeros(
        size=(filtration.number_of_paths, 1, filtration.spatial_dimensions)
    )
    dBt = filtration.brownian_increments

    ito_integral = torch.cat(
        [initial, torch.cumsum(process[:, :-1, :] * dBt, axis=1)], dim=1
    )

    return ito_integral


def analytical_X(filtration: Filtration):
    m_t = analytical_mean_X(filtration)
    t = filtration.time_process
    T = t[:, -1].unsqueeze(-1)
    niu = niu_function(t, T, module=torch)

    theta = a + q + niu

    exp_int_theta = torch.exp(riemman_integral(theta, filtration))

    drift_term = riemman_integral(theta * exp_int_theta * m_t, filtration)

    vol_term = ito_integral(SIGMA * exp_int_theta, filtration)

    value = (XI + drift_term + vol_term) / exp_int_theta

    return value


def analytical_Y(filtration: CommonNoiseFiltration):
    t = filtration.time_process
    T = t[:, -1].unsqueeze(-1)
    X_t = filtration.forward_process
    m_t = analytical_mean_X(filtration)

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


class SystemicRiskCommonNoiseArtist(PicardIterationsArtist):
    def plot_single_path(self: PicardIterationsArtist):
        _, axs = plt.subplots(3, 1, layout="constrained")
        t = cast_to_np(self.filtration.time_process)[0, :, :]

        x = cast_to_np(self.analytical_forward_solution(self.filtration))[0, :, :]
        axs[0].plot(t, x, color="r", label="Forward Process - Analytical")

        mean_x = cast_to_np(analytical_mean_X(self.filtration))[0, :, :]
        axs[1].plot(t, mean_x, color="r", label="Forward Process Mean - Analytical")

        y = cast_to_np(self.analytical_backward_solution(self.filtration))[0, :, :]
        axs[2].plot(t, y, color="r", label="Backward Process - Analytical")

        y_hat = cast_to_np(self.filtration.backward_process)[0, :, :]
        x_hat = cast_to_np(self.filtration.forward_process)[0, :, :]
        mean_x_hat = cast_to_np(self.filtration.forward_mean_field)[0, :, :]
        axs[0].plot(t, x_hat, "b--", label="Forward Process - Approximation")
        axs[1].plot(
            t,
            mean_x_hat,
            "b--",
            label="Forward Process Mean - Approximation",
        )
        axs[2].plot(t, y_hat, "b--", label="Backward Process - Approximation")

        for i in [0, 1, 2]:
            axs[i].legend()
        path = f"./.figures/single_path_{self.iteration}.png"

        plt.savefig(path)

        plt.close()


iterations_artist = SystemicRiskCommonNoiseArtist(
    FILTRATION,
    analytical_forward_solution=analytical_X,
    analytical_backward_solution=analytical_Y,
    analytical_backward_volatility=analytical_Z,
    analytical_forward_mean=analytical_mean_X,
)


"Solving"

PICARD_ITERATION_ARGS = {
    "training_strategy_args": {
        "batch_size": 2048,
        "number_of_iterations": 1000,
        "number_of_batches": 1000,
        "plotter": artist,
        "number_of_plots": 1,
    },
}


forward_backward_sde.backward_solve(
    number_of_iterations=100,
    plotter=iterations_artist,
    approximator_args=PICARD_ITERATION_ARGS,
)

import pdb

pdb.set_trace()

PLOTTING_FILTRATION = CommonNoiseFiltration(
    spatial_dimensions=SPATIAL_DIMENSIONS,
    time_domain=TIME_DOMAIN,
    number_of_paths=NUMBER_OF_PATHS,
    common_noise_coefficient=COMMON_NOISE_COEFFICIENT,
    seed=0,
)


PLOTTING_FILTRATION.brownian_increments = (
    PLOTTING_FILTRATION.common_noise_coefficient
    * PLOTTING_FILTRATION.common_noise_increments[0, :, :]
    + ((1 - PLOTTING_FILTRATION.common_noise_coefficient**2) ** 0.5)
    * PLOTTING_FILTRATION.idiosyncratic_noise_increments
)
PLOTTING_FILTRATION.brownian_process = PLOTTING_FILTRATION._generate_brownian_process(
    PLOTTING_FILTRATION.brownian_increments
)


PLOTTING_FILTRATION.common_noise = PLOTTING_FILTRATION.common_noise[4, :, 0].repeat(
    repeats=(NUMBER_OF_PATHS, 1)
)
PLOTTING_FILTRATION.common_noise = torch.unsqueeze(
    PLOTTING_FILTRATION.common_noise, dim=-1
)

measure_flow.filtration = PLOTTING_FILTRATION
input = measure_flow._set_elicitability_input()
PLOTTING_FILTRATION.forward_mean_field = measure_flow.mean_approximator.detached_call(
    input
)

import pdb

pdb.set_trace()

print("Solving again for plotting")

forward_sde.filtration = PLOTTING_FILTRATION
backward_sde.filtration = PLOTTING_FILTRATION
forward_backward_sde.filtration = PLOTTING_FILTRATION
forward_backward_sde.measure_flow = None

poster_artist = SystemicRiskCommonNoiseArtist(
    PLOTTING_FILTRATION,
    analytical_forward_solution=analytical_X,
    analytical_backward_solution=analytical_Y,
    analytical_backward_volatility=analytical_Z,
    output_folder="./poster_plots",
)
poster_artist.plot_error_hist_for_iterations = lambda: None

forward_backward_sde.backward_solve(
    number_of_iterations=10,
    plotter=poster_artist,
    approximator_args=PICARD_ITERATION_ARGS,
)

# Questions
# - Why the decoupling field plot is scattered?
# - Why Z is not approximating? Or is it approximating too slowly?
