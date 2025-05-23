from mean_field_tools.deep_bsde.forward_backward_sde import (
    BackwardSDE,
    NumericalForwardSDE,
    ForwardBackwardSDE,
)
from mean_field_tools.deep_bsde.filtration import CommonNoiseFiltration, Filtration
from mean_field_tools.deep_bsde.artist import (
    FunctionApproximatorArtist,
    PicardIterationsArtist,
    cast_to_np,
)

from mean_field_tools.deep_bsde.measure_flow import MeasureFlow
import torch
import numpy as np
import matplotlib.pyplot as plt


torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"

NUMBER_OF_TIMESTEPS = 101
NUMBER_OF_PATHS = 10_000
SPATIAL_DIMENSIONS = 2

TIME_DOMAIN = torch.linspace(0, 1, NUMBER_OF_TIMESTEPS)

FILTRATION = Filtration(
    spatial_dimensions=SPATIAL_DIMENSIONS,
    time_domain=TIME_DOMAIN,
    number_of_paths=NUMBER_OF_PATHS,
    seed=0,
)


XI = torch.distributions.normal.Normal(loc=0.5, scale=0.5).sample(
    sample_shape=(NUMBER_OF_PATHS, 1, SPATIAL_DIMENSIONS)
)

"Parameters"


DELTA = 0.05
XI = 0

alpha = 0.5
beta = 1 - alpha
C = 0.5
tolerance = 1e-3

SIGMA_A = 1
SIGMA_H = 1


def feedback_consumption(p):
    c0 = 0
    return np.power(p, -1)  # c0 - p


def feedback_work_effort(h, p, q, bar_w):
    value = bar_w * h * p - np.power(h, XI) * q
    return np.maximum(np.minimum(value, 1), 0)


def mean_field_interest_rate(k, h):
    safe_k = np.maximum(k, tolerance)
    safe_h = np.maximum(h, tolerance)
    return C * alpha * np.power(safe_k, alpha - 1) * np.power(safe_h, beta)


def mean_field_wage_rate(k, h):
    safe_k = np.maximum(k, tolerance)
    safe_h = np.maximum(h, tolerance)
    return C * beta * np.power(safe_k, alpha) * np.power(safe_h, beta - 1)


def education_efficiency(effort):
    return effort


def capital_costate_terminal_condition(k):
    return 1


"Forward SDE definition"


def ZERO_FUNCTION(filtration: Filtration):
    zero = torch.zeros_like(filtration.time_process)

    return zero


def FORWARD_DRIFT(filtration: Filtration):
    A_t = filtration.forward_process[:, :, 0]
    H_t = filtration.forward_process[:, :, 1]
    P_t = filtration.backward_process[:, :, 0]
    Q_t = filtration.backward_process[:, :, 1]
    hatA_t = filtration.forward_mean_field[:, :, 0]
    hat_H_eff_t = filtration.forward_mean_field[:, :, 1]

    r_t = mean_field_interest_rate(k=hatA_t, h=hat_H_eff_t)
    w_t = mean_field_wage_rate(k=hatA_t, h=hat_H_eff_t)

    u_t = feedback_work_effort(
        h=H_t,
        p=P_t,
        q=Q_t,
        bar_w=w_t,
    )
    c_t = feedback_consumption(p=P_t)

    dA_t = (r_t - DELTA) * A_t + w_t * H_t * u_t - c_t
    dH_t = H_t * education_efficiency(1 - u_t)

    value = torch.cat([dA_t, dH_t])
    return value


def VOLATILITY(filtration: Filtration):
    A_t = filtration.forward_process[:, :, 0]
    H_t = filtration.forward_process[:, :, 1]
    vol_A = SIGMA_A * A_t
    vol_H = SIGMA_H * H_t

    value = torch.cat([vol_A, vol_H])
    return value


forward_sde = NumericalForwardSDE(
    filtration=FILTRATION,
    initial_value=XI,
    drift=FORWARD_DRIFT,
    volatility=VOLATILITY,
    tolerance=1e-6,
)

"Backward SDE definition"


def BACKWARD_DRIFT(filtration: Filtration):
    A_t = filtration.forward_process[:, :, 0]
    H_t = filtration.forward_process[:, :, 1]
    P_t = filtration.backward_process[:, :, 0]
    Q_t = filtration.backward_process[:, :, 1]
    Zp_t = filtration.backward_volatility[:, :, 0]
    Zq_t = filtration.backward_volatility[:, :, 1]

    hatA_t = filtration.forward_mean_field[:, :, 0]
    hat_H_eff_t = filtration.forward_mean_field[:, :, 1]

    r_t = mean_field_interest_rate(k=hatA_t, h=hat_H_eff_t)
    w_t = mean_field_wage_rate(k=hatA_t, h=hat_H_eff_t)

    u_t = feedback_work_effort(
        h=H_t,
        p=P_t,
        q=Q_t,
        bar_w=w_t,
    )
    c_t = feedback_consumption(p=P_t)

    dP_t = -((r_t - DELTA) * P_t + SIGMA_A * Zp_t)
    dQ_t = -(
        XI * torch.power(H_t, XI - 1) * education_efficiency(1 - u_t) * Q_t
        + w_t * u_t * P_t
        + SIGMA_H * Zq_t
    )
    value = torch.cat([dP_t, dQ_t])
    return value


### Continue from here


def TERMINAL_CONDITION(filtration: Filtration):
    X_T = filtration.forward_process[:, -1, :]
    # mu_T = filtration.forward_mean_field[:, -1, :]
    value = -(X_T - 5)

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

"measure flow definition"

measure_flow = MeasureFlow(filtration=FILTRATION)
# measure_flow.initialize_approximator(
#    training_args={
#        "training_strategy_args": {
#            "batch_size": 512,
#            "number_of_iterations": 100,
#            "number_of_batches": 100,
#        }
#    },
# )

"FBSDE definition"

forward_backward_sde = ForwardBackwardSDE(
    filtration=FILTRATION,
    forward_sde=forward_sde,
    backward_sde=backward_sde,
    damping=lambda i: 0.5,
    measure_flow=measure_flow,
)


"Plotting Parameters"


"Solving"

PICARD_ITERATION_ARGS = {
    "training_strategy_args": {
        "batch_size": 512,
        "number_of_iterations": 100,
        "number_of_batches": 100,
        "number_of_plots": 1,
    },
}

iterations_artist = PicardIterationsArtist(
    FILTRATION,
)

forward_backward_sde.backward_solve(
    number_of_iterations=20,
    approximator_args=PICARD_ITERATION_ARGS,
    initial_backward_process=torch.ones_like(FILTRATION.brownian_process),
    plotter=iterations_artist,
)


X_t = FILTRATION.forward_process.detach().cpu().numpy()
Y_t = FILTRATION.backward_process.detach().cpu().numpy()
m_t = FILTRATION.forward_mean_field.detach().cpu().numpy()
t = FILTRATION.time_process.detach().cpu().numpy()[0, :, 0]

plot_number_of_trajectories = 10
fig, axs = plt.subplots(3, 1, layout="constrained")

axs[0].plot(t, m_t[0, :, 0])

for i in range(plot_number_of_trajectories):
    x = X_t[i, :, 0]
    y = Y_t[i, :, 0]
    axs[1].plot(t, x)
    axs[2].plot(t, 1 / y)


axs[0].set_title("Forward process sample paths")
axs[1].set_title("Backward process sample paths")

path = f"./.figures/sample_paths.png"
plt.savefig(path)

plt.close()

n_bis = 50

fig, axs = plt.subplots(3, 1, figsize=(4, 8), layout="constrained")

n_bins = 50
axs[0].hist(X_t[:, 0, 0], bins=n_bins, density=True)
axs[1].hist(X_t[:, 50, 0], bins=n_bins, density=True)
axs[2].hist(X_t[:, -1, 0], bins=n_bins, density=True)

for i in range(2):
    axs[i].grid(True)
    # axs[i].set_xlim(-1, 1)

plt.savefig(f"./.figures/error_histogram_iteration.png")
plt.close()
