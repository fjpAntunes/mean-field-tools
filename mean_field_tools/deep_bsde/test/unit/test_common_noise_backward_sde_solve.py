from mean_field_tools.deep_bsde.forward_backward_sde import (
    Filtration,
    CommonNoiseBackwardSDE,
)
from mean_field_tools.deep_bsde.utils import (
    IDENTITY_TERMINAL,
)
import torch


NUMBER_OF_TIMESTEPS = 101
TIME_DOMAIN = torch.linspace(0, 1, NUMBER_OF_TIMESTEPS)
NUMBER_OF_PATHS = 1000
SPATIAL_DIMENSIONS = 1


filtration = Filtration(SPATIAL_DIMENSIONS, TIME_DOMAIN, NUMBER_OF_PATHS, seed=0)

dt = TIME_DOMAIN[1] - TIME_DOMAIN[0]

bsde = CommonNoiseBackwardSDE(
    terminal_condition_function=IDENTITY_TERMINAL,
    filtration=filtration,
)

bsde.initialize_approximator()


def mock_generate_backward_process():
    return filtration.brownian_process


bsde.generate_backward_process = mock_generate_backward_process

bsde.solve_for_z(
    approximator_args={
        "training_strategy_args": {
            "batch_size": 512,
            "number_of_iterations": 100,
            "number_of_batches": 100,
            "number_of_plots": 5,
        },
    }
)


def test_solve_for_z_value():

    z = torch.ones_like(filtration.time_process)

    z_approx = bsde.z_approximator
    input = bsde.set_approximator_input()

    grad = z_approx.grad(input)[:, :, 0:1]

    grad = bsde._remove_padding(grad)

    z_hat = grad

    err = z_hat - z

    assert (err.mean() ** 2 + err.var()) ** 0.5 < 0.1
