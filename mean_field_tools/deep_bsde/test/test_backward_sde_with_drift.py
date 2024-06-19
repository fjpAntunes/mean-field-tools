from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, BackwardSDE
from mean_field_tools.deep_bsde.utils import tensors_are_close
import torch

torch.manual_seed(0)

# BackwardSDE():

TIME_DOMAIN = torch.linspace(0, 1, 101)

FILTRATION = Filtration(
    spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=3
)


def QUADRATIC(x):
    return x**2


bsde = BackwardSDE(
    terminal_condition_function=QUADRATIC,
    drift=lambda t: 2 * t[:, :, 0],
    filtration=FILTRATION,
)
_, integral = bsde.set_drift_path()


terminal_brownian = bsde.filtration.brownian_process[:, -1, 0]
terminal_condition = bsde.set_terminal_condition(terminal_brownian)

optimization_target = bsde.set_optimization_target(
    terminal_condition=terminal_condition, drift_integral=bsde.drift_integral
)


def test_drift_integral_at_0():
    assert tensors_are_close(integral[:, 0, 0].squeeze(), torch.Tensor([1, 1, 1]), 3e-2)


def test_drift_integral_at_T():
    assert tensors_are_close(
        integral[:, -1, 0].squeeze(), torch.Tensor([0, 0, 0]), 3e-2
    )


def test_set_terminal_condition():
    benchmark = torch.Tensor([0.7749, 0.1563, 0.0753])
    assert tensors_are_close(
        bsde.set_terminal_condition(terminal_brownian), benchmark, 1e-3
    )


def test_set_optimization_target_shape():
    assert optimization_target.shape == (3, 101, 1)


def test_set_optimization_target_value_at_T():
    assert tensors_are_close(optimization_target[:, -1, 0], terminal_condition, 1e-2)


def test_set_optimization_target_value_at_0():

    benchmark = terminal_condition + torch.Tensor([1, 1, 1])
    assert tensors_are_close(optimization_target[:, 0, 0], benchmark, 1e-1)
