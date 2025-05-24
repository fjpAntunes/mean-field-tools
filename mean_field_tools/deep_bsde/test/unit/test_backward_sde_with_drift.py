from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, BackwardSDE
from mean_field_tools.deep_bsde.utils import tensors_are_close, QUADRATIC_TERMINAL
import torch


TIME_DOMAIN = torch.linspace(0, 1, 101)

FILTRATION = Filtration(
    spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=3, seed=0
)


def DRIFT(filtration: Filtration):
    t = filtration.time_process
    return 2 * t  # [:, :, 0]


def setup():
    bsde = BackwardSDE(
        terminal_condition_function=QUADRATIC_TERMINAL,
        drift=DRIFT,
        filtration=FILTRATION,
    )
    _, integral = bsde.set_drift_path()

    terminal_condition = bsde.set_terminal_condition()

    optimization_target = bsde.set_optimization_target(
        terminal_condition=terminal_condition, drift_integral=bsde.drift_integral
    )

    return bsde, optimization_target


def test_setup():
    setup()


def test_drift_integral_at_0():
    bsde, _ = setup()
    assert tensors_are_close(
        bsde.drift_integral[:, 0, 0].squeeze(), torch.Tensor([1, 1, 1]), 3e-2
    )


def test_drift_integral_at_T():
    bsde, _ = setup()
    assert tensors_are_close(
        bsde.drift_integral[:, -1, 0].squeeze(), torch.Tensor([0, 0, 0]), 3e-2
    )


def test_set_terminal_condition():
    bsde, _ = setup()
    benchmark = torch.Tensor([[0.7749], [0.1563], [0.0753]])
    terminal_condition = bsde.set_terminal_condition()
    assert tensors_are_close(terminal_condition, benchmark, 1e-3)


def test_set_optimization_target_shape():
    bsde, optimization_target = setup()

    time_len = len(bsde.filtration.time_domain)
    padding = bsde.padding_size
    assert optimization_target.shape == (3, time_len + padding, 1)


def test_set_optimization_target_value_at_T():
    bsde, optimization_target = setup()
    optimization_target_at_T = bsde.terminal_condition
    assert tensors_are_close(
        optimization_target[:, -1, :], optimization_target_at_T, 1e-2
    )


def test_set_optimization_target_value_at_0():
    bsde, optimization_target = setup()
    benchmark = bsde.terminal_condition + torch.Tensor([[1], [1], [1]])
    assert tensors_are_close(optimization_target[:, 0, :], benchmark, 1e-1)
