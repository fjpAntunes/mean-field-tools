from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, BackwardSDE
from mean_field_tools.deep_bsde.utils import tensors_are_close, QUADRATIC_TERMINAL
import torch

torch.manual_seed(0)

# BackwardSDE():

TIME_DOMAIN = torch.linspace(0, 1, 101)

FILTRATION = Filtration(
    spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=3, seed=0
)


zero_drift_bsde = BackwardSDE(
    terminal_condition_function=QUADRATIC_TERMINAL,
    filtration=FILTRATION,
)
_, integral = zero_drift_bsde.set_drift_path()


terminal_condition = zero_drift_bsde.set_terminal_condition()

optimization_target = zero_drift_bsde.set_optimization_target(
    terminal_condition=zero_drift_bsde.terminal_condition,
    drift_integral=zero_drift_bsde.drift_integral,
)

time_length = len(TIME_DOMAIN)
padding = zero_drift_bsde.padding_size


def test_set_optimization_target_shape_zero_drift_case():
    assert optimization_target.shape == (3, time_length + padding, 1)


def test_set_optimization_target_consistency_along_path():
    assert tensors_are_close(
        optimization_target[:, 0, 0], optimization_target[:, -1, 0], tolerance=1e-5
    )


FILTRATION_2D = Filtration(
    spatial_dimensions=2, time_domain=TIME_DOMAIN, number_of_paths=3, seed=0
)


zero_drift_bsde_2d = BackwardSDE(
    terminal_condition_function=QUADRATIC_TERMINAL,
    filtration=FILTRATION_2D,
)
_, integral = zero_drift_bsde_2d.set_drift_path()


terminal_condition_2d = zero_drift_bsde_2d.set_terminal_condition()

optimization_target_2d = zero_drift_bsde_2d.set_optimization_target(
    terminal_condition=zero_drift_bsde_2d.terminal_condition,
    drift_integral=zero_drift_bsde_2d.drift_integral,
)


def test_set_optimization_target_shape_zero_drift_2d_case():
    assert optimization_target_2d.shape == (3, time_length + padding, 2)


def test_set_optimization_target_consistency_along_path_2d():
    assert tensors_are_close(
        optimization_target_2d[:, 0, :],
        optimization_target_2d[:, -1, :],
        tolerance=1e-5,
    )
