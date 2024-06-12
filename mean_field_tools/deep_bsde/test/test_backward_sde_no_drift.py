from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, BackwardSDE
from mean_field_tools.deep_bsde.utils import tensors_are_close
import torch

torch.manual_seed(0)

# BackwardSDE():

TIME_DOMAIN = torch.linspace(0, 1, 101)

FILTRATION = Filtration(
    spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=3
)

FILTRATION.generate_paths()

zero_drift_bsde = BackwardSDE(
    spatial_dimensions=1,
    time_domain=TIME_DOMAIN,
    terminal_condition_function=lambda x: x**2,
    filtration=FILTRATION,
)
_, integral = zero_drift_bsde.set_drift_path()


terminal_brownian = zero_drift_bsde.filtration.brownian_paths[:, -1, 1]
terminal_condition = zero_drift_bsde.set_terminal_condition(terminal_brownian)

optimization_target = zero_drift_bsde.set_optimization_target(
    terminal_condition=zero_drift_bsde.terminal_condition, drift_integral=zero_drift_bsde.drift_integral
)

def test_set_optimization_target_shape_zero_drift_case():
    assert optimization_target.shape == (3,101,1)

def test_set_optimization_target_consistency_along_path():
    assert tensors_are_close(
        optimization_target[:,0,0], optimization_target[:,-1,0], tolerance=1e-5
    )