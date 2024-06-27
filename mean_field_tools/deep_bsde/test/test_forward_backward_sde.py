from mean_field_tools.deep_bsde.forward_backward_sde import (
    ForwardBackwardSDE,
    ForwardSDE,
    BackwardSDE,
)
from mean_field_tools.deep_bsde.filtration import Filtration
import torch

TIME_DOMAIN = torch.linspace(0, 1, 101)

FILTRATION = Filtration(
    spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=3000, seed=0
)


K = 1


def OU_FUNCTIONAL_FORM(filtration):
    dummy_time = filtration.time_process[:, 1:, 0].unsqueeze(-1)
    integrand = torch.exp(K * dummy_time) * filtration.brownian_increments

    initial = torch.zeros(
        size=(filtration.number_of_paths, 1, filtration.spatial_dimensions)
    )
    integral = torch.cat([initial, torch.cumsum(integrand, dim=1)], dim=1)

    time = filtration.time_process[:, :, 0].unsqueeze(-1)
    path = torch.exp(-K * time) * integral
    return path


def BACKWARD_DRIFT(forward_backward_sde: ForwardBackwardSDE, filtration: Filtration):
    X_t = forward_backward_sde.forward_sde.generate_paths(filtration)

    return 2 * X_t


def TERMINAL_CONDITION(filtration: Filtration):
    X_T = filtration.path.forward[:, -1, :]

    return X_T**2


def setup():
    forward_backward_sde = ForwardBackwardSDE(
        filtration=FILTRATION,
        forward_functional_form=OU_FUNCTIONAL_FORM,
        backward_drift=BACKWARD_DRIFT,
        terminal_condition_function=TERMINAL_CONDITION,
    )
    return forward_backward_sde


def test_initialize():
    forward_backward_sde = setup()


def test_backward_solve():
    forward_backward_sde = setup()
    forward_backward_sde.backward_solve()
