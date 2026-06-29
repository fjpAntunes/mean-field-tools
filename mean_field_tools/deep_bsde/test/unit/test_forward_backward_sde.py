from mean_field_tools.deep_bsde.forward_backward_sde import (
    ForwardBackwardSDE,
    AnalyticForwardSDE,
    BackwardSDE,
)

from mean_field_tools.deep_bsde.utils import tensors_are_close, L_inf_norm
from mean_field_tools.deep_bsde.filtration import Filtration
import torch

TIME_DOMAIN = torch.linspace(0, 1, 101)

FILTRATION = Filtration(
    spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=100, seed=0
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


def BACKWARD_DRIFT(filtration: Filtration):
    X_t = filtration.forward_process

    return 2 * X_t


def TERMINAL_CONDITION(filtration: Filtration):
    X_T = filtration.forward_process[:, -1, :]

    return X_T**2


def setup():
    FILTRATION = Filtration(
        spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=100, seed=0
    )

    forward_sde = AnalyticForwardSDE(
        filtration=FILTRATION,
        functional_form=OU_FUNCTIONAL_FORM,
    )

    backward_sde = BackwardSDE(
        terminal_condition_function=TERMINAL_CONDITION,
        exogenous_process=["time_process", "forward_process"],
        filtration=FILTRATION,
        drift=BACKWARD_DRIFT,
    )
    backward_sde.initialize_approximator()

    forward_backward_sde = ForwardBackwardSDE(
        filtration=FILTRATION, forward_sde=forward_sde, backward_sde=backward_sde
    )
    return forward_backward_sde


def test_initialize():
    forward_backward_sde = setup()


def test_backward_picard_iteration_convergence():
    FILTRATION = Filtration(
        spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=100, seed=0
    )

    forward_sde = AnalyticForwardSDE(
        filtration=FILTRATION,
        functional_form=OU_FUNCTIONAL_FORM,
        volatility_functional_form=lambda f: torch.ones_like(f.time_process),
    )

    backward_sde = BackwardSDE(
        terminal_condition_function=lambda filtration: filtration.forward_process[
            :, -1, :
        ]
        * 0,
        filtration=FILTRATION,
        drift=lambda filtration: filtration.backward_process,
    )
    backward_sde.initialize_approximator()

    forward_backward_sde = ForwardBackwardSDE(
        filtration=FILTRATION, forward_sde=forward_sde, backward_sde=backward_sde
    )

    forward_backward_sde.backward_solve(number_of_iterations=3)

    output = forward_backward_sde.filtration.backward_process[0, :, 0]

    assert torch.mean(output**2) + output.var() < 5 * 1e-3


def test_per_variable_damping():
    """Test that per-variable damping applies different damping functions to different variables."""
    filtration = Filtration(
        spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=100, seed=0
    )

    forward_sde = AnalyticForwardSDE(
        filtration=filtration,
        functional_form=OU_FUNCTIONAL_FORM,
    )

    backward_sde = BackwardSDE(
        terminal_condition_function=TERMINAL_CONDITION,
        exogenous_process=["time_process", "forward_process"],
        filtration=filtration,
        drift=BACKWARD_DRIFT,
    )
    backward_sde.initialize_approximator()

    damping_config = {
        "forward_process": lambda i: 0.9,
        "backward_process": lambda i: 0.0,
    }

    fbsde = ForwardBackwardSDE(
        filtration=filtration,
        forward_sde=forward_sde,
        backward_sde=backward_sde,
        damping=damping_config,
    )
    fbsde.iteration = 1

    current = torch.ones(10)
    update = torch.zeros(10)

    # forward_process: coefficient=0.9 -> 0.9*current + 0.1*update = 0.9
    result_fp = fbsde._damping_update(current, update, variable_name="forward_process")
    assert torch.allclose(result_fp, torch.full((10,), 0.9))

    # backward_process: coefficient=0.0 -> 0.0*current + 1.0*update = 0.0
    result_bp = fbsde._damping_update(
        current, update, variable_name="backward_process"
    )
    assert torch.allclose(result_bp, torch.zeros(10))

    # forward_volatility: not specified, defaults to no damping -> pure update
    result_fv = fbsde._damping_update(
        current, update, variable_name="forward_volatility"
    )
    assert torch.allclose(result_fv, torch.zeros(10))


def test_single_callable_damping_backward_compatible():
    """Verify single callable damping still works uniformly across all variables."""
    filtration = Filtration(
        spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=100, seed=0
    )

    forward_sde = AnalyticForwardSDE(
        filtration=filtration,
        functional_form=OU_FUNCTIONAL_FORM,
    )

    backward_sde = BackwardSDE(
        terminal_condition_function=TERMINAL_CONDITION,
        exogenous_process=["time_process", "forward_process"],
        filtration=filtration,
        drift=BACKWARD_DRIFT,
    )
    backward_sde.initialize_approximator()

    fbsde = ForwardBackwardSDE(
        filtration=filtration,
        forward_sde=forward_sde,
        backward_sde=backward_sde,
        damping=lambda i: 0.5,
    )
    fbsde.iteration = 0

    current = torch.ones(10)
    update = torch.zeros(10)

    # All variables should get coefficient=0.5 -> 0.5*1 + 0.5*0 = 0.5
    for var in ("forward_process", "backward_process", "forward_volatility"):
        result = fbsde._damping_update(current, update, variable_name=var)
        assert torch.allclose(result, torch.full((10,), 0.5))
