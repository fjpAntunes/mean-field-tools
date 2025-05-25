from mean_field_tools.deep_bsde.filtration import CommonNoiseFiltration
from mean_field_tools.deep_bsde.forward_backward_sde import CommonNoiseBackwardSDE
from mean_field_tools.deep_bsde.utils import IDENTITY_TERMINAL, L_2_norm
import torch

torch.manual_seed(0)

# Filtration

TIME_DOMAIN = torch.linspace(0, 1, 101)
RHO = 0.3

FILTRATION = CommonNoiseFiltration(
    spatial_dimensions=1,
    time_domain=TIME_DOMAIN,
    number_of_paths=1000,
    common_noise_coefficient=RHO,
    seed=0,
)

"""
Y_t = \sqrt( 1 - \rho^2) W_t + \rho W_t^0
"""
bsde = CommonNoiseBackwardSDE(
    terminal_condition_function=IDENTITY_TERMINAL,
    filtration=FILTRATION,
)


def test_solve_for_idiosyncratic():
    # Z should be sqrt(1 - \rho^2)
    z_hat = bsde.solve_for_idiosyncratic_volatility()

    z = (1 - RHO**2) ** 0.5 * torch.ones_like(FILTRATION.brownian_process)

    err = z_hat - z

    assert L_2_norm(err) < 0.1


def test_solve_for_common():
    # Z_0 should be \rho
    z_0_hat = bsde.solve_for_common_volatility()

    z_0 = RHO * torch.ones_like(FILTRATION.brownian_process)

    err = z_0_hat - z_0

    assert L_2_norm(err) < 0.1
