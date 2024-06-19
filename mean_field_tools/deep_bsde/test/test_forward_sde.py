from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, ForwardSDE
from mean_field_tools.deep_bsde.utils import tensors_are_close, L_inf_norm
import torch

torch.manual_seed(0)

# BackwardSDE():

TIME_DOMAIN = torch.linspace(0, 1, 101)

FILTRATION = Filtration(
    spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=3000
)


K = 1


def OU_FUNCTIONAL_FORM(filtration):
    dummy_time = filtration.time_process[:, :, 0]

    integrand = torch.exp(K * dummy_time) * filtration.brownian_increments.squeeze()

    integral = torch.cumsum(integrand, dim=1)

    time = filtration.time_process[:, :, 0]
    return torch.exp(-K * time) * integral


ornstein_uhlenbeck = ForwardSDE(
    filtration=FILTRATION, functional_form=OU_FUNCTIONAL_FORM
)

ornstein_uhlenbeck.generate_paths()


def test_path_mean():
    mean = torch.mean(ornstein_uhlenbeck.paths, dim=0)

    assert tensors_are_close(
        mean,
        torch.zeros_like(mean),
        tolerance=1e-1,
        norm=L_inf_norm,
    )


def test_path_variance():
    empirical = torch.var(ornstein_uhlenbeck.paths, dim=0)
    analytical = (1 - torch.exp(-2 * TIME_DOMAIN)) / 2
    assert tensors_are_close(empirical, analytical, tolerance=1e-1, norm=L_inf_norm)
