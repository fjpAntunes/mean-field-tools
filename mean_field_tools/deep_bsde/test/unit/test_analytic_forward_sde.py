from mean_field_tools.deep_bsde.forward_backward_sde import (
    Filtration,
    AnalyticForwardSDE,
    NumericalForwardSDE,
)
from mean_field_tools.deep_bsde.utils import tensors_are_close, L_inf_norm
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


ornstein_uhlenbeck = AnalyticForwardSDE(
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
    assert tensors_are_close(empirical, analytical, tolerance=5e-1, norm=L_inf_norm)


def test_ou_path():
    filtration = Filtration(
        spatial_dimensions=1,
        time_domain=torch.linspace(0, 1, 4),
        number_of_paths=1,
        seed=0,
    )

    ornstein_uhlenbeck = AnalyticForwardSDE(
        filtration=filtration, functional_form=OU_FUNCTIONAL_FORM
    )

    ornstein_uhlenbeck.generate_paths()
    benchmark = torch.Tensor(
        [[[0.0], [0.8896945714950562], [0.46808281540870667], [-0.9225288033485413]]]
    )

    assert tensors_are_close(ornstein_uhlenbeck.paths, benchmark)


def ZERO_INITIAL_VALUE(filtration: Filtration):
    t = filtration.time_process
    zeros = torch.zero_like(t[:, 0, :])

    return zeros


def OU_DRIFT(filtration: Filtration):
    X_t = filtration.forward_process

    return -K * X_t


def OU_VOL(filtration: Filtration):
    t = filtration.time_process

    return torch.ones_like(t)
