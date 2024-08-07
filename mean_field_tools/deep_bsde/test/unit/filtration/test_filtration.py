from mean_field_tools.deep_bsde.filtration import Filtration, BrownianIncrementGenerator
from mean_field_tools.deep_bsde.utils import tensors_are_close, L_inf_norm
import torch

torch.manual_seed(0)

# Filtration

TIME_DOMAIN = torch.linspace(0, 1, 101)

FILTRATION = Filtration(
    spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=1000
)


def test_path_shape():
    assert FILTRATION.brownian_process.shape == (1000, 101, 1)


def test_inital_value_equal_zero():
    benchmark = torch.zeros(size=(1000, 1))
    assert tensors_are_close(
        FILTRATION.brownian_process[:, 0, 0], benchmark, tolerance=1e-3
    )


def test_brownian_process_mean():
    mean = torch.mean(FILTRATION.brownian_process, dim=0)

    assert tensors_are_close(
        mean,
        torch.zeros_like(mean),
        tolerance=1e-1,
        norm=L_inf_norm,
    )


def test_brownian_process_variance():
    empirical = torch.var(FILTRATION.brownian_process, dim=0).squeeze()
    analytical = TIME_DOMAIN
    assert tensors_are_close(empirical, analytical, tolerance=3e-1, norm=L_inf_norm)


def test_brownian_process_increments():
    filtration = Filtration(
        spatial_dimensions=1,
        time_domain=torch.linspace(0, 1, 4),
        number_of_paths=1,
        seed=0,
    )

    benchmark = torch.Tensor(
        [[[0.8896945118904114], [-0.16941122710704803], [-1.2579247951507568]]]
    )

    assert tensors_are_close(filtration.brownian_increments, benchmark)


def test_brownian_process_path():
    filtration = Filtration(
        spatial_dimensions=1,
        time_domain=torch.linspace(0, 1, 4),
        number_of_paths=1,
        seed=0,
    )
    benchmark = torch.Tensor(
        [[[0.0], [0.8896945118904114], [0.7202832698822021], [-0.5376415252685547]]]
    )

    assert tensors_are_close(filtration.brownian_process, benchmark)
