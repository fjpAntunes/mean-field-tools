from mean_field_tools.deep_bsde.filtration import CommonNoiseFiltration
from mean_field_tools.deep_bsde.utils import tensors_are_close, L_inf_norm
import torch

torch.manual_seed(0)

# Filtration

TIME_DOMAIN = torch.linspace(0, 1, 101)

FILTRATION = CommonNoiseFiltration(
    spatial_dimensions=1,
    time_domain=TIME_DOMAIN,
    number_of_paths=1000,
    common_noise_coefficient=0.3,
    seed=0,
)


def test_path_shape():
    assert FILTRATION.brownian_process.shape == (1000, 101, 1)


def test_brownian_inital_value_equal_zero():
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


def test_idiosyncratic_noise_shape():
    assert FILTRATION.idiosyncratic_noise.shape == (1000, 101, 1)


def test_idiosyncratic_inital_value_equal_zero():
    benchmark = torch.zeros(size=(1000, 1))
    assert tensors_are_close(
        FILTRATION.idiosyncratic_noise[:, 0, 0], benchmark, tolerance=1e-3
    )


def test_idiosyncratic_process_mean():
    mean = torch.mean(FILTRATION.idiosyncratic_noise, dim=0)

    assert tensors_are_close(
        mean,
        torch.zeros_like(mean),
        tolerance=1e-1,
        norm=L_inf_norm,
    )


def test_idiosyncratic_process_variance():
    empirical = torch.var(FILTRATION.idiosyncratic_noise, dim=0).squeeze()
    analytical = TIME_DOMAIN
    assert tensors_are_close(empirical, analytical, tolerance=3e-1, norm=L_inf_norm)


def test_common_noise_shape():
    assert FILTRATION.common_noise.shape == (1000, 101, 1)


def test_common_noise_inital_value_equal_zero():
    benchmark = torch.zeros(size=(1000, 1))
    assert tensors_are_close(
        FILTRATION.common_noise[:, 0, 0], benchmark, tolerance=1e-3
    )


def test_common_noise_process_mean():
    mean = torch.mean(FILTRATION.common_noise, dim=0)

    assert tensors_are_close(
        mean,
        torch.zeros_like(mean),
        tolerance=1e-1,
        norm=L_inf_norm,
    )


def test_common_noise_process_variance():
    empirical = torch.var(FILTRATION.common_noise, dim=0).squeeze()
    analytical = TIME_DOMAIN
    assert tensors_are_close(empirical, analytical, tolerance=3e-1, norm=L_inf_norm)


def process_covariance(X, Y):
    mean_of_products = torch.mean(X * Y, dim=0)
    product_of_means = torch.mean(X, dim=0) * torch.mean(Y, dim=0)

    covariance = mean_of_products - product_of_means
    return covariance


def test_brownian_and_common_noise_covariance():
    "should evaluate to rho * t"
    brownian_process = FILTRATION.brownian_process
    common_noise = FILTRATION.common_noise
    rho = FILTRATION.common_noise_coefficient
    t = TIME_DOMAIN

    covariance = process_covariance(brownian_process, common_noise).squeeze(-1)

    analytical = rho * t

    assert tensors_are_close(covariance, analytical, tolerance=3e-1, norm=L_inf_norm)


def test_common_noise_and_idiosyncratic_covariance():
    "should evaluate to zero"
    idiosyncratic = FILTRATION.idiosyncratic_noise
    common_noise = FILTRATION.common_noise
    t = TIME_DOMAIN

    covariance = process_covariance(idiosyncratic, common_noise).squeeze(-1)

    assert tensors_are_close(
        covariance, torch.zeros_like(t), tolerance=3e-1, norm=L_inf_norm
    )


def test_independence_of_increments():
    idiosyncratic = FILTRATION.idiosyncratic_noise_increments
    common = FILTRATION.common_noise_increments
    dt = FILTRATION.dt

    assert torch.mean((common * idiosyncratic) / dt) < 5e-3
