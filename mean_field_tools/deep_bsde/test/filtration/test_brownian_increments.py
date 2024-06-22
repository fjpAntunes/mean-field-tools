from mean_field_tools.deep_bsde.filtration import Filtration, BrownianIncrementGenerator
from mean_field_tools.deep_bsde.utils import tensors_are_close, L_inf_norm
import torch

torch.manual_seed(0)

# Filtration

TIME_DOMAIN = torch.linspace(0, 1, 101)

FILTRATION = Filtration(
    spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=1000
)

covariance = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
brownian_increment_generator = BrownianIncrementGenerator(
    number_of_paths=10,
    spatial_dimensions=3,
    sampling_times=TIME_DOMAIN,
    spatial_covariance=covariance,
    seed=0,
)


def test_set_spatial_covariance():
    covariance = brownian_increment_generator._set_spatial_covariance(2, None)

    assert tensors_are_close(covariance, torch.Tensor([[1, 0], [0, 1]]))


def test_set_standard_deviation():
    dt = torch.Tensor([1, 4, 9])
    cov = torch.Tensor([[1, 4], [4, 1]])
    standard_deviation = brownian_increment_generator._set_standard_deviation(
        dt=dt, covariance=cov
    )
    benchmark = torch.Tensor(
        [[[1.0, 2.0], [2.0, 1.0]], [[2.0, 4.0], [4.0, 2.0]], [[3.0, 6.0], [6.0, 3.0]]]
    )
    assert tensors_are_close(standard_deviation, benchmark)


def test_calculate_increments():
    standard_deviation = torch.Tensor(
        [
            [[1.0, 2.0], [2.0, 1.0]],
            [[2.0, 4.0], [4.0, 2.0]],
            [[3.0, 6.0], [6.0, 3.0]],
        ]
    )

    standard_normal = brownian_increment_generator._generate_standard_normal(
        size=(5, 3, 2), seed=0
    )

    brownian_increments = brownian_increment_generator._calculate_brownian_increments(
        standard_normal, standard_deviation
    )

    benchmark = torch.Tensor(
        [
            [
                [-1.9683237075805664, 1.349577784538269],
                [-2.9885921478271484, -7.534229755401611],
                [-4.320230960845947, -4.377163887023926],
            ],
            [
                [-4.092759132385254, -2.3574817180633545],
                [1.1028350591659546, -1.3707129955291748],
                [-3.153538703918457, -2.8414483070373535],
            ],
            [
                [0.012105509638786316, 0.2804141044616699],
                [-0.7387422323226929, -2.3252172470092773],
                [-4.0020670890808105, -10.037311553955078],
            ],
            [
                [-2.422741174697876, -1.313749074935913],
                [8.680110931396484, 8.5287446975708],
                [0.8840415477752686, -2.325535774230957],
            ],
            [
                [3.2769317626953125, 2.9035258293151855],
                [-3.940969467163086, -4.142290115356445],
                [-0.31446003913879395, -0.7131817936897278],
            ],
        ]
    )
    assert tensors_are_close(brownian_increments, benchmark)


# def test_brownian_increment_generator_call():
#    brownian_increment_generator()
