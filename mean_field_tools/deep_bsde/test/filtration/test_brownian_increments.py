from mean_field_tools.deep_bsde.filtration import Filtration, BrownianIncrementGenerator
from mean_field_tools.deep_bsde.utils import tensors_are_close, L_inf_norm
import torch

TIME_DOMAIN = torch.linspace(0, 1, 4)

FILTRATION = Filtration(
    spatial_dimensions=2, time_domain=TIME_DOMAIN, number_of_paths=5
)

covariance = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
brownian_increment_generator = BrownianIncrementGenerator(
    number_of_paths=5,
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


def test_generate_standard_normal():
    pure_function = lambda: brownian_increment_generator._generate_standard_normal(
        size=(5, 3, 2), seed=0
    )
    assert tensors_are_close(pure_function(), pure_function())


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
                [-3.4305601119995117, -3.4040398597717285],
                [-2.2366724014282227, -1.8700718879699707],
                [6.698185920715332, 7.168289661407471],
            ],
            [
                [-4.546451568603516, -2.7472448348999023],
                [-4.408789157867432, -1.237569808959961],
                [2.8987531661987305, 3.0243008136749268],
            ],
            [
                [2.5951573848724365, 1.4773409366607666],
                [-0.7333850264549255, -0.7971140742301941],
                [-1.6511927843093872, -3.5867202281951904],
            ],
            [
                [0.004152536392211914, -0.7369391918182373],
                [1.3290364742279053, 1.9836057424545288],
                [-16.419445037841797, -11.994514465332031],
            ],
            [
                [1.4825782775878906, 0.5878247022628784],
                [-0.36930543184280396, -1.0536558628082275],
                [15.381814002990723, 10.043779373168945],
            ],
        ]
    )
    assert tensors_are_close(brownian_increments, benchmark)


def test_brownian_increment_generator_call():
    brownian_increment_generator = BrownianIncrementGenerator(
        number_of_paths=5,
        spatial_dimensions=3,
        sampling_times=TIME_DOMAIN,
        spatial_covariance=covariance,
        seed=0,
    )
    increments = brownian_increment_generator()
    benchmark = torch.Tensor(
        [
            [
                [-0.7946755290031433, -0.6653154492378235, -0.7946755290031433],
                [0.1490316092967987, 0.4900031089782715, 0.1490316092967987],
                [0.003615453839302063, -1.2212225198745728, 0.003615453839302063],
            ],
            [
                [-0.5514854192733765, 0.20206288993358612, -0.5514854192733765],
                [0.7139620184898376, 0.7145620584487915, 0.7139620184898376],
                [-1.1219124794006348, -0.7809550762176514, -1.1219124794006348],
            ],
            [
                [0.6728960275650024, 0.45813223719596863, 0.6728960275650024],
                [0.17199891805648804, -0.1970844864845276, 0.17199891805648804],
                [0.3330114483833313, -0.3380371928215027, 0.3330114483833313],
            ],
            [
                [-0.28594326972961426, 0.8021509051322937, -0.28594326972961426],
                [0.15309640765190125, 0.2822490930557251, 0.15309640765190125],
                [0.6066074967384338, 0.6093109846115112, 0.6066074967384338],
            ],
            [
                [-1.0457714796066284, -0.17707574367523193, -1.0457714796066284],
                [1.4148913621902466, -0.2576235830783844, 1.4148913621902466],
                [-0.005891740322113037, 1.9690548181533813, -0.005891740322113037],
            ],
        ]
    )

    assert tensors_are_close(increments, benchmark)
