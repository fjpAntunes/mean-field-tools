from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, BackwardSDE
import torch

torch.manual_seed(0)

# Filtration
FILTRATION = Filtration(
    spatial_dimensions=1, time_domain=torch.linspace(0, 1, 101), number_of_paths=1
)
FILTRATION.generate_paths()


def test_path_shape():
    assert FILTRATION.brownian_paths.shape == (1, 101, 1)


def test_inital_value_equal_zero():
    assert FILTRATION.brownian_paths[:, 0, :] == torch.zeros(size=(1, 1))


def test_sample_paths():
    filtration = Filtration(
        spatial_dimensions=2, time_domain=torch.linspace(0, 1, 101), number_of_paths=10
    )
    filtration.generate_paths()
    sample = filtration.sample_paths(5)
    assert sample.shape == (5, 101, 2)


# BackwardSDE():

TIME_DOMAIN = torch.Tensor([0, 1, 2, 3])

mock_filtration = Filtration(
    spatial_dimensions=1, time_domain=TIME_DOMAIN, number_of_paths=3
)
mock_filtration.brownian_paths = torch.Tensor(
    [
        [0, 1, 2, 3],
        [0, 1, 0, 1],
        [0, -1, 0, -1],
    ]
).unsqueeze(-1)

bsde = BackwardSDE(
    spatial_dimensions=1,
    time_domain=TIME_DOMAIN,
    terminal_condition_function=lambda x: x**2,
    filtration=mock_filtration,
)


def test_set_terminal_condition():
    terminal_brownian = bsde.filtration.brownian_paths[:, -1, 0]
    terminal_condition = bsde.set_terminal_condition(terminal_brownian)

    assert torch.equal(terminal_condition, torch.Tensor([9, 1, 1]))


def test_set_optimization_target_dummy():
    optimization_target = bsde.set_optimization_target(
        terminal_condition=torch.Tensor([0, 1, 2])
    )

    benchmark_tensor = torch.Tensor(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
        ]
    ).unsqueeze(-1)
    assert torch.equal(optimization_target, benchmark_tensor)


def test_terminal_condition_and_optimization_target():
    terminal_brownian = bsde.filtration.brownian_paths[:, -1, 0]
    terminal_condition = bsde.set_terminal_condition(terminal_brownian)
    optimization_target = bsde.set_optimization_target(
        terminal_condition=terminal_condition
    )

    benchmark_tensor = torch.Tensor(
        [[9, 9, 9, 9], [1, 1, 1, 1], [1, 1, 1, 1]]
    ).unsqueeze(-1)

    assert torch.equal(optimization_target, benchmark_tensor)
