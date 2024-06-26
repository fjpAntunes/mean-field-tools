from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, BackwardSDE
import torch

torch.manual_seed(0)

NUMBER_OF_TIMESTEPS = 101
TIME_DOMAIN = torch.linspace(0, 1, NUMBER_OF_TIMESTEPS)
NUMBER_OF_PATHS = 100
SPATIAL_DIMENSIONS = 1

TERMINAL_CONDITION = lambda x: x**2

filtration = Filtration(SPATIAL_DIMENSIONS, TIME_DOMAIN, NUMBER_OF_PATHS)
filtration.generate_paths()

bsde = BackwardSDE(
    spatial_dimensions=SPATIAL_DIMENSIONS,
    time_domain=TIME_DOMAIN,
    terminal_condition_function=TERMINAL_CONDITION,
    filtration=filtration,
)

bsde.initialize_approximator()

approximate_solution = bsde.solve(
    approximator_args={
        "batch_size": 100,
        "number_of_iterations": 500,
        "number_of_epochs": 5,
        "number_of_plots": 5,
    }
)


def test_approximate_solution_shape():
    assert approximate_solution.shape == (
        NUMBER_OF_PATHS,
        NUMBER_OF_TIMESTEPS,
        SPATIAL_DIMENSIONS,
    )
