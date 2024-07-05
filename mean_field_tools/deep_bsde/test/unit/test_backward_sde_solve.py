from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, BackwardSDE
from mean_field_tools.deep_bsde.utils import QUADRATIC_TERMINAL
import torch


NUMBER_OF_TIMESTEPS = 101
TIME_DOMAIN = torch.linspace(0, 1, NUMBER_OF_TIMESTEPS)
NUMBER_OF_PATHS = 100
SPATIAL_DIMENSIONS = 1


filtration = Filtration(SPATIAL_DIMENSIONS, TIME_DOMAIN, NUMBER_OF_PATHS, seed=0)

bsde = BackwardSDE(
    terminal_condition_function=QUADRATIC_TERMINAL,
    filtration=filtration,
)

bsde.initialize_approximator()

bsde.solve(
    approximator_args={
        "batch_size": 100,
        "number_of_iterations": 500,
        "number_of_epochs": 5,
        "number_of_plots": 5,
    }
)

approximate_solution = bsde.generate_paths()


def test_approximate_solution_shape():
    assert approximate_solution.shape == (
        NUMBER_OF_PATHS,
        NUMBER_OF_TIMESTEPS,
        SPATIAL_DIMENSIONS,
    )
