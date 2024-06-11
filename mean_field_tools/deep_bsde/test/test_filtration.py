from mean_field_tools.deep_bsde.forward_backward_sde import Filtration
import torch

torch.manual_seed(0)

# Filtration
FILTRATION = Filtration(
    spatial_dimensions=1, time_domain=torch.linspace(0, 1, 101), number_of_paths=1
)
FILTRATION.generate_paths()


def test_path_shape():
    assert FILTRATION.brownian_paths.shape == (1, 101, 2)


def test_inital_value_equal_zero():
    assert FILTRATION.brownian_paths[:, 0, 1] == torch.zeros(size=(1, 1))
