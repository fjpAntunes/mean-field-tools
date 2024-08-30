from mean_field_tools.deep_bsde.filtration import Filtration
import torch
from typing import Callable


def path_average_along_time(paths: torch.Tensor) -> torch.Tensor:
    average_along_time = torch.mean(paths, dim=0)
    one = torch.ones_like(paths)
    average_along_time = one * average_along_time
    return average_along_time


class MeasureFlow:
    def __init__(
        self,
        filtration: Filtration,
        parametrization: Callable[
            [
                torch.Tensor,
            ],
            torch.Tensor,
        ] = path_average_along_time,
    ):
        self.filtration = filtration
        self.parametrization = parametrization

    def _validate_parametrization_shape(self, parameterized_mean_field: torch.Tensor):
        number_of_paths = self.filtration.number_of_paths
        number_of_timesteps = len(self.filtration.time_domain)

        if number_of_paths != parameterized_mean_field.shape[0]:
            raise TypeError(f"Parameterization shape [0] should match number of paths")
        if number_of_timesteps != parameterized_mean_field.shape[1]:
            raise TypeError(
                f"Parameterization shape [1] should match number of timesteps"
            )

    def parameterize(self, paths: torch.Tensor):

        parameterized_mean_field = self.parametrization(paths)
        self._validate_parametrization_shape(parameterized_mean_field)
        return parameterized_mean_field
