from mean_field_tools.deep_bsde.filtration import Filtration, CommonNoiseFiltration
from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator
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


class CommonNoiseMeasureFlow(MeasureFlow):
    def __init__(
        self,
        filtration: CommonNoiseFiltration,
        parametrization=None,
    ):
        self.filtration = filtration

    def initialize_approximator(
        self,
        approximator: FunctionApproximator = None,
        nn_args: dict = {},
        training_args={},
    ):
        self.nn_args = nn_args
        self.training_args = training_args

        domain_dimensions = 1 + self.filtration.spatial_dimensions
        if approximator is None:
            self.mean_approximator = FunctionApproximator(
                domain_dimension=domain_dimensions,
                output_dimension=self.filtration.spatial_dimensions,
                **nn_args,
            )
        else:
            self.mean_approximator = approximator

    def _set_elicitability_input(self) -> torch.Tensor:
        processes = [self.filtration.time_process, self.filtration.common_noise]
        out = torch.cat(processes, dim=2)
        return out

    def _elicit_mean_as_function_of_common_noise(self, paths):

        self.mean_approximator.minimize_over_sample(
            self.elicitability_input, paths, **self.training_args
        )

    def parameterize(self, paths: torch.Tensor):
        """Calculates conditional mean of `paths` using elicitability over the common noise."""
        self.elicitability_input = self._set_elicitability_input()
        self._elicit_mean_as_function_of_common_noise(paths)
        mean_field_parametrization = self.mean_approximator.detached_call(
            self.elicitability_input
        )
        
        return mean_field_parametrization
