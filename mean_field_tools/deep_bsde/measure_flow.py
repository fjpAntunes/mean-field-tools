from mean_field_tools.deep_bsde.filtration import Filtration, CommonNoiseFiltration
from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator
import torch
from typing import Callable, List, Optional


def forward_process_path_average_along_time(filtration: Filtration) -> torch.Tensor:
    paths = filtration.forward_process
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
        ] = forward_process_path_average_along_time,
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

    def parameterize(self, filtration: Filtration):

        parameterized_mean_field = self.parametrization(filtration)
        self._validate_parametrization_shape(parameterized_mean_field)
        return parameterized_mean_field


class CommonNoiseMeasureFlow(MeasureFlow):
    def __init__(
        self,
        filtration: CommonNoiseFiltration,
        parametrization=None,
    ):
        self.filtration = filtration
        self.mean_approximators: List[FunctionApproximator] = []
        self.training_args_list: List[dict] = []
        self._use_single_network: bool = True

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

        self.mean_approximators = [self.mean_approximator]
        self.training_args_list = [training_args]
        self._use_single_network = True

    def initialize_approximators(
        self,
        approximators: Optional[List[FunctionApproximator]] = None,
        nn_args_list: Optional[List[dict]] = None,
        training_args_list: Optional[List[dict]] = None,
        output_dimensions: Optional[List[int]] = None,
    ):
        if output_dimensions is None:
            output_dimensions = [1] * self.filtration.spatial_dimensions

        n_networks = len(output_dimensions)

        if nn_args_list is None:
            nn_args_list = [{}] * n_networks
        if training_args_list is None:
            training_args_list = [{}] * n_networks

        assert len(nn_args_list) == n_networks, (
            f"nn_args_list length {len(nn_args_list)} != n_networks {n_networks}"
        )
        assert len(training_args_list) == n_networks, (
            f"training_args_list length {len(training_args_list)} != n_networks {n_networks}"
        )

        if approximators is not None:
            assert len(approximators) == n_networks, (
                f"approximators length {len(approximators)} != n_networks {n_networks}"
            )
            self.mean_approximators = list(approximators)
        else:
            domain_dimensions = 1 + self.filtration.spatial_dimensions
            self.mean_approximators = [
                FunctionApproximator(
                    domain_dimension=domain_dimensions,
                    output_dimension=out_dim,
                    **nn_args,
                )
                for out_dim, nn_args in zip(output_dimensions, nn_args_list)
            ]

        self.training_args_list = list(training_args_list)
        self.mean_approximator = self.mean_approximators[0]
        self._use_single_network = False

    def _set_elicitability_input(self) -> torch.Tensor:
        processes = [self.filtration.time_process, self.filtration.common_noise]
        out = torch.cat(processes, dim=2)
        return out

    def _elicit_mean_as_function_of_common_noise(self, paths):
        if self._use_single_network:
            self.mean_approximators[0].minimize_over_sample(
                self.elicitability_input, paths, **self.training_args_list[0]
            )
        else:
            if not isinstance(paths, list):
                paths = [paths] * len(self.mean_approximators)
            for approximator, target, training_args in zip(
                self.mean_approximators, paths, self.training_args_list
            ):
                approximator.minimize_over_sample(
                    self.elicitability_input, target, **training_args
                )

    def parameterize(self, filtration: Filtration):
        """Calculates conditional mean of `paths` using elicitability over the common noise."""
        paths = filtration.forward_process
        self.elicitability_input = self._set_elicitability_input()
        self._elicit_mean_as_function_of_common_noise(paths)
        if self._use_single_network:
            mean_field_parametrization = self.mean_approximators[0].detached_call(
                self.elicitability_input
            )
        else:
            components = [
                approx.detached_call(self.elicitability_input)
                for approx in self.mean_approximators
            ]
            mean_field_parametrization = torch.cat(components, dim=2)
        return mean_field_parametrization
