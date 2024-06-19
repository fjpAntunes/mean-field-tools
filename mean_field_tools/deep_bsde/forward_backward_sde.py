import torch
from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator
from typing import Callable, List

# Maybe create a path class with time and value - (t,X_t) in general

DriftType = Callable[
    [torch.Tensor],  # Shape should be (num_paths, path_length, time_dim+spatial_dims)
    torch.Tensor,  # Shape should be (num_paths, path_length, spatial_dim)
]


def zero_drift(x):
    return x[:, :, 0] * 0


class StochasticProcess:
    pass


class Filtration:
    processes: list[StochasticProcess]

    def __init__(
        self,
        spatial_dimensions: int,
        time_domain,  # torch.linspace like
        number_of_paths,
    ):
        self.spatial_dimensions = spatial_dimensions
        self.time_domain = time_domain
        self.dt = self.time_domain[1] - self.time_domain[0]

        self.number_of_paths = number_of_paths

        (
            self.brownian_increments,
            self.brownian_process,
        ) = self.generate_brownian_process()
        self.time_process = self.generate_time_process()

        self.processes = [self.time_process, self.brownian_process]

    def generate_time_process(self):
        time_process = self.time_domain.repeat(repeats=(self.number_of_paths, 1))
        time_process = torch.unsqueeze(time_process, dim=-1)
        return time_process

    def generate_brownian_process(self):
        brownian_increments = (
            torch.randn(
                size=(
                    self.number_of_paths,
                    len(self.time_domain) - 1,
                    self.spatial_dimensions,
                )
            )
            * self.dt**0.5
        )
        brownian_increments = torch.cat(
            [
                torch.zeros(size=(self.number_of_paths, 1, self.spatial_dimensions)),
                brownian_increments,
            ],
            dim=1,
        )
        brownian_process = torch.cumsum(brownian_increments, axis=1)
        return brownian_increments, brownian_process

    def get_paths(self):
        return torch.cat(self.processes, dim=2)


class ForwardSDE:
    """Implements stochastic process of the form X_t = f(t, B_t)"""

    def __init__(
        self,
        filtration: Filtration,
        functional_form,
    ):
        self.filtration = filtration
        self.functional_form = functional_form

    def generate_paths(self):
        self.paths = self.functional_form(self.filtration)
        return


class BackwardSDE:
    def __init__(
        self,
        terminal_condition_function,  # Callable over space dimensions
        filtration: Filtration,
        drift: DriftType = zero_drift,  # Callable over tensors of shape (num_paths, path_length, time+spatial_dimension).
    ):
        self.terminal_condition_function = terminal_condition_function
        self.drift = drift
        self.filtration = filtration

    def initialize_approximator(self, nn_args: dict = {}):
        self.y_approximator = FunctionApproximator(
            domain_dimension=self.filtration.spatial_dimensions + 1,
            output_dimension=1,
            **nn_args
        )

    def generate_paths(self):
        return self.y_approximator(self.filtration.get_paths())

    def set_drift_path(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates drift path and backwards drift integral.

        Returns:
            self.drift_path : path of the drift function over the samples.
            self.drift_integral : backwards integral of the drift path over the samples.
        """
        self.drift_path = self.drift(self.filtration.get_paths())
        self.drift_integral = (
            torch.sum(self.drift_path, dim=1) - torch.cumsum(self.drift_path, dim=1).T
        )
        self.drift_integral = self.drift_integral * self.filtration.dt
        self.drift_integral = self.drift_integral.T.unsqueeze(-1)
        return self.drift_path, self.drift_integral

    def set_terminal_condition(self, terminal_brownian: torch.Tensor) -> torch.Tensor:
        """Calculates terminal condition for the BSDE

        Args:
            terminal_brownian : values of the exogenous process at terminal time. Shape should be (num_paths, num_of_spatial_dimensions)

        Returns:
            terminal_condition: value of the terminal condition of the BSDE for each of the sample paths.
        """
        terminal_condition = self.terminal_condition_function(terminal_brownian)
        return terminal_condition

    def set_optimization_target(
        self, terminal_condition: torch.Tensor, drift_integral: torch.Tensor
    ) -> torch.Tensor:
        """Calculates, for each sampled path and time t, the value
        $$ \\xi + \\int_t^T f_s ds $$
        which is the optimization target for the elicitability method of conditional expectation calculation.

        Args:
            terminal_condition : value of the terminal condition of the BSDE for each of the sample paths. Shape: (num_paths, num_spatial_dimensions)
            drift_integral : backwards integral of the drift path over the samples.

        Returns:
            optimization_target : optimization target for elicitability method of conditional expectation calculation.
        """
        optimization_target = terminal_condition + torch.swapaxes(drift_integral, 0, 2)
        optimization_target = torch.swapaxes(optimization_target, 0, 2)
        return optimization_target

    def solve(self, approximator_args: dict = None):
        _, drift_integral = self.set_drift_path()
        self.terminal_brownian = self.filtration.brownian_process[:, -1, 0]
        self.terminal_condition = self.set_terminal_condition(self.terminal_brownian)
        optimization_target = self.set_optimization_target(
            self.terminal_condition, drift_integral
        )
        self.y_approximator.minimize_over_sample(
            self.filtration.get_paths(), optimization_target, **approximator_args
        )

        return self.generate_paths()
