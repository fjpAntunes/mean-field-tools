import torch
from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator
from mean_field_tools.deep_bsde.filtration import Filtration
from typing import Callable, List

# Maybe create a path class with time and value - (t,X_t) in general

DriftType = Callable[
    [Filtration],
    torch.Tensor,  # Shape should be (num_paths, path_length, spatial_dim)
]


def zero_drift(filtration: Filtration):
    return filtration.time_process * 0


class ForwardSDE:
    """Implements stochastic process of the form X_t = f(t, B_t)"""

    def __init__(
        self,
        filtration: Filtration,
        functional_form,
    ):
        self.filtration = filtration
        self.functional_form = functional_form

    def generate_paths(self, filtration: Filtration):
        self.paths = self.functional_form(filtration)
        return self.paths


class BackwardSDE:
    def __init__(
        self,
        terminal_condition_function: Callable[[Filtration], torch.Tensor],
        filtration: Filtration,
        exogenous_process=["time_process", "brownian_process"],
        drift: DriftType = zero_drift,  # Callable over tensors of shape (num_paths, path_length, time+spatial_dimension).
    ):
        self.terminal_condition_function = terminal_condition_function
        self.drift = drift
        self.filtration = filtration
        self.exogenous_process = exogenous_process

    def initialize_approximator(
        self, nn_args: dict = {}
    ):  # Maybe we could just pass a FunctionApproximator object on initialization
        number_of_spatial_processes = len(self.exogenous_process) - 1
        domain_dimensions = (
            1 + number_of_spatial_processes * self.filtration.spatial_dimensions
        )
        self.y_approximator = FunctionApproximator(
            domain_dimension=domain_dimensions,
            output_dimension=self.filtration.spatial_dimensions,
            **nn_args
        )

    def generate_paths(self):
        input = self.set_approximator_input()
        return self.y_approximator(input)

    def set_drift_path(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates drift path and backwards drift integral.

        Returns:
            self.drift_path : path of the drift function over the samples.
            self.drift_integral : backwards integral of the drift path over the samples.
        """
        self.drift_path = self.drift(self.filtration)
        total = torch.sum(self.drift_path, dim=1).unsqueeze(1)
        self.drift_integral = total - torch.cumsum(self.drift_path, dim=1)
        self.drift_integral = self.drift_integral * self.filtration.dt
        return self.drift_path, self.drift_integral

    def set_terminal_condition(self) -> torch.Tensor:
        """Calculates terminal condition for the BSDE

        Args:
            terminal_brownian : values of the exogenous process at terminal time. Shape should be (num_paths, num_of_spatial_dimensions)

        Returns:
            terminal_condition: value of the terminal condition of the BSDE for each of the sample paths. Shape should be (num_paths, num_dimensions)
        """
        self.terminal_condition = self.terminal_condition_function(self.filtration)

        return self.terminal_condition

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
        optimization_target = terminal_condition.unsqueeze(1)
        optimization_target = optimization_target + drift_integral

        return optimization_target

    def set_approximator_input(self):
        processes = [
            self.filtration.__dict__.get(name) for name in self.exogenous_process
        ]
        out = torch.cat(processes, dim=2)
        return out

    def solve(self, approximator_args: dict = None):
        _, drift_integral = self.set_drift_path()
        terminal_condition = self.set_terminal_condition()
        optimization_target = self.set_optimization_target(
            terminal_condition, drift_integral
        )
        optimization_input = self.set_approximator_input()

        self.y_approximator.minimize_over_sample(
            optimization_input, optimization_target, **approximator_args
        )


class ForwardBackwardSDE:
    """This class manipulates both forward and backward SDE objects in order to implement Picard iterations numerical scheme."""

    def __init__(
        self, filtration: Filtration, forward_sde: ForwardSDE, backward_sde: BackwardSDE
    ):
        self.filtration = filtration
        self.forward_sde = forward_sde
        self.backward_sde = backward_sde

    def _add_forward_process_to_filtration(self):
        forward_process = self.forward_sde.functional_form(self.filtration)
        self.filtration.forward_process = forward_process

    def backward_solve(self, approximator_args: dict = {}):
        self._add_forward_process_to_filtration()
        self.backward_sde.solve(approximator_args)
