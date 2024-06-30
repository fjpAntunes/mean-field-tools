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
        drift: DriftType = zero_drift,  # Callable over tensors of shape (num_paths, path_length, time+spatial_dimension).
    ):
        self.terminal_condition_function = terminal_condition_function
        self.drift = drift
        self.filtration = filtration

    def initialize_approximator(
        self, nn_args: dict = {}
    ):  # Maybe we could just pass a FunctionApproximator object on initialization
        self.y_approximator = FunctionApproximator(
            domain_dimension=self.filtration.spatial_dimensions + 1,
            output_dimension=1,
            **nn_args
        )

    def generate_paths(self, filtration: Filtration):
        return self.y_approximator(filtration.get_paths())

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
            terminal_condition: value of the terminal condition of the BSDE for each of the sample paths.
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
        optimization_target = terminal_condition.unsqueeze(-1)
        optimization_target = optimization_target + drift_integral

        return optimization_target

    def solve(self, approximator_args: dict = None):
        _, drift_integral = self.set_drift_path()
        terminal_condition = self.set_terminal_condition()
        optimization_target = self.set_optimization_target(
            terminal_condition, drift_integral
        )
        self.y_approximator.minimize_over_sample(
            self.filtration.get_paths(), optimization_target, **approximator_args
        )


class ForwardBackwardSDE:
    def __init__(
        self,
        filtration: Filtration,
        forward_functional_form,
        # forward_drift: DriftType,
        backward_drift,
        terminal_condition_function,
    ):
        self.forward_sde = ForwardSDE(
            filtration=filtration, functional_form=forward_functional_form
        )
        self.backward_sde = BackwardSDE(
            terminal_condition_function=terminal_condition_function,
            filtration=filtration,
            drift=lambda filtration: backward_drift(self, filtration),
        )

    def backward_solve(self, approximator_args: dict = None):
        self.backward_sde.solve(approximator_args)
