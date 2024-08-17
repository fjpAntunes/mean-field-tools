import torch
from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator
from mean_field_tools.deep_bsde.filtration import Filtration
from typing import Callable, List

# Maybe create a path class with time and value - (t,X_t) in general

filtrationMeasurableFunction = Callable[
    [Filtration],
    torch.Tensor,  # Shape should be (num_paths, path_length, spatial_dim)
]


def zero_function(filtration: Filtration):
    return filtration.time_process * 0


class ForwardSDE:
    """Implements stochastic process of the form X_t = f(t, B_t)"""

    def __init__(
        self,
        filtration: Filtration,
        functional_form: filtrationMeasurableFunction,
        volatility_functional_form: filtrationMeasurableFunction = zero_function,
    ):
        self.filtration = filtration
        self.functional_form = functional_form
        self.volatility_functional_form = volatility_functional_form

    def generate_paths(self, filtration: Filtration):
        self.paths = self.functional_form(filtration)
        return self.paths

    def get_volatility(self):
        volatility = self.volatility_functional_form(self.filtration)
        return volatility


class BackwardSDE:
    r"""This class holds the logic related to the dynamics and solution through elicitability of a backward stochastic equation of the form
    $$
    d Y_t = -f(t,X_t,Y_t) dt + Z_t dW_t, \quad Y_T = \xi
    $$
    In general, the solution can be written as
    $$
    Y_t = \mathbb{E}[ \xi + \int_t^T f(s,X_s,Y_s) ds | \mathcal{F}_t ].
    $$
    In order to calculate this conditional expectation, we rewrite it as
    $$
    \mathbb{E}[ \xi + \int_t^T f(s,X_s,Y_s) ds | \mathcal{F}_t ] =  \arg \min_{Y \in \mathcal{F}_t} \mathbb{E}[(\xi + \int_t^T f(s,X_s,Y_s) ds - Y)^2]
    $$
    Now, $Y \in \mathcal{F}_t$ is a random variable - that is, it is a function
    $Y: \Omega \mapsto \mathbb{R}^d$ such that $Y^{-1}$ is $\mathcal{F}_t$ measurable.
    In order to find an approximate solution to the minimization problem, we minimize
    the expectation over the set of functions expressed as the output of neural networks.
    """

    def __init__(
        self,
        terminal_condition_function: Callable[[Filtration], torch.Tensor],
        filtration: Filtration,
        exogenous_process=["time_process", "brownian_process"],
        drift: filtrationMeasurableFunction = zero_function,  # Callable over tensors of shape (num_paths, path_length, time+spatial_dimension).
    ):
        r"""Initialization function of the class.

        Args:
            terminal_condition_function (Callable[[Filtration], torch.Tensor]): Terminal condition $\xi$ for the BSDE. Should be a function of the filtration.
            filtration (Filtration): Filtration object that holds the state of the processes in a time structured manner.
            exogenous_process (list, optional): Processes on which the BSDE depends.
              possible processes are "time_process", "brownian_process", "forward_process", "backward_process".
              Defaults to ["time_process", "brownian_process"].
            drift (DriftType, optional): Drift function $f$ for the BSDE. Note the sign convention. Defaults to zero_drift.
        """
        self.terminal_condition_function = terminal_condition_function
        self.drift = drift
        self.filtration = filtration
        self._set_exogenous_process(exogenous_process_list=exogenous_process)

    def _set_exogenous_process(self, exogenous_process_list: list):
        for process in exogenous_process_list:
            if process not in [
                "time_process",
                "brownian_process",
                "forward_process",
                "backward_process",
            ]:
                raise ValueError(
                    'Every element of `exogenous_process` must be one of "time_process", "brownian_process", "forward_process", "backward_process"'
                )

        self.exogenous_process = exogenous_process_list

    def initialize_approximator(
        self, nn_args: dict = {}
    ):  # Maybe we could just pass a FunctionApproximator object on initialization
        r"""Initializes FunctionApproximator neural net class to use in the elicitability solver.

        Args:
            nn_args (dict, optional): Optional args for FunctionApproximator class. Defaults to {}.
        """
        number_of_spatial_processes = len(self.exogenous_process) - 1
        domain_dimensions = (
            1 + number_of_spatial_processes * self.filtration.spatial_dimensions
        )
        self.y_approximator = FunctionApproximator(
            domain_dimension=domain_dimensions,
            output_dimension=self.filtration.spatial_dimensions,
            **nn_args
        )

    def generate_backward_process(self):
        input = self.set_approximator_input()
        return self.y_approximator.detached_call(input)

    def generate_backward_volatility(self):
        input = self.set_approximator_input()
        grad_y_wrt_x = self.y_approximator.grad(input)[:, :, 1:]
        if "brownian_process" in self.exogenous_process:
            return grad_y_wrt_x

        if "forward_process" in self.exogenous_process:
            volatility_of_x = self.filtration.forward_volatility
            return grad_y_wrt_x * volatility_of_x

    def set_drift_path(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the value of the drift $f(t,X_t,Y_t)$ for each given path,
          and the backwards drift integral $\\int_t^T f(s,X_s,Y_s) ds$.

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
        r"""Calculates terminal condition for the BSDE

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

    def set_approximator_input(self) -> torch.Tensor:
        """Creates tensor input for the FunctionApproximator object, using the processes over which the BSDE depends.

        In order to find an approximate solution to the minimization problem
        $$Y_t = \\arg \\min_{Y \\in \\mathcal{F}_t} \\mathbb{E}[(\\xi + \\int_t^T f(s,X_s,Y_s) ds - Y)^2]$$
        we parameterize $Y(\\omega), \\omega \\in \\mathcal{F}_t$ as the output of a neural network $NN_{w}(\\omega)$.

        This function creates a tensor to represent the input $\\omega$ for each equivalent time $t$ and sample path.
        Returns:
            torch.Tensor: input tensor for the FunctionApproximator object.
        """
        processes = [
            self.filtration.__dict__.get(name) for name in self.exogenous_process
        ]
        out = torch.cat(processes, dim=2)
        return out

    def solve(self, approximator_args: dict = None):
        """Performs the minimization step in order to calculate the conditional expectation through the elicitability method.


        Args:
            approximator_args (dict, optional): _description_. Defaults to None.
        """
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

    def _add_forward_volatility_to_filtration(self):
        self.filtration.forward_volatility = self.forward_sde.get_volatility()

    def _add_backward_process_to_filtration(self):
        backward_process = self.backward_sde.generate_backward_process()
        self.filtration.backward_process = backward_process

    def _add_backward_volatility_to_filtration(self):
        backward_volatility = self.backward_sde.generate_backward_volatility()
        self.filtration.backward_volatility = backward_volatility

    def _single_picard_step(self, approximator_args: dict = {}):
        """Performs a single step of the Picard operator for the backward SDE.

        The steps performed are:
        1. Calculate a new forward process $X^1_t$ using $X^0_t, Y^0_t$ as input, through Euler-Maruyama discretization, - not implemented
        2. Calculate a new backward process $Y^1_t$ using $X^0_t, Y^0_t$ as input, calculating the conditional expectation by elicitability.

        Args:
            approximator_args (dict, optional): Arguments for the neural network training. Defaults to {}.
        """
        self.backward_sde.solve(approximator_args)

    def backward_solve(self, number_of_iterations: int, approximator_args: dict = {}):
        """Solve the FBSDE system through Picard Iterations.


        In this case, the Picard iteration works as follows:
        1. Given an initial forward process $ X^0_t$, and an initial backward process $Y^0_t$:
        2. Calculate a new forward process $X^1_t$ using $X^0_t, Y^0_t$ as input, through Euler-Maruyama discretization,
        3. Calculate a new backward process $Y^1_t$ using $X^0_t, Y^0_t$ as input, calculating the conditional expectation by elicitability.
        4. Go back to step 1. After a number of iterations, or if convergence is achieved, end the algorithm.

        Args:
            number_of_iterations (int): _description_
            approximator_args (dict, optional): _description_. Defaults to {}.
        """
        self._add_forward_process_to_filtration()
        self._add_forward_volatility_to_filtration()
        if "backward_process" not in self.backward_sde.exogenous_process:
            self._single_picard_step(approximator_args)

        if "backward_process" in self.backward_sde.exogenous_process:
            self.filtration.backward_process = self.filtration.forward_process
            for _ in range(number_of_iterations):
                self._single_picard_step(approximator_args)
                self._add_backward_process_to_filtration()
                self._add_backward_volatility_to_filtration()
