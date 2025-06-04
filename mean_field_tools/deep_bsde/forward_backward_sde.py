import torch
from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator
from mean_field_tools.deep_bsde.filtration import Filtration, CommonNoiseFiltration
from mean_field_tools.deep_bsde.measure_flow import MeasureFlow
from mean_field_tools.deep_bsde.artist import PicardIterationsArtist
from typing import Callable, List

# Maybe create a path class with time and value - (t,X_t) in general

filtrationMeasurableFunction = Callable[
    [Filtration],
    torch.Tensor,  # Shape should be (num_paths, path_length, spatial_dim)
]


def zero_function(filtration: Filtration):
    return filtration.time_process * 0


class ForwardSDE:
    def __init__(self, filtration: Filtration):
        self.filtration = filtration

    def solve(self):
        pass

    def generate_paths(self):
        pass

    def get_volatility(self):
        pass


class NumericalForwardSDE(ForwardSDE):
    """Implements numerical solution for stochastic differential equations
    through Picard iterations.
    """

    def __init__(
        self,
        filtration: Filtration,
        initial_value: filtrationMeasurableFunction,
        drift: filtrationMeasurableFunction,
        volatility: filtrationMeasurableFunction,
        tolerance: float = 1e-6,
    ):
        self.filtration = filtration
        self.initial_value = initial_value
        self.drift = drift
        self.volatility = volatility
        self.tolerance = tolerance

    def _initial_integral_term(self):
        initial = torch.zeros_like(self.filtration.brownian_process[:, 0, :]).unsqueeze(
            1
        )
        return initial

    def calculate_riemman_integral(self):
        initial = self._initial_integral_term()
        drift = self.drift(self.filtration)[:, :-1, :]
        dt = self.filtration.dt
        riemman_integral = torch.cat([initial, torch.cumsum(drift * dt, axis=1)], dim=1)
        return riemman_integral

    def calculate_ito_integral(self):
        initial = self._initial_integral_term()
        vol = self.volatility(self.filtration)[:, :-1, :]
        dBt = self.filtration.brownian_increments

        ito_integral = torch.cat([initial, torch.cumsum(vol * dBt, axis=1)], dim=1)

        return ito_integral

    def solve(self):
        if self.filtration.forward_process is None:
            self.filtration.forward_process = self.filtration.time_process

        delta = 1
        while delta > self.tolerance:
            paths = self.generate_paths()
            deviation = paths - self.filtration.forward_process
            delta = torch.mean(deviation**2) + deviation.var()
            self.filtration.forward_process = paths

    def generate_paths(self):
        initial_value = self.initial_value
        riemman_term = self.calculate_riemman_integral()

        ito_term = self.calculate_ito_integral()

        path = initial_value + riemman_term + ito_term

        return path

    def get_volatility(self):
        return self.volatility(self.filtration)


class AnalyticForwardSDE(ForwardSDE):
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

    def generate_paths(self):
        self.paths = self.functional_form(self.filtration)
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
        number_of_dimensions: int = None,
    ):
        r"""Initialization function of the class.

        Args:
            terminal_condition_function (Callable[[Filtration], torch.Tensor]): Terminal condition $\xi$ for the BSDE. Should be a function of the filtration.
            filtration (Filtration): Filtration object that holds the state of the processes in a time structured manner.
            exogenous_process (list, optional): Processes on which the BSDE depends.
              possible processes are "time_process", "brownian_process", "forward_process","forward_mean_field".
              Defaults to ["time_process", "brownian_process"].
            drift (DriftType, optional): Drift function $f$ for the BSDE. Note the sign convention. Defaults to zero_drift.
            number_of_dimensions (int): Number of dimensions of the process Y_t.
                Parameter defaults to none, in which case the process has the same number of dimensions as the brownian motion.
        """
        self.terminal_condition_function = terminal_condition_function
        self.drift = drift
        self.filtration = filtration
        self._set_exogenous_process(exogenous_process_list=exogenous_process)
        self._set_number_of_dimensions(number_of_dimensions)
        self.padding_size = len(self.filtration.time_domain) // 2

    def _set_exogenous_process(self, exogenous_process_list: list):
        for process in exogenous_process_list:
            if process not in [
                "time_process",
                "brownian_process",
                "forward_process",
                "forward_mean_field",
            ]:
                raise ValueError(
                    'Every element of `exogenous_process` must be one of "time_process", "brownian_process", "forward_process", "forward_mean_field"'
                )

        self.exogenous_process = exogenous_process_list

    def _set_number_of_dimensions(self, number_of_dimensions):
        if number_of_dimensions is None:
            self.number_of_dimensions = self.filtration.brownian_process.shape[-1]
        else:
            self.number_of_dimensions = number_of_dimensions

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
            output_dimension=self.number_of_dimensions,
            **nn_args,
        )

        return self.y_approximator

    def generate_backward_process(self):
        input = self.set_approximator_input()
        out = self.y_approximator.detached_call(input)
        out = self._remove_padding(out)
        return out

    def generate_backward_volatility(self):
        input = self.set_approximator_input()

        grad_y_wrt_x = self.y_approximator.grad(input)[
            :, :, 1 : 1 + self.number_of_dimensions
        ]

        grad_y_wrt_x = self._remove_padding(grad_y_wrt_x)
        if "brownian_process" in self.exogenous_process:
            out = grad_y_wrt_x

        if "forward_process" in self.exogenous_process:
            volatility_of_x = self.filtration.forward_volatility
            out = grad_y_wrt_x * volatility_of_x

        return out

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

        optimization_target = self._add_padding(optimization_target)

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
        out = self._add_padding(out)

        return out

    def calculate_volatility_integral(self) -> torch.Tensor:
        volatility = self.generate_backward_volatility()[:, :-1, :]
        increments = volatility * self.filtration.brownian_increments
        total = torch.sum(increments, dim=1).unsqueeze(1)
        self.volatility_integral = total - torch.cumsum(increments, dim=1)
        terminal = torch.zeros_like(self.volatility_integral[:, -1:, :])

        self.volatility_integral = torch.cat(
            [self.volatility_integral, terminal], dim=1
        )
        return self.volatility_integral

    def calculate_picard_operator(self) -> torch.Tensor:
        terminal_condition = self.set_terminal_condition().unsqueeze(1)
        _, drift_integral = self.set_drift_path()
        volatility_integral = self.calculate_volatility_integral()
        return terminal_condition + drift_integral - volatility_integral

    def _add_padding(self, tensor):
        right_padding = tensor[:, -1, :]
        right_padding = right_padding.reshape(tensor.shape[0], 1, tensor.shape[2])
        right_padding = right_padding.repeat(1, self.padding_size, 1)

        out = torch.cat([tensor, right_padding], dim=1)
        return out

    def _remove_padding(self, tensor):

        out = tensor[:, : -self.padding_size, :]

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


class CommonNoiseBackwardSDE(BackwardSDE):
    def __init__(
        self,
        terminal_condition_function: Callable[[Filtration], torch.Tensor],
        filtration: CommonNoiseFiltration,
        exogenous_process=["time_process", "brownian_process"],
        drift: filtrationMeasurableFunction = zero_function,  # Callable over tensors of shape (num_paths, path_length, time+spatial_dimension).
        number_of_dimensions: int = None,
    ):
        super().__init__(
            terminal_condition_function,
            filtration,
            exogenous_process,
            drift,
            number_of_dimensions,
        )
        self.filtration = filtration

    def initialize_approximator(self, nn_args={}):
        number_of_spatial_processes = len(self.exogenous_process) - 1
        domain_dimensions = (
            1 + number_of_spatial_processes * self.filtration.spatial_dimensions
        )
        self.y_approximator = FunctionApproximator(
            domain_dimension=domain_dimensions,
            output_dimension=self.number_of_dimensions,
            **nn_args,
        )

        self.z_approximator = FunctionApproximator(
            domain_dimension=domain_dimensions,
            output_dimension=self.number_of_dimensions,
            **nn_args,
        )

        self.z_zero_approximator = FunctionApproximator(
            domain_dimension=domain_dimensions,
            output_dimension=self.number_of_dimensions,
            **nn_args,
        )
        return self.y_approximator, self.z_approximator, self.z_zero_approximator

    def _check_if_common_noise_filtration(self):
        filtration_type = type(self.filtration).__name__
        if filtration_type != "CommonNoiseFiltration":
            raise ValueError("Filtration should be of the CommonNoiseFiltration class")

    def set_z_optimization_target(
        self,
        terminal_condition: torch.Tensor,
        drift_integral: torch.Tensor,
        brownian_motion: torch.Tensor,
    ):

        backward_process = self.generate_backward_process()

        backward_delta = backward_process[:, 1:, :] - backward_process[:, :-1, :]

        optimization_target = (backward_delta / self.filtration.dt) + self.drift_path[
            :, :-1, :
        ]

        brownian_delta = brownian_motion[:, 1:, :] - brownian_motion[:, :-1, :]

        optimization_target = optimization_target * brownian_delta

        optimization_target = torch.cat(
            [
                torch.zeros_like(optimization_target[:, 0, :].unsqueeze(1)),
                optimization_target,
            ],
            dim=1,
        )

        optimization_target = self._add_padding(optimization_target)

        return optimization_target

    def solve_for_z(
        self,
        approximator: FunctionApproximator,
        brownian: torch.Tensor,
        approximator_args: dict = None,
    ):
        _, drift_integral = self.set_drift_path()
        terminal_condition = self.set_terminal_condition()

        optimization_target = self.set_z_optimization_target(
            terminal_condition, drift_integral, brownian
        )
        optimization_input = self.set_approximator_input()
        approximator.minimize_over_sample(
            optimization_input, optimization_target, **approximator_args
        )

    def solve_for_idiosyncratic_volatility(self, approximator_args: dict = None):
        self.solve_for_z(
            self.z_approximator, self.filtration.idiosyncratic_noise, approximator_args
        )

    def solve_for_common_volatility(self, approximator_args: dict = None):
        self.solve_for_z(
            self.z_zero_approximator, self.filtration.common_noise, approximator_args
        )

    def _calculate_volatility(
        self, z_approximator: FunctionApproximator
    ) -> torch.Tensor:
        input = self.set_approximator_input()
        z_hat = z_approximator.detached_call(input)
        z_hat = self._remove_padding(z_hat)
        return z_hat

    def generate_common_noise_volatility(self) -> torch.Tensor:
        return self._calculate_volatility(self.z_zero_approximator)

    def generate_idiosyncratic_noise_volatility(self) -> torch.Tensor:
        return self._calculate_volatility(self.z_approximator)

    def solve(self):
        super().solve()
        # Solve for Z
        # Solve for Z_0


class ForwardBackwardSDE:
    """This class manipulates both forward and backward SDE objects in order to implement Picard iterations numerical scheme."""

    def __init__(
        self,
        filtration: Filtration,
        forward_sde: ForwardSDE,
        backward_sde: BackwardSDE,
        measure_flow: MeasureFlow = None,
        damping: Callable[[int], float] = lambda i: 0,
    ):
        self.filtration = filtration
        self.forward_sde = forward_sde
        self.backward_sde = backward_sde
        self.measure_flow = measure_flow
        self.damping = damping
        self.iteration = 0

    def _damping_update(self, current, update):
        if current is None:
            return update

        coefficient = self.damping(self.iteration)

        damped_update = coefficient * current + (1 - coefficient) * update

        return damped_update

    def _add_forward_process_to_filtration(self):
        updated_forward_process = self.forward_sde.generate_paths()
        damped_update_forward_process = self._damping_update(
            current=self.filtration.forward_process,
            update=updated_forward_process,
        )
        self.filtration.forward_process = damped_update_forward_process

    def _add_forward_volatility_to_filtration(self):
        updated_forward_volatility = self.forward_sde.get_volatility()
        damped_update_forward_volatility = self._damping_update(
            current=self.filtration.forward_volatility,
            update=updated_forward_volatility,
        )
        self.filtration.forward_volatility = damped_update_forward_volatility

    def _add_forward_mean_field_to_filtration(self):
        if self.measure_flow is not None:
            updated_forward_mean_field = self.measure_flow.parameterize(
                self.filtration.forward_process.detach()
            )
            damped_update_forward_mean_field = self._damping_update(
                current=self.filtration.forward_mean_field,
                update=updated_forward_mean_field,
            )
            self.filtration.forward_mean_field = damped_update_forward_mean_field

    def _add_backward_process_to_filtration(self):
        updated_backward_process = self.backward_sde.generate_backward_process()
        damped_update_backward_process = self._damping_update(
            current=self.filtration.backward_process,
            update=updated_backward_process,
        )

        self.filtration.backward_process = damped_update_backward_process

    def _add_backward_volatility_to_filtration(self):
        updated_backward_volatility = self.backward_sde.generate_backward_volatility()
        damped_updated_backward_volatility = self._damping_update(
            current=self.filtration.backward_volatility,
            update=updated_backward_volatility,
        )

        self.filtration.backward_volatility = damped_updated_backward_volatility

    def _single_picard_step(self, approximator_args: dict = {}):
        """Performs a single step of the Picard operator for the backward SDE.

        The steps performed are:
        1. Calculate a new forward process $X^1_t$ using $X^0_t, Y^0_t$ as input, through Euler-Maruyama discretization, - not implemented
        2. Calculate a new backward process $Y^1_t$ using $X^0_t, Y^0_t$ as input, calculating the conditional expectation by elicitability.

        Args:
            approximator_args (dict, optional): Arguments for the neural network training. Defaults to {}.
        """
        self.forward_sde.solve()
        self.backward_sde.solve(approximator_args)

    def _initialize_forward_process(self, forward_process, forward_volatility):
        if forward_process is None:
            self.filtration.forward_process = self.filtration.brownian_process
        else:
            self.filtration.forward_process = forward_process

        if forward_volatility is None:
            self.filtration.forward_volatility = torch.ones_like(
                self.filtration.time_process
            )
        else:
            self.filtration.forward_volatility = forward_volatility
        self._add_forward_mean_field_to_filtration()

    def _initialize_backward_process(self, backward_process, backward_volatility):
        if backward_process is None:
            self.filtration.backward_process = self.filtration.time_process
        else:
            self.filtration.backward_process = backward_process

        if backward_volatility is None:
            self.filtration.backward_volatility = torch.ones_like(
                self.filtration.brownian_process
            )
        else:
            self.filtration.backward_volatility = backward_volatility

    def _update_states(self):
        self._add_backward_process_to_filtration()
        self._add_backward_volatility_to_filtration()
        self._add_forward_process_to_filtration()
        self._add_forward_volatility_to_filtration()
        self._add_forward_mean_field_to_filtration()

    def backward_solve(
        self,
        number_of_iterations: int,
        initial_forward_process=None,
        initial_forward_volatility=None,
        initial_backward_process=None,
        initial_backward_volatility=None,
        plotter: PicardIterationsArtist = None,
        approximator_args: dict = {},
        end_of_iteration_callback=None,
    ):
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
        self._initialize_forward_process(
            initial_forward_process, initial_forward_volatility
        )
        self._initialize_backward_process(
            initial_backward_process, initial_backward_volatility
        )
        for i in range(number_of_iterations):
            self.iteration = i
            self._single_picard_step(approximator_args)
            self._update_states()
            if plotter is not None:
                plotter.end_of_iteration_callback(fbsde=self, iteration=i)
            if end_of_iteration_callback is not None:
                end_of_iteration_callback()
        if plotter is not None:
            plotter.end_of_solver_callback(fbsde=self)
