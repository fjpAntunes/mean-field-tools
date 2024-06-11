import torch
from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator


class Filtration:
    def __init__(
        self,
        spatial_dimensions: int,
        time_domain,  # torch.linspace like
        number_of_paths,
    ):
        self.spatial_dimensions = spatial_dimensions
        self.time_domain = time_domain
        self.number_of_paths = number_of_paths

    def generate_paths(self):
        dt = self.time_domain[1] - self.time_domain[0]
        self.brownian_increments = (
            torch.randn(
                size=(
                    self.number_of_paths,
                    len(self.time_domain) - 1,
                    self.spatial_dimensions,
                )
            )
            * dt**0.5
        )
        self.brownian_increments = torch.cat(
            [
                torch.zeros(size=(self.number_of_paths, 1, self.spatial_dimensions)),
                self.brownian_increments,
            ],
            dim=1,
        )
        self.path_at_t = torch.cumsum(self.brownian_increments, axis=1)
        t = self.time_domain.repeat(repeats=(self.number_of_paths, 1))
        t = torch.unsqueeze(t, dim=-1)
        self.brownian_paths = torch.cat([t, self.path_at_t], dim=2)


class BackwardSDE:
    def __init__(
        self,
        spatial_dimensions: int,
        time_domain,  # torch.linspace like
        terminal_condition_function,  # Callable over space dimensions
        # exogenous_process,
        filtration: Filtration,
        drift = lambda x: x[:,:,0]*0, # Callable over tensors of shape (num_paths, path_length, time+spatial_dimension).
    ):
        self.spatial_dimensions = spatial_dimensions
        self.time_domain = time_domain
        self.dt = self.time_domain[1] - self.time_domain[0]
        self.terminal_condition_function = terminal_condition_function
        self.drift = drift
        # self.exogenous_process = exogenous_process
        self.filtration = filtration

    def initialize_approximator(self, nn_args: dict = {}):
        self.y_approximator = FunctionApproximator(
            domain_dimension=self.spatial_dimensions + 1, output_dimension=1, **nn_args
        )

    def generate_paths(self):
        return self.y_approximator(self.filtration.brownian_paths)

    # def path_sampler(self, number_of_samples):
    #    indexes = torch.perm(self.filtration.number_of_paths)[:number_of_samples]
    #    return self.solution_paths[indexes,:,:]

    def set_drift_path(self):
        self.drift_path = self.drift(self.filtration.brownian_paths).unsqueeze(-1)
        self.drift_integral = torch.cumsum(self.drift_path * self.dt, dim = 1).unsqueeze(-1)
        return self.drift_path, self.drift_integral

    def set_terminal_condition(self, terminal_brownian):
        self.terminal_condition = self.terminal_condition_function(terminal_brownian)
        return self.terminal_condition

    def set_optimization_target(self, terminal_condition, drift_integral):
        optimization_target = terminal_condition + drift_integral[:,-1,:].squeeze()   
        optimization_target = optimization_target.repeat(
            repeats=(1, 1, len(self.time_domain))
        )
        optimization_target = optimization_target.reshape(
            (
                len(self.time_domain),
                self.filtration.number_of_paths,
                self.spatial_dimensions,
            )
        )
        optimization_target = torch.swapaxes(optimization_target, 0, 1)
        return optimization_target

    def solve(self, approximator_args: dict = None):
        _, drift_integral = self.set_drift_path()
        terminal_brownian = self.filtration.brownian_paths[:, -1, 1]
        terminal_condition = self.set_terminal_condition(terminal_brownian)
        optimization_target = self.set_optimization_target(terminal_condition, drift_integral)
        self.y_approximator.minimize_over_sample(
            self.filtration.brownian_paths, optimization_target, **approximator_args
        )

        return self.generate_paths()
