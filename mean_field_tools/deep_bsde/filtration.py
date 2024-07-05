import torch


class StochasticProcess:
    pass


class BrownianIncrementGenerator:
    def __init__(
        self,
        number_of_paths: int,
        spatial_dimensions: int,
        sampling_times: int,
        spatial_covariance: torch.Tensor = None,
        seed=None,
    ):
        self.number_of_paths = number_of_paths
        self.spatial_dimensions = spatial_dimensions
        self.sampling_times = sampling_times
        self.dt = self.sampling_times[1:] - self.sampling_times[:-1]
        self.spatial_covariance = self._set_spatial_covariance(
            spatial_dimensions, spatial_covariance
        )
        self.seed = seed

    def _set_spatial_covariance(self, spatial_dimensions, spatial_covariance):
        if spatial_covariance is not None:
            return spatial_covariance
        else:
            return torch.eye(spatial_dimensions)

    def _set_standard_deviation(self, dt, covariance):
        spatial_standard_dev = torch.sqrt(covariance).unsqueeze(0)
        time_standard_dev = (dt**0.5).unsqueeze(-1).unsqueeze(-1)
        return time_standard_dev * spatial_standard_dev

    def _calculate_brownian_increments(self, standard_normal, standard_deviation):
        standard_normal = standard_normal.unsqueeze(-2)
        brownian_increments = torch.matmul(standard_normal, standard_deviation)
        return brownian_increments.squeeze(-2)

    def _generate_standard_normal(self, size, seed):
        if seed is not None:
            torch.manual_seed(seed)
        standard_normal = torch.randn(size=size)
        return standard_normal

    def __call__(self) -> torch.Tensor:
        size = (
            self.number_of_paths,
            len(self.sampling_times) - 1,
            self.spatial_dimensions,
        )
        standard_normal = self._generate_standard_normal(size, self.seed)
        standard_deviation = self._set_standard_deviation(
            self.dt, self.spatial_covariance
        )
        brownian_increments = self._calculate_brownian_increments(
            standard_normal, standard_deviation
        )
        return brownian_increments


class Filtration:
    """This class register the state of the system in a time-wise manner"""

    processes: list[StochasticProcess]

    def __init__(
        self,
        spatial_dimensions: int,
        time_domain,  # torch.linspace like
        number_of_paths,
        seed=None,
    ):
        generator = BrownianIncrementGenerator(
            number_of_paths=number_of_paths,
            spatial_dimensions=spatial_dimensions,
            sampling_times=time_domain,
            seed=seed,
        )
        self.spatial_dimensions = spatial_dimensions
        self.time_domain = time_domain
        self.dt = time_domain[1] - time_domain[0]
        self.number_of_paths = number_of_paths

        self.brownian_increments = generator()

        self.brownian_process = self._generate_brownian_process(
            self.brownian_increments
        )
        self.time_process = self._generate_time_process()

        self.forward_process = None
        self.backward_process = None

        self.processes = [
            self.time_process,
            self.brownian_process,
            self.forward_process,
            self.backward_process,
        ]

    def _generate_time_process(self):
        time_process = self.time_domain.repeat(repeats=(self.number_of_paths, 1))
        time_process = torch.unsqueeze(time_process, dim=-1)
        return time_process

    def _generate_brownian_process(self, brownian_increments):
        initial = torch.zeros(size=(self.number_of_paths, 1, self.spatial_dimensions))
        brownian_process = torch.cat(
            [initial, torch.cumsum(brownian_increments, axis=1)], dim=1
        )
        return brownian_process

    def get_paths(self):
        processes = []
        for process in [
            self.time_process,
            self.brownian_process,
            self.forward_process,
            self.backward_process,
        ]:
            if process is not None:
                processes.append(process)
        out = torch.cat(processes, dim=2)

        return out
