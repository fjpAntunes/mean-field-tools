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

    def __call__(self):
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
