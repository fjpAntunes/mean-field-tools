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
        time_domain: torch.Tensor,  # torch.linspace like
        number_of_paths: int,
        seed=None,
    ):
        self.brownian_increment_generator = BrownianIncrementGenerator(
            number_of_paths=number_of_paths,
            spatial_dimensions=spatial_dimensions,
            sampling_times=time_domain,
            seed=seed,
        )
        self.spatial_dimensions = spatial_dimensions
        self.time_domain = time_domain
        self.dt = time_domain[1] - time_domain[0]
        self.number_of_paths = number_of_paths

        self.brownian_increments = self.brownian_increment_generator()

        self.brownian_process = self._generate_brownian_process(
            self.brownian_increments
        )
        self.time_process = self._generate_time_process()

        self.forward_process = None
        self.forward_volatility = None
        self.backward_process = None
        self.backward_volatility = None
        self.forward_mean_field = None

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


class CommonNoiseFiltration(Filtration):
    "This class register the state of the system in a time-wise manner subject to a common noise"

    def __init__(
        self,
        spatial_dimensions: int,
        time_domain: torch.Tensor,
        number_of_paths: int,
        common_noise_coefficient: float,
        seed=None,
    ):
        super().__init__(spatial_dimensions, time_domain, number_of_paths, seed)

        self.brownian_increment_generator = BrownianIncrementGenerator(
            number_of_paths=number_of_paths,
            spatial_dimensions=2 * spatial_dimensions,
            sampling_times=time_domain,
            seed=seed,
        )

        self.common_noise_coefficient = common_noise_coefficient

        increments = self.brownian_increment_generator()

        self.common_noise_increments = increments[:, :, :spatial_dimensions]
        self.idiosyncratic_noise_increments = increments[:, :, spatial_dimensions:]

        self.common_noise = self._generate_brownian_process(
            self.common_noise_increments
        )

        self.idiosyncratic_noise = self._generate_brownian_process(
            self.idiosyncratic_noise_increments
        )

        self.brownian_increments = (
            self.common_noise_coefficient * self.common_noise_increments
            + ((1 - self.common_noise_coefficient**2) ** 0.5)
            * self.idiosyncratic_noise_increments
        )

        self.brownian_process = self._generate_brownian_process(
            self.brownian_increments
        )
