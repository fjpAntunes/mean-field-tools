from mean_field_tools.deep_bsde.measure_flow import CommonNoiseMeasureFlow
from mean_field_tools.deep_bsde.filtration import CommonNoiseFiltration
from mean_field_tools.deep_bsde.artist import PicardIterationsArtist, cast_to_np
from mean_field_tools.deep_bsde.utils import L_2_norm

import torch
import numpy as np


torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
NUMBER_OF_TIMESTEPS = 101
NUMBER_OF_PATHS = 10_000
TIME_DOMAIN = torch.linspace(0, 1, NUMBER_OF_TIMESTEPS)
SPATIAL_DIMENSIONS = 1

FILTRATION = CommonNoiseFiltration(
    spatial_dimensions=SPATIAL_DIMENSIONS,
    time_domain=TIME_DOMAIN,
    number_of_paths=NUMBER_OF_PATHS,
    common_noise_coefficient=0.30,
    seed=0,
)

error = FILTRATION.brownian_process - FILTRATION.common_noise

import pdb

pdb.set_trace()

measure_flow = CommonNoiseMeasureFlow(filtration=FILTRATION)
measure_flow.initialize_approximator(
    training_args={
        "training_strategy_args": {
            "batch_size": 512,
            "number_of_iterations": 200,
            "number_of_batches": 200,
        }
    },
)


rho = FILTRATION.common_noise_coefficient
common_noise = FILTRATION.common_noise
process = FILTRATION.brownian_process


conditional_mean = measure_flow.parameterize(process)
deviation = conditional_mean - rho * common_noise


artist = PicardIterationsArtist(filtration=FILTRATION)


t = cast_to_np(FILTRATION.time_domain)
error = cast_to_np(deviation.squeeze(-1))

import matplotlib.pyplot as plt

_, axs = plt.subplots(1, 1, figsize=(12, 5))


quantile_values = [0.5, 0.8, 0.95]
for value in quantile_values:

    # Violin plot
    artist.violin_plot(axs, t, error, value)

axs.legend()
axs.grid(True)
axs.set_title(f"Quantile of errors along time")
axs.set_ylabel(r"$(\hat X - X)$")
axs.set_xlabel("Time")
plt.savefig(f"./.figures/error_quantiles_along_time_measure_flow.png")
plt.close()


_, axs = plt.subplots(1, 1, figsize=(12, 5))

single_path = cast_to_np(conditional_mean[0, :, 0])

axs.plot(t, single_path, label="approximation")

analytical_path = cast_to_np(rho * common_noise[0, :, 0])

axs.plot(t, analytical_path, label="analytical")
axs.legend()
axs.grid(True)
axs.set_xlabel("Time")
plt.savefig(f"./.figures/measure_flow_single_path.png")
plt.close()
