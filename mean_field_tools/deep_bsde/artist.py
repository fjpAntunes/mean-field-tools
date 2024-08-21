import matplotlib.pyplot as plt
from typing import Callable
import torch
import numpy as np
from mean_field_tools.deep_bsde.filtration import Filtration

AnalyticalSolution = Callable[
    [
        torch.Tensor,  # Should be of size (num_samples)
        float,  # current_time
        float,  # terminal time
    ],
    torch.Tensor,  # Should be of size (num_samples)
]


def cast_to_np(tensor: torch.Tensor) -> np.ndarray:
    """Casts tensor to numpy array for plotting."""
    return tensor.detach().cpu().numpy()


class FunctionApproximatorArtist:
    def __init__(
        self,
        filtration: Filtration = None,
        save_figures=False,
        analytical_solution: AnalyticalSolution = None,
    ):
        self.save_figures = save_figures
        self.filtration = filtration
        self.analytical_solution = analytical_solution

    def _handle_fig_output(self, path: str) -> None:
        if self.save_figures:
            plt.savefig(path)
        else:
            plt.plot()
            plt.show()

    def plot_loss_history(self, number, loss_history):
        fig, axs = plt.subplots()
        iteration = range(len(loss_history))
        axs.set_title("Loss history")
        axs.plot(iteration, loss_history)
        axs.set_yscale("log")
        path = f"./.figures/loss_plot_{number}"
        self._handle_fig_output(path)
        plt.close()

    def plot_paths(
        self,
        approximator: torch.nn.Module,
        sample: torch.Tensor,
        number_of_paths: int,
        iteration: int,
    ):
        """Plots paths of forward proccess and backward process along training.

        Args:
            approximator (nn.Module): function approximator object.
            sample (torch.Tensor): sample used in training
            number_of_paths (int): number of paths to be plotted.
            iteration (int): training iteration.
        """
        fig, axs = plt.subplots(2, 1, layout="constrained")
        t = cast_to_np(sample[0, :, 0])
        for i in range(number_of_paths):
            x = cast_to_np(sample[i, :, 1])
            y = cast_to_np(approximator(sample[i, :, :]))
            axs[0].plot(t, x)
            axs[1].plot(t, y)

        axs[0].set_title("Forward process sample paths")
        axs[1].set_title("Backward process sample paths")

        path = f"./.figures/sample_path_{iteration}.png"
        self._handle_fig_output(path)

        plt.close()

    def plot_single_path(
        self,
        approximator: torch.nn.Module,
        sample: torch.Tensor,
        iteration: int,
    ):
        """Plot a single realization of the forward process,
        the analytical solution for the backward proccess
        and the estimated backward process.

        Args:
            approximator (nn.Module): function approximator object.
            sample (torch.Tensor): sample used in training
            number_of_paths (int): number of paths to be plotted.
            iteration (int): training iteration.
        """
        fig, axs = plt.subplots(2, 1, layout="constrained")
        t = cast_to_np(sample[0, :, 0])
        T = cast_to_np(sample[0, -1, 0])
        x = cast_to_np(sample[0, :, 1])
        y_hat = cast_to_np(approximator(sample[0, :, :]))
        axs[0].plot(t, x, label="Forward Process")
        axs[1].plot(t, y_hat, color="b", label="Backward Process - Approximation")
        if self.analytical_solution:
            y = self.analytical_solution(x, t, T)
            axs[1].plot(t, y, color="r", label="Backward Process - Analytical")

        for i in [0, 1]:
            axs[i].legend()
        path = f"./.figures/single_path_{iteration}.png"
        self._handle_fig_output(path)

        plt.close()

    def plot_fit_against_analytical(self, approximator, sample, number):
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        _, time_length, _ = sample.shape
        for i, time_index in enumerate([0, time_length // 2, time_length - 1]):
            x = sample[:, time_index, 1]
            T = cast_to_np(sample[0, -1, 0])
            t = T * (time_index / time_length)
            y_hat = approximator(sample)[:, time_index, 0]

            x = cast_to_np(x.reshape(-1))
            y_hat = cast_to_np(y_hat.reshape(-1))

            # axs[i].set_ylim(0, 2)
            # axs[i].set_xlim(-1.5, 1.5)
            axs[i].set_title(f"t = {np.round(t, decimals=1)}")
            axs[i].scatter(x, y_hat, color="b", s=0.5, label="Approximation")
            if self.analytical_solution:
                y = self.analytical_solution(x, t, T)  # x**2 + (T - t)
                axs[i].scatter(x, y, color="r", s=0.5, label="Analytical")
            axs[i].legend()

        path = f"./.figures/fit_plot_{number}.png"

        self._handle_fig_output(path)
        plt.close()

    def make_training_plots(self, approximator, sample, iteration):
        self.plot_loss_history(iteration, approximator.loss_history)
        self.plot_fit_against_analytical(approximator, sample, iteration)
        self.plot_paths(
            approximator=approximator,
            sample=sample,
            number_of_paths=20,
            iteration=iteration,
        )
        self.plot_single_path(
            approximator=approximator, sample=sample, iteration=iteration
        )


class PicardIterationsArtist:
    def __init__(
        self,
        filtration: Filtration,
        analytical_backward_solution: Callable[[Filtration], torch.Tensor] = None,
        analytical_backward_volatility: Callable[[Filtration], torch.Tensor] = None,
    ):
        self.filtration = filtration

        self.analytical_backward_solution = analytical_backward_solution
        self.analytical_backward_volatility = analytical_backward_volatility

    def violin_plot(self, ax, time, errors, quantile_value):
        boundary = np.quantile(
            errors, [0.5 - quantile_value / 2, 0.5 + quantile_value / 2], axis=0
        ).T
        ax.fill_between(
            time,
            boundary[:, 0],
            boundary[:, 1],
            color="r",
            alpha=1 - quantile_value,
            label=f"{quantile_value:.0%} of values",
        )

    def quantiles_along_time(self, ax, time, errors, quantile_value):
        ax.fill_between(
            time,
            np.quantile(errors, quantile_value, axis=0).T,
            color="r",
            alpha=1 - quantile_value,
            label=f"{quantile_value:.0%} of values",
        )

    def calculate_errors(self):
        y = self.analytical_backward_solution(self.filtration)
        y_hat = self.filtration.backward_process
        error_y = y_hat - y
        error_y = cast_to_np(error_y)[:, :, 0]

        z = self.analytical_backward_volatility(self.filtration)
        z_hat = self.filtration.backward_volatility
        error_z = z_hat - z
        error_z = cast_to_np(error_z)[:, :, 0]

        return error_y, error_z

    def plot_error_along_time(self):
        _, axs = plt.subplots(4, 1, figsize=(12, 16))

        error_y, error_z = self.calculate_errors()
        quadratic_error_y = error_y**2
        quadratic_error_z = error_z**2

        t = cast_to_np(self.filtration.time_domain)

        quantile_values = [0.5, 0.9, 0.95]
        for value in quantile_values:

            # Violin plot
            self.violin_plot(axs[0], t, error_y, value)
            self.violin_plot(axs[2], t, error_z, value)

            # squared errors along time
            self.quantiles_along_time(axs[1], t, quadratic_error_y, value)
            self.quantiles_along_time(axs[3], t, quadratic_error_z, value)

        for i in range(4):
            axs[i].legend()
            axs[i].grid(True)
        axs[0].set_title(
            f"Quantile of errors along time - Iteration {self.iteration + 1}"
        )
        axs[0].set_ylabel(r"$(\hat Y - Y)$")
        axs[1].set_ylabel(r"$(\hat Y - Y)^2$")
        axs[2].set_ylabel(r"$(\hat Z - Z)$")
        axs[3].set_ylabel(r"$(\hat Z - Z)^2$")
        axs[3].set_xlabel("Time")
        plt.savefig(f"./.figures/error_quantiles_plot_{self.iteration}.png")
        plt.close()

    def plot_error_histogram(self):
        _, axs = plt.subplots(2, 1, figsize=(4, 8))

        error_y, error_z = self.calculate_errors()
        n_bins = 50
        axs[0].hist(error_y.reshape(-1), bins=n_bins, density=True)
        axs[1].hist(error_z.reshape(-1), bins=n_bins, density=True)

        for i in range(2):
            axs[i].grid(True)
            axs[i].set_xlim(-1, 1)
        plt.savefig(f"./.figures/error_histogram_{self.iteration}.png")
        plt.close()

    def plot_picard_operator_error(self):
        _, axs = plt.subplots(1, 1, figsize=(12, 4))

        y = self.filtration.backward_process

    def end_of_iteration_callback(self, fbsde, iteration):
        self.iteration = iteration
        self.fbsde = fbsde
        self.plot_error_along_time()
        self.plot_error_histogram()
        self.plot_picard_operator_error()
