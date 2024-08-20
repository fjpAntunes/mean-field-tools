import matplotlib.pyplot as plt
from typing import Callable
import torch
import numpy as np

AnalyticalSolution = Callable[
    [
        torch.Tensor,  # Should be of size (num_samples)
        float,  # current_time
        float,  # terminal time
    ],
    torch.Tensor,  # Should be of size (num_samples)
]


class FunctionApproximatorArtist:
    def __init__(
        self, save_figures=False, analytical_solution: AnalyticalSolution = None
    ):
        self.save_figures = save_figures
        self.analytical_solution = analytical_solution

    def _handle_fig_output(self, path: str) -> None:
        if self.save_figures:
            plt.savefig(path)
        else:
            plt.plot()
            plt.show()

    def cast_to_np(self, tensor: torch.Tensor) -> np.ndarray:
        """Casts tensor to numpy array for plotting."""
        return tensor.detach().cpu().numpy()

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
        t = self.cast_to_np(sample[0, :, 0])
        for i in range(number_of_paths):
            x = self.cast_to_np(sample[i, :, 1])
            y = self.cast_to_np(approximator(sample[i, :, :]))
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
        t = self.cast_to_np(sample[0, :, 0])
        T = self.cast_to_np(sample[0, -1, 0])
        x = self.cast_to_np(sample[0, :, 1])
        y_hat = self.cast_to_np(approximator(sample[0, :, :]))
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
            T = self.cast_to_np(sample[0, -1, 0])
            t = T * (time_index / time_length)
            y_hat = approximator(sample)[:, time_index, 0]

            x = self.cast_to_np(x.reshape(-1))
            y_hat = self.cast_to_np(y_hat.reshape(-1))

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
