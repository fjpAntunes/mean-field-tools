import matplotlib.pyplot as plt
import matplotlib as mpl
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
        analytical_forward_solution: Callable[[Filtration], torch.Tensor] = None,
        output_folder: str = "./.figures",
    ):
        self.filtration = filtration

        self.analytical_backward_solution = analytical_backward_solution
        self.analytical_forward_solution = analytical_forward_solution
        self.analytical_backward_volatility = analytical_backward_volatility

        self.path_register = {
            "x_hat": [],
            "y_hat": [],
            "z_hat": [],
        }

        self.color_map = mpl.colormaps["Blues"]
        self.errors = {"error_x": [], "error_y": [], "error_m": []}
        self.error_plot_iterations = [0, 1, 2]
        self.output_folder = output_folder

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
        error_x, error_y, error_z = (None, None, None)
        if self.analytical_forward_solution is not None:
            x = self.analytical_forward_solution(self.filtration)
            x_hat = self.filtration.forward_process
            error_x = x_hat - x
            error_x = cast_to_np(error_x)[:, :, 0]

        if self.analytical_backward_solution is not None:
            y = self.analytical_backward_solution(self.filtration)
            y_hat = self.filtration.backward_process
            error_y = y_hat - y
            error_y = cast_to_np(error_y)[:, :, 0]

        if self.analytical_backward_volatility is not None:
            z = self.analytical_backward_volatility(self.filtration)
            z_hat = self.filtration.backward_volatility
            error_z = z_hat - z
            error_z = cast_to_np(error_z)[:, :, 0]

        return error_x, error_y, error_z

    def plot_error_quantiles_along_time(self):
        if self.analytical_backward_solution is None:
            return
        _, axs = plt.subplots(3, 1, figsize=(12, 16))

        error_x, error_y, error_z = self.calculate_errors()

        t = cast_to_np(self.filtration.time_domain)

        quantile_values = [0.5, 0.8, 0.95]
        for value in quantile_values:

            # Violin plot
            self.violin_plot(axs[0], t, error_x, value)
            self.violin_plot(axs[1], t, error_y, value)
            self.violin_plot(axs[2], t, error_z, value)

        for i in range(3):
            axs[i].legend()
            axs[i].grid(True)
        axs[0].set_title(
            f"Quantile of errors along time - Iteration {self.iteration + 1}"
        )
        axs[0].set_ylabel(r"$(\hat X - X)$")
        axs[1].set_ylabel(r"$(\hat Y - Y)$")
        axs[2].set_ylabel(r"$(\hat Z - Z)$")
        axs[2].set_xlabel("Time")
        plt.savefig(
            f"./.figures/error_quantiles_along_time_iteration_{self.iteration+1}.png"
        )
        plt.close()

    def plot_quadratic_error_quantiles_along_time(self):
        if self.analytical_backward_solution is None:
            return
        _, axs = plt.subplots(3, 1, figsize=(12, 16))

        error_x, error_y, error_z = self.calculate_errors()

        quadratic_error_x = error_x**2
        quadratic_error_y = error_y**2
        quadratic_error_z = error_z**2

        t = cast_to_np(self.filtration.time_domain)

        quantile_values = [0.5, 0.9, 0.95]
        for value in quantile_values:

            # squared errors along time
            self.quantiles_along_time(axs[0], t, quadratic_error_x, value)
            self.quantiles_along_time(axs[1], t, quadratic_error_y, value)
            self.quantiles_along_time(axs[2], t, quadratic_error_z, value)

        for i in range(3):
            axs[i].legend()
            axs[i].grid(True)
        axs[0].set_title(
            f"Quantile of quadratic errors along time - Iteration {self.iteration + 1}"
        )
        axs[0].set_ylabel(r"$(\hat X - X)^2$")
        axs[1].set_ylabel(r"$(\hat Y - Y)^2$")
        axs[2].set_ylabel(r"$(\hat Z - Z)^2$")
        axs[2].set_xlabel("Time")
        plt.savefig(
            f"./.figures/quadratic_error_quantiles_iteration_{self.iteration + 1}.png"
        )
        plt.close()

    def plot_error_histogram(self):
        if self.analytical_backward_solution is None:
            return
        _, axs = plt.subplots(3, 1, figsize=(4, 8), layout="constrained")

        error_x, error_y, error_z = self.calculate_errors()
        n_bins = 50
        axs[0].hist(error_x.reshape(-1), bins=n_bins, density=True)
        axs[1].hist(error_y.reshape(-1), bins=n_bins, density=True)
        axs[2].hist(error_z.reshape(-1), bins=n_bins, density=True)

        for i in range(2):
            axs[i].grid(True)
            # axs[i].set_xlim(-1, 1)

        axs[0].set_ylabel(r"$(\hat X - X)$")
        axs[1].set_ylabel(r"$(\hat Y - Y)$")
        axs[2].set_ylabel(r"$(\hat Z - Z)$")
        plt.savefig(f"./.figures/error_histogram_iteration_{self.iteration + 1}.png")
        plt.close()

    def plot_picard_operator_error(self):
        _, axs = plt.subplots(1, 1, figsize=(12, 4))

        y = self.filtration.backward_process
        picard_operator = self.fbsde.backward_sde.calculate_picard_operator()
        quadratic_error = cast_to_np((picard_operator - y) ** 2)[:, :, 0]

        t = cast_to_np(self.filtration.time_domain)
        quantile_values = [0.5, 0.9, 0.95]
        for value in quantile_values:
            self.quantiles_along_time(axs, t, quadratic_error, value)

        axs.legend()
        axs.grid(True)
        axs.set_title(
            f"Quantile of errors against Picard operator value - Iteration {self.iteration + 1}"
        )

        axs.set_xlabel("Time")
        axs.set_ylabel(r"$(\hat Y_t - P(\hat Y)_t)^2$")
        plt.savefig(f"./.figures/picard_operator_error_{self.iteration}.png")
        plt.close()

    def plot_single_path(self):
        _, axs = plt.subplots(2, 1, layout="constrained")
        t = cast_to_np(self.filtration.time_process)[0, :, :]

        if self.analytical_forward_solution is not None:
            x = cast_to_np(self.analytical_forward_solution(self.filtration))[0, :, :]
            axs[0].plot(t, x, color="r", label="Forward Process - Analytical")

        if self.analytical_backward_solution is not None:
            y = cast_to_np(self.analytical_backward_solution(self.filtration))[0, :, :]
            axs[1].plot(t, y, color="r", label="Backward Process - Analytical")

        y_hat = cast_to_np(self.filtration.backward_process)[0, :, :]
        x_hat = cast_to_np(self.filtration.forward_process)[0, :, :]

        axs[0].plot(t, x_hat, "b--", label="Forward Process - Approximation")
        axs[1].plot(t, y_hat, "b--", label="Backward Process - Approximation")

        for i in [0, 1]:
            axs[i].legend()
        path = f"./.figures/single_path_{self.iteration}.png"

        plt.savefig(path)

        plt.close()

    def plot_loss_along_iteration(self):
        loss_history = self.fbsde.backward_sde.y_approximator.loss_history
        _, axs = plt.subplots(layout="constrained")
        iteration = range(len(loss_history))
        axs.set_title(f"Elicitability Loss history for iteration {self.iteration + 1}")
        axs.plot(iteration, loss_history)
        axs.set_yscale("log")
        path = f"./.figures/loss_plot_{self.iteration}"
        plt.savefig(path)
        plt.close()

    def register_single_path(self):
        x_hat = cast_to_np(self.filtration.forward_process)[0, :, :]
        y_hat = cast_to_np(self.filtration.backward_process)[0, :, :]
        z_hat = cast_to_np(self.filtration.backward_volatility)[0, :, :]
        self.path_register["x_hat"].append(x_hat)
        self.path_register["y_hat"].append(y_hat)
        self.path_register["z_hat"].append(z_hat)

    def plot_approximator_paths_along_iterations(self):
        _, axs = plt.subplots(2, 1, layout="constrained")
        t = cast_to_np(self.filtration.time_process)[0, :, :]

        color_range = self.color_map(np.linspace(0, 1, self.iteration + 1))
        for i in range(self.iteration):
            x_hat = self.path_register["x_hat"][i]
            y_hat = self.path_register["y_hat"][i]
            axs[0].plot(
                t,
                x_hat,
                color=color_range[i],
                # label=f"Forward Process - Iteration {i + 1}",
            )
            axs[1].plot(
                t,
                y_hat,
                color=color_range[i],
                # label=f"Backward Process - Iteration {i + 1}",
            )

        if self.analytical_forward_solution is not None:
            x = cast_to_np(self.analytical_forward_solution(self.filtration))[0, :, :]
            axs[0].plot(
                t,
                x,
                color="r",
                linestyle="dashed",
                label="Forward Process - Analytical",
            )

        if self.analytical_backward_solution is not None:
            y = cast_to_np(self.analytical_backward_solution(self.filtration))[0, :, :]
            axs[1].plot(
                t,
                y,
                color="r",
                linestyle="dashed",
                label="Backward Process - Analytical",
            )

        for i in [0, 1]:
            axs[i].legend()
        path = f"./.figures/approximations_along_picard_iterations.png"

        plt.savefig(path)

        plt.close()

    def plot_population_measure_flow(self):
        _, axs = plt.subplots(1, 1, layout="constrained", figsize=(9, 6))

        t = cast_to_np(self.filtration.time_process)[0, :, :].reshape(-1)

        hat_X = cast_to_np(self.filtration.forward_process)
        positive_quantiles = [0.99, 0.95, 0.9, 0.75]
        negative_quantiles = [0.01, 0.05, 0.1, 0.25]
        alphas = [0.05, 0.1, 0.2, 0.3]
        for i in range(4):
            lower_bound = np.quantile(hat_X, negative_quantiles[i], axis=0).reshape(-1)
            upper_bound = np.quantile(hat_X, positive_quantiles[i], axis=0).reshape(-1)
            percentage = positive_quantiles[i] - negative_quantiles[i]
            axs.fill_between(
                t,
                lower_bound,
                upper_bound,
                color="r",
                alpha=alphas[i],
                label=f"{percentage:.0%} of agents",
            )

        blues = mpl.colormaps["Blues"](np.linspace(0.5, 0.75, 2))
        greens = mpl.colormaps["Greens"](np.linspace(0.5, 0.75, 2))
        reds = mpl.colormaps["Reds"](np.linspace(0.5, 0.75, 2))

        path_index = 1
        hat_mean = cast_to_np(self.filtration.forward_mean_field)[path_index, :, :]

        axs.plot(
            t,
            hat_mean,
            color=greens[0],
            linestyle="dashed",
            label="Agents mean - Approximation",
        )

        mean = (
            self.filtration.common_noise_coefficient
            * self.filtration.common_noise[path_index, :, :]
        )
        axs.plot(
            t,
            mean,
            color=greens[1],
            label="Agents mean - Analytical",
        )

        if self.analytical_forward_solution is not None:
            x = cast_to_np(self.analytical_forward_solution(self.filtration))[
                path_index, :, :
            ]
            axs.plot(t, x, color=reds[1], label="Forward Process - Analytical")

        x_hat = cast_to_np(self.filtration.forward_process)[1, :, :]

        axs.plot(
            t,
            x_hat,
            color=reds[0],
            linestyle="dashed",
            label="Forward Process - Approximation",
        )

        if self.analytical_backward_solution is not None:
            y = cast_to_np(self.analytical_backward_solution(self.filtration))[
                path_index, :, :
            ]
            axs.plot(t, y, color=blues[1], label="Backward Process - Analytical")

        y_hat = cast_to_np(self.filtration.backward_process)[path_index, :, :]
        axs.plot(
            t,
            y_hat,
            color=blues[0],
            linestyle="dashed",
            label="Backward Process - Approximation",
        )

        axs.legend()
        axs.grid(True)
        axs.set_xlim([0, 1])
        path = f"{self.output_folder}/population_measure_flow_iteration_{self.iteration + 1}.png"

        plt.savefig(path)

        plt.close()

    def save_errors(self):
        error_x, error_y, _ = self.calculate_errors()
        mean = self.filtration.common_noise_coefficient * self.filtration.common_noise
        error_m = self.filtration.forward_mean_field - mean
        self.errors["error_x"].append(error_x)
        self.errors["error_y"].append(error_y)
        self.errors["error_m"].append(error_m)

    def plot_error_hist_for_iterations(self):
        if self.analytical_backward_solution is None:
            return
        _, axs = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
        axs[0].set_xlabel(r"$(X^i_t - X_t)$")
        axs[1].set_xlabel(r"$(Y^i_t - Y_t)$")
        axs[2].set_xlabel(r"$(m^i_t - m_t)$")

        n_bins = 50

        alphas = [0.3, 0.4, 0.6]
        for i, iteration in enumerate(self.error_plot_iterations):
            axs[0].hist(
                self.errors["error_x"][i].reshape(-1),
                bins=n_bins,
                density=True,
                label=f"Iteration {iteration}",
                color="r",
                alpha=alphas[i],
            )
            axs[1].hist(
                self.errors["error_y"][i].reshape(-1),
                bins=n_bins,
                density=True,
                label=f"Iteration {iteration}",
                color="r",
                alpha=alphas[i],
            )
            axs[2].hist(
                self.errors["error_m"][i].reshape(-1),
                bins=n_bins,
                density=True,
                label=f"Iteration {iteration}",
                color="r",
                alpha=alphas[i],
            )

        axs[0].set_xlim(-0.5, 0.5)
        axs[1].set_xlim(-0.5, 0.5)
        axs[2].set_xlim(-0.5, 0.5)
        for i in range(3):
            axs[i].grid(True)

        axs[2].legend()

        plt.savefig(f"{self.output_folder}/error_histograms.png")
        plt.close()

    def end_of_iteration_callback(self, fbsde, iteration):
        self.iteration = iteration
        self.fbsde = fbsde
        self.plot_error_quantiles_along_time()
        self.plot_error_histogram()
        self.plot_picard_operator_error()
        self.plot_single_path()
        self.register_single_path()
        self.plot_loss_along_iteration()
        self.plot_population_measure_flow()
        if self.iteration in self.error_plot_iterations:
            self.save_errors()

    def end_of_solver_callback(self, fbsde):
        self.plot_approximator_paths_along_iterations()
        self.plot_error_hist_for_iterations()
