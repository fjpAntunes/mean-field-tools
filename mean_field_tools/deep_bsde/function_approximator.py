import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

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
        approximator: nn.Module,
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
        approximator: nn.Module,
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


class FunctionApproximator(nn.Module):
    def __init__(
        self,
        domain_dimension,
        output_dimension,
        number_of_layers=5,
        number_of_nodes=36,
        scoring=lambda x, y: (x - y) ** 2,  # Function to be minimized over sample
        optimizer=optim.Adam,
        optimizer_params={"lr": 0.005},
        scheduler=optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 5, "gamma": 0.9997},
        device="cpu",
    ):

        super(FunctionApproximator, self).__init__()
        self.domain_dimension = domain_dimension
        self.output_dimension = output_dimension
        self.sgd_parameters = {
            "optimizer": optimizer,
            "optimizer_params": optimizer_params,
            "scheduler": scheduler,
            "scheduler_params": scheduler_params,
        }

        self.device = device
        self.is_training = False

        self.input = nn.Linear(domain_dimension, number_of_nodes).to(self.device)
        self.hidden = nn.ModuleList(
            [
                nn.Linear(number_of_nodes, number_of_nodes).to(self.device)
                for _ in range(number_of_layers - 1)
            ]
        )
        self.output = nn.Linear(number_of_nodes, output_dimension).to(self.device)

        self.activation = nn.SiLU()

        self.scoring = scoring

    def preprocess(self, input):
        if input.device.type != self.device:
            return input.to(self.device)
        else:
            return input

    def postprocess(self, output, training_status):
        if training_status == True:
            return output
        else:
            return output.to("cpu")

    def forward(self, x):
        self.x = self.preprocess(x)
        out = self.activation(self.input(self.x))
        for layer in self.hidden:
            out = self.activation(layer(out))

        out = self.output(out)
        out = self.postprocess(out, training_status=self.is_training)
        return out
    
    def grad(self,x):
        y = self(x)
        return torch.autograd.grad(y,x)

    def detached_call(self, x):
        return self.forward(x).detach()

    def _generate_batch(
        self,
        batch_size: int,
        sample: torch.Tensor,
        target: torch.Tensor,
        seed: int = None,
    ):
        """Implements random sampling with replacement over paths of tensor.

        Args:
            batch_size (int): size of batch used in each training epoch.
            sample (torch.Tensor): sample tensor.
            target (torch.Tensor): optimization target tensor.
            seed (int): (Optional) random seed.

        Returns:
            batch_sample (torch.Tensor): sampled paths.
            batch_target (torch.Tensor): optimization targets for the sampled paths.
        """
        if seed is not None:
            torch.manual_seed(seed)

        sample_size = sample.shape[0]

        batch_index = torch.randperm(sample_size)[:batch_size]
        batch_sample = sample[batch_index, :, :].to(self.device)
        batch_target = target[batch_index, :, :].to(self.device)

        return batch_sample, batch_target

    def training_setup(self):
        """Pre-training object state configuration."""
        self.is_training = True
        self.optimizer = self.sgd_parameters["optimizer"](
            self.parameters(), **self.sgd_parameters["optimizer_params"]
        )
        self.scheduler = self.sgd_parameters["scheduler"](
            self.optimizer, **self.sgd_parameters["scheduler_params"]
        )
        self.loss_history = []
        self.loss_recent_history = []
        return None

    def single_gradient_descent_step(
        self, batch_sample: torch.Tensor, batch_target: torch.Tensor
    ):
        """Performs a single step of gradient descent.

        Args:
            batch_sample (torch.Tensor): Sampled paths.
            batch_target (torch.Tensor): Optimization targets for the sampled paths.

        Returns:
            None.
        """

        estimated = self(batch_sample)
        empirical_loss = torch.mean(self.scoring(estimated, batch_target))
        self.optimizer.zero_grad()
        empirical_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self._append_loss_moving_average(empirical_loss.item(), window_size=100)

        return None

    def _apply_training_strategy(
        self, training_strategy: Callable, training_strategy_args: dict = {}
    ):
        """Implements simplified strategy design pattern in order to enable different NN training methodologies.

        Args:
            training_strategy (Callable): Training method for the NN.
            training_strategy_args (dict, optional): Arguments for the training. Defaults to {}.
        """

        training_strategy(self, **training_strategy_args)

    def _batch_sgd_training(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        batch_size: int = 512,
        number_of_batches: int = 100,
        number_of_iterations: int = 10_000,
        plotter: FunctionApproximatorArtist = None,
        number_of_plots: int = 1,
    ):
        """Performs gradient descent on random batch of paths sampled with replacement.

        Steps:
        1. Samples a batch of paths,
        2. performs gradient descent on the empirical loss function over the batch
        3. Repeats 1-2 until `number_of_iterations` is reached.

        Args:
            input (torch.Tensor): Input dataset for the neural network
            target (torch.Tensor): Target output dataset for the neural network
            batch_size (int, optional):  Number of paths sampled with replacement from `input`. Defaults to 512.
            number_of_batches (int, optional):  Defaults to 100.
            number_of_iterations (int, optional): Total number of gradient descent steps taken by the algorithm. Defaults to 10_000.
            plotter (FunctionApproximatorArtist, optional): Auxiliary plotting object. Defaults to None.
            number_of_plots (int, optional): number of plots to display during training, at the end of the iterations over a batch. Defaults to 1.
        """
        for j in range(1, number_of_batches + 1):
            print(f"Batch {j}")
            batch_sample, batch_target = self._generate_batch(batch_size, input, target)

            for i in tqdm(range(number_of_iterations // number_of_batches)):
                self.single_gradient_descent_step(batch_sample, batch_target)

            if plotter and np.mod(j, number_of_batches // number_of_plots) == 0:
                plotter.plot_loss_history(j, self.loss_history)
                plotter.plot_fit_against_analytical(self, batch_sample, j)
                plotter.plot_paths(
                    approximator=self,
                    sample=batch_sample,
                    number_of_paths=20,
                    iteration=j,
                )
                plotter.plot_single_path(
                    approximator=self, sample=batch_sample, iteration=j
                )

    def minimize_over_sample(
        self,
        sample,  #  (sample_size, path_length, time_dimension + spatial_dimensions)
        target,  # shape :  (output_dimension, sample_size)
        training_strategy: Callable = None,
        training_strategy_args: dict = {
            "batch_size": 100,
            "number_of_iterations": 500,
            "number_of_batches": 5,
            "number_of_plots": 5,
        },
    ):
        if training_strategy is None:
            training_strategy = self._batch_sgd_training

        training_strategy_args["input"] = sample
        training_strategy_args["target"] = target

        self.training_setup()

        training_strategy(**training_strategy_args)

        self.is_training = False

    def _append_loss_moving_average(self, loss, window_size):
        self.loss_recent_history.append(loss)
        if len(self.loss_recent_history) == window_size:
            self.loss_history.append(np.mean(self.loss_recent_history))
            self.loss_recent_history.pop(0)
