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

    def plot_loss_history(self, number, loss_history):
        fig, axs = plt.subplots()
        iteration = range(len(loss_history))
        axs.plot(iteration, loss_history)
        axs.set_yscale("log")
        if self.save_figures:
            plt.savefig(f"./.figures/loss_plot_{number}")
        else:
            plt.plot()
            plt.show()
        plt.close()

    def plot_sample_paths(self, sample, number):
        fig, axs = plt.subplots()
        t = sample[0, :, 0].detach().cpu().numpy()
        for i in range(sample.shape[0]):
            x = sample[i, :, 1].detach().cpu().numpy()
            axs.plot(t, x)
        if self.save_figures:
            plt.savefig(f"./.figures/sample_path_{number}.png")
        else:
            plt.plot()
            plt.show()
        plt.close()

    def plot_fit_against_analytical(self, approximator, sample, number):
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        _, time_length, _ = sample.shape
        for i, time_index in enumerate([0, time_length // 2, time_length - 1]):
            x = sample[:, time_index, 1]
            T = sample[0, -1, 0].detach().cpu().numpy()
            t = T * (time_index / time_length)
            y_hat = approximator(sample)[:, time_index, 0]

            x = x.reshape(-1).detach().cpu().numpy()
            y_hat = y_hat.reshape(-1).detach().cpu().numpy()

            axs[i].set_ylim(0, 2)
            axs[i].set_xlim(-1.5, 1.5)
            axs[i].scatter(x, y_hat, color="b", s=0.5, label="Approximation")
            if self.analytical_solution:
                y = self.analytical_solution(x, t, T)  # x**2 + (T - t)
                axs[i].scatter(x, y, color="r", s=0.5, label="Analytical")
            axs[i].legend()

        if self.save_figures:
            plt.savefig(f"./.figures/fit_plot_{number}.png")
        else:
            plt.plot()
            plt.show()
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
        x = self.preprocess(x)
        out = self.activation(self.input(x))
        for layer in self.hidden:
            out = self.activation(layer(out))

        out = self.output(out)
        out = self.postprocess(out, training_status=self.is_training)
        return out

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

    def single_training_step(
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

    def minimize_over_sample(
        self,
        sample,  #  (sample_size, path_length, time_dimension + spatial_dimensions)
        target,  # shape :  (output_dimension, sample_size)
        batch_size=512,
        number_of_iterations=10_000,
        number_of_epochs=100,
        number_of_plots=10,
        plotter: FunctionApproximatorArtist = None,
    ):
        self.training_setup()

        for j in range(1, number_of_epochs + 1):
            print(f"Epoch {j}")
            batch_sample, batch_target = self._generate_batch(
                batch_size, sample, target
            )

            for i in tqdm(range(number_of_iterations // number_of_epochs)):
                self.single_training_step(batch_sample, batch_target)

            if plotter and np.mod(j, number_of_epochs // number_of_plots) == 0:
                plotter.plot_loss_history(j, self.loss_history)
                plotter.plot_fit_against_analytical(self, batch_sample, j)
                plotter.plot_sample_paths(batch_sample, j)

        self.is_training = False

    def _append_loss_moving_average(self, loss, window_size):
        self.loss_recent_history.append(loss)
        if len(self.loss_recent_history) == window_size:
            self.loss_history.append(np.mean(self.loss_recent_history))
            self.loss_recent_history.pop(0)
