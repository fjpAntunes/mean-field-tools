import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class FunctionApproximator(nn.Module):

    def __init__(
        self,
        domain_dimension,
        output_dimension,
        number_of_layers=5,
        number_of_nodes=36,
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

        self.input = nn.Linear(domain_dimension, number_of_nodes).to(self.device)
        self.hidden = nn.ModuleList(
            [
                nn.Linear(number_of_nodes, number_of_nodes).to(self.device)
                for _ in range(number_of_layers - 1)
            ]
        )
        self.output = nn.Linear(number_of_nodes, output_dimension).to(self.device)

        self.activation = nn.SiLU()

    def forward(self, x):
        out = self.activation(self.input(x))
        for layer in self.hidden:
            out = self.activation(layer(out))

        out = self.output(out)

        return out

    def minimize_over_sample(
        self,
        sample,  # Callable - receives int n and generates n samples with shape : (sample_size, domain_dimension)
        target,  # shape :  (output_dimension, sample_size)
        scoring=lambda x, y: (x - y) ** 2,  # Function to be minimized over sample
        batch_size=512,
        number_of_iterations=10_000,
        steps_between_plots=1_000,
        plotting=False,
        save_figures=False,
    ):
        self.save_figures = save_figures
        self.optimizer = self.sgd_parameters["optimizer"](
            self.parameters(), **self.sgd_parameters["optimizer_params"]
        )
        self.scheduler = self.sgd_parameters["scheduler"](
            self.optimizer, **self.sgd_parameters["scheduler_params"]
        )
        sample_size = sample.shape[0]
        for i in tqdm(range(number_of_iterations)):
            batch_index = torch.randperm(sample_size)[:batch_size]
            batch_sample = sample[batch_index, :, :]
            batch_target = target[batch_index, :, :]
            self.loss_history = []
            estimated = self(batch_sample)

            empirical_loss = torch.mean(scoring(estimated, batch_target))
            self.optimizer.zero_grad()
            empirical_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.loss_history.append(empirical_loss.item())

            if plotting and np.mod(i, steps_between_plots) == 0:
                self.plot_loss_history()
                self.plot_terminal_fit(batch_sample, batch_target, i)
                self.plot_sample_paths(batch_sample, i)

    def plot_loss_history(self):
        pass

    def plot_sample_paths(self, sample, number):
        fig, axs = plt.subplots()
        t = sample[0, :, 0]
        for i in range(sample.shape[0]):
            x = sample[i, :, 1]
            axs.plot(t, x)
        if self.save_figures:
            plt.savefig(f"./.figures/sample_path_{number}.png")

    def plot_terminal_fit(self, sample, target, number):
        x = sample[:, -1, 1]
        y_hat = self(sample)[:, -1, 0]

        x = x.reshape(-1).detach().numpy()
        y_hat = y_hat.reshape(-1).detach().numpy()
        y = target[:, -1].reshape(-1).detach().numpy()

        fig, axs = plt.subplots()
        axs.set_ylim(0, 2)
        axs.set_xlim(-1.5, 1.5)
        axs.scatter(x, y, color="r", s=0.5)
        axs.scatter(x, y_hat, color="b", s=0.5)
        if self.save_figures:
            plt.savefig(f"./.figures/fit_plot_{number}.png")
        else:
            plt.plot()