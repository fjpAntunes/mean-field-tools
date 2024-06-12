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

    def preprocess(self, input):
        if input.device.type != self.device:
            return input.to(self.device)
        else:
            return input
        
    def postprocess(self, output, training_status):
        if training_status == True:
            return output
        else:
            return output.to('cpu')

    def forward(self, x):
        x = self.preprocess(x)
        out = self.activation(self.input(x))
        for layer in self.hidden:
            out = self.activation(layer(out))

        out = self.output(out)
        out = self.postprocess(out, training_status = self.is_training)
        return out

    def minimize_over_sample(
        self,
        sample,  #  (sample_size, path_length, time_dimension + spatial_dimensions)
        target,  # shape :  (output_dimension, sample_size)
        scoring=lambda x, y: (x - y) ** 2,  # Function to be minimized over sample
        batch_size=512,
        number_of_iterations=10_000,
        number_of_epochs=100,
        number_of_plots=10,
        plotting=False,
        save_figures=False,
    ):
        self.is_training = True
        self.save_figures = save_figures
        self.optimizer = self.sgd_parameters["optimizer"](
            self.parameters(), **self.sgd_parameters["optimizer_params"]
        )
        self.scheduler = self.sgd_parameters["scheduler"](
            self.optimizer, **self.sgd_parameters["scheduler_params"]
        )
        sample_size = sample.shape[0]
        self.loss_history = []
        self.loss_recent_history = []

        for j in range(1, number_of_epochs + 1):
            print(f"Epoch {j}")
            batch_index = torch.randperm(sample_size)[:batch_size]
            batch_sample = sample[batch_index, :, :].to(self.device)
            batch_target = target[batch_index, :, :].to(self.device)
            for i in tqdm(range(number_of_iterations // number_of_epochs)):
                estimated = self(batch_sample)

                empirical_loss = torch.mean(scoring(estimated, batch_target))
                self.optimizer.zero_grad()
                empirical_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self._append_loss_moving_average(empirical_loss.item(), window_size=100)

            if plotting and np.mod(j, number_of_epochs // number_of_plots) == 0:
                self.plot_loss_history(j)
                self.plot_terminal_fit(batch_sample, batch_target, j)
                self.plot_sample_paths(batch_sample, j)
        
        self.is_training = False

    def _append_loss_moving_average(self, loss, window_size):
        self.loss_recent_history.append(loss)
        if len(self.loss_recent_history) == window_size:
            self.loss_history.append(np.mean(self.loss_recent_history))
            self.loss_recent_history.pop(0)

    def plot_loss_history(self, number):
        fig, axs = plt.subplots()
        iteration = range(len(self.loss_history))
        axs.plot(iteration, self.loss_history)
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

    def plot_terminal_fit(self, sample, target, number):
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        _, time_length, _ = sample.shape
        for i, time_index in enumerate([0, time_length // 2, time_length - 1]):
            x = sample[:, time_index, 1]
            T = sample[0, -1, 0].detach().cpu().numpy()
            t = T * (time_index / time_length)
            y_hat = self(sample)[:, time_index, 0]

            x = x.reshape(-1).detach().cpu().numpy()
            y_hat = y_hat.reshape(-1).detach().cpu().numpy()
            y = x**2 + (T - t)

            axs[i].set_ylim(0, 2)
            axs[i].set_xlim(-1.5, 1.5)
            axs[i].scatter(x, y, color="r", s=0.5)
            axs[i].scatter(x, y_hat, color="b", s=0.5)
        if self.save_figures:
            plt.savefig(f"./.figures/fit_plot_{number}.png")
        else:
            plt.plot()
            plt.show()
        plt.close()
