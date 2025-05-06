import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Callable
import numpy as np


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
        self.has_trained = False

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
        out = self.postprocess(out, training_status=self.has_trained)
        return out

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates approximate gradient for the approximate function

        Uses `torch.autograd.grad` to calculate gradients for each point.
        We consider that the last dimension of x defines a point, and other
        dimensions are looped over.

        `torch.autograd.grad` actually calculates the vector product between
        the jacobian and an auxiliary vector. We define this auxiliary vector
        as a vector of ones in order to return the original gradient.
        Args:
            x (torch.Tensor): input tensor. Should be of shape (num_paths, path_length, input_dimensions)

        Returns:
            torch.Tensor: Gradients for each point of each path. Should be of shape (num_paths, path_length, input_dimensions);
        """
        # import pdb

        # pdb.set_trace()
        x.requires_grad = True

        y = self(x)
        if y.shape != (1,):
            y = y.squeeze(-1)
        aux_tensor = torch.ones(y.shape)
        gradient = torch.autograd.grad(y, x, aux_tensor)[0]
        return gradient

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
        self.has_trained = True
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
        self._append_loss_moving_average(empirical_loss.item(), window_size=1)

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
        plotter=None,
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
        for j in tqdm(range(1, number_of_batches + 1)):
            batch_sample, batch_target = self._generate_batch(batch_size, input, target)

            for i in range(number_of_iterations // number_of_batches):
                self.single_gradient_descent_step(batch_sample, batch_target)

            if plotter and np.mod(j, number_of_batches // number_of_plots) == 0:
                plotter.make_training_plots(
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

        if not self.has_trained:
            self.training_setup()

        training_strategy(**training_strategy_args)

        # self.has_trained = False

    def _append_loss_moving_average(self, loss, window_size):
        self.loss_recent_history.append(loss)
        if len(self.loss_recent_history) == window_size:
            self.loss_history.append(np.mean(self.loss_recent_history))
            self.loss_recent_history.pop(0)


class OperatorApproximator(FunctionApproximator):
    def __init__(self):
        super(OperatorApproximator, self).__init__(
            domain_dimension=1, output_dimension=1
        )
        self.hidden_size = 3
        self.num_layers = 1
        self.input_size = 1
        self.gru = nn.GRU(
            self.input_size, self.hidden_size, self.num_layers, batch_first=True
        )

        self.output = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)

        out = self.output(out)
        out = out
        return out
