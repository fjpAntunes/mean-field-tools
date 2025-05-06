from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator
import torch

neural_net = FunctionApproximator(domain_dimension=1, output_dimension=1)

x = torch.linspace(-5, 5, 101).reshape(1, 101, 1)
x_2 = x**2


neural_net.minimize_over_sample(
    sample=x,
    target=x_2,
    training_strategy_args={
        "batch_size": 101,
        "number_of_iterations": 1000,
        "number_of_batches": 1000,
    },
)

x_2_hat = neural_net.detached_call(x)

error = x_2 - x_2_hat

print(f"f fit: mean: {error.mean()}, var: {error.var()}")

grad = neural_net.grad(x)

f_prime = 2 * x

f_prime_error = f_prime - grad

print(f"f prime fit:  mean: {f_prime_error.mean()}, var: {f_prime_error.var()}  ")

y = neural_net(x)

y.backward(torch.ones(y.shape))

f_prime_2 = x.grad
f_prime_error_2 = f_prime - f_prime_2

print(f"f prime 2 fit: {f_prime_error_2.mean()}, var: {f_prime_error_2.var()}")

import matplotlib.pyplot as plt
from mean_field_tools.deep_bsde.artist import cast_to_np

fig, axs = plt.subplots(3, 1)

x = cast_to_np(x)
x_2_hat = cast_to_np(x_2_hat)
grad = cast_to_np(grad)
f_prime = cast_to_np(f_prime)


axs[0].scatter(x, x_2_hat)
axs[1].scatter(x, grad)
axs[1].scatter(x, f_prime, color="red")

axs[2].scatter(x, f_prime, color="red")
axs[2].scatter(x, cast_to_np(f_prime_2))


fig.show()
