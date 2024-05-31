import torch
from deep_bsde.function_approximator import FunctionApproximator

# Test different input shapes

# Test different output shapes

# Test different scoring functions


# Test simple fit
def test_different_input_shapes():
    sampler = lambda batch_size: torch.linspace(-1, 1, 101).reshape(-1, 1)
    target = (sampler(1)) ** 2

    approximator = FunctionApproximator(
        domain_dimension=1, output_dimension=1, number_of_layers=5
    )
    approximator.minimize_over_sample(sampler, target, number_of_iterations=100)
    x = sampler(1)
    torch.testing.assert_close(
        expected=torch.zeros_like(target),
        actual=approximator(x) - target,
        atol=2e-2,
        rtol=0.1,
    )
