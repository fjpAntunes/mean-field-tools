r"""Tests quadratic with drift
Equation:
$$
dY_t =-2t\, dt Z_t\, dW_t, \quad Y_T = W^2_T, \\
$$
Where $W_t$ is the standard brownian motion.

"""

from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, BackwardSDE
import torch
from mean_field_tools.deep_bsde.utils import tensors_are_close


def test_terminal_quadratic_with_deterministic_drift():
    """
    Tests BSDE solver for the equation
    """
    TIME_DOMAIN = torch.linspace(0, 1, 101)
    NUMBER_OF_PATHS = 100
    SPATIAL_DIMENSIONS = 1

    def TERMINAL_CONDITION(filtration: Filtration):
        B_T = filtration.brownian_process[:, -1, :]
        return B_T**2

    def DRIFT(filtration: Filtration):
        t = filtration.time_process
        return 2 * t

    def ANALYTICAL_SOLUTION(x, t, T):
        return x**2 + (T - t) + (T**2 - t**2)

    filtration = Filtration(SPATIAL_DIMENSIONS, TIME_DOMAIN, NUMBER_OF_PATHS, seed=0)

    bsde = BackwardSDE(
        terminal_condition_function=TERMINAL_CONDITION,
        drift=DRIFT,
        filtration=filtration,
    )

    bsde.initialize_approximator()

    bsde.solve(
        approximator_args={
            "training_strategy_args": {
                "batch_size": 100,
                "number_of_iterations": 500,
                "number_of_batches": 500,
            },
        }
    )

    forward_path = bsde.filtration.get_paths()[:1, :, :]

    output = bsde.y_approximator(forward_path).tolist()
    forward_path = bsde.filtration.get_paths()[:1, :, :]

    output = bsde.y_approximator(forward_path)

    
    benchmark = ANALYTICAL_SOLUTION(forward_path, filtration.time_process, 1)

    error = torch.norm(output - benchmark)
    # Error is lenient because this is a fast run on cpu
    assert error < 20
