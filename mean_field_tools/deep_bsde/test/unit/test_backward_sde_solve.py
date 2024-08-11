from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, BackwardSDE
from mean_field_tools.deep_bsde.utils import QUADRATIC_TERMINAL
import torch


NUMBER_OF_TIMESTEPS = 101
TIME_DOMAIN = torch.linspace(0, 1, NUMBER_OF_TIMESTEPS)
NUMBER_OF_PATHS = 100
SPATIAL_DIMENSIONS = 1


filtration = Filtration(SPATIAL_DIMENSIONS, TIME_DOMAIN, NUMBER_OF_PATHS, seed=0)

bsde = BackwardSDE(
    terminal_condition_function=QUADRATIC_TERMINAL,
    filtration=filtration,
)

bsde.initialize_approximator()

bsde.solve(
    approximator_args={
        "training_strategy_args": {
            "batch_size": 100,
            "number_of_iterations": 500,
            "number_of_batches": 5,
            "number_of_plots": 5,
        },
    }
)




def test_backward_process_shape():
    backward_process = bsde.generate_backward_process()
    assert backward_process.shape == (
        NUMBER_OF_PATHS,
        NUMBER_OF_TIMESTEPS,
        SPATIAL_DIMENSIONS,
    )

def test_backward_volatility_shape():
    backward_volatility = bsde.generate_backward_volatility()

    assert backward_volatility.shape == (
        NUMBER_OF_PATHS,
        NUMBER_OF_TIMESTEPS-1,
        SPATIAL_DIMENSIONS
    )

def test_backward_volatility_value():
    backward_process = bsde.generate_backward_process()
    backward_volatility = bsde.generate_backward_volatility()
    drift = bsde.drift_path[0,0,0]
    dt = bsde.filtration.dt
    dw = bsde.filtration.brownian_increments[0,0,0]
    
    dy = backward_process[0,1,0] - backward_process[0,0,0]
    
    benchmark = (dy - drift*dt)/ dw
    assert benchmark.tolist() == backward_volatility[0,0,0].tolist()