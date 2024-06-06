from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, BackwardSDE
import torch

TIME_DOMAIN = torch.linspace(0,1,101)
NUMBER_OF_PATHS = 1000
SPATIAL_DIMENSIONS = 1

TERMINAL_CONDITION = lambda x: x**2

filtration = Filtration(SPATIAL_DIMENSIONS,TIME_DOMAIN,NUMBER_OF_PATHS)
filtration.generate_paths()

bsde = BackwardSDE(
    spatial_dimensions=SPATIAL_DIMENSIONS,
    time_domain=TIME_DOMAIN,
    terminal_condition_function= TERMINAL_CONDITION,
    filtration=filtration
)

bsde.initialize_approximator()

bsde.solve(
    approximator_args={
        'batch_size':100,
        'number_of_iterations': 1000,
        'steps_between_plots' : 100,
        'plotting' : True,
        'save_figures' : True
    }
)

import pdb; pdb.set_trace()