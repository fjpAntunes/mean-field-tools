from mean_field_tools.deep_bsde.filtration import CommonNoiseFiltration
from mean_field_tools.deep_bsde.forward_backward_sde import CommonNoiseBackwardSDE
from mean_field_tools.deep_bsde.utils import IDENTITY_TERMINAL
import torch

torch.manual_seed(0)

TIME_DOMAIN = torch.linspace(0, 1, 11)
NUMBER_OF_TIMESTEPS = len(TIME_DOMAIN)
NUMBER_OF_PATHS = 50
RHO = 0.3
SPATIAL_DIMENSIONS = 1
NUMBER_OF_PARAMETERS = 2


def setup(number_of_parameters: int = 0):
    filtration = CommonNoiseFiltration(
        spatial_dimensions=SPATIAL_DIMENSIONS,
        time_domain=TIME_DOMAIN,
        number_of_paths=NUMBER_OF_PATHS,
        common_noise_coefficient=RHO,
        seed=0,
    )
    if number_of_parameters:
        filtration.set_parameter(
            torch.ones(NUMBER_OF_PATHS, NUMBER_OF_TIMESTEPS, number_of_parameters)
        )

    filtration.forward_process = filtration.brownian_process

    bsde = CommonNoiseBackwardSDE(
        terminal_condition_function=IDENTITY_TERMINAL,
        filtration=filtration,
    )
    bsde.initialize_z_approximator()

    return bsde


def test_z_input_width():
    """Input is (t, X_t, W^0_t): there is no initial condition channel."""
    bsde = setup()

    assert bsde.set_z_input().shape == (
        NUMBER_OF_PATHS,
        NUMBER_OF_TIMESTEPS,
        1 + 2 * SPATIAL_DIMENSIONS,
    )


def test_z_input_width_with_parameter():
    """A parameter is prepended to (t, X_t, W^0_t)."""
    bsde = setup(number_of_parameters=NUMBER_OF_PARAMETERS)

    assert bsde.set_z_input().shape == (
        NUMBER_OF_PATHS,
        NUMBER_OF_TIMESTEPS,
        NUMBER_OF_PARAMETERS + 1 + 2 * SPATIAL_DIMENSIONS,
    )


def test_z_approximators_accept_z_input():
    """Both volatility approximators must be sized for what `set_z_input` builds."""
    for number_of_parameters in [0, NUMBER_OF_PARAMETERS]:
        bsde = setup(number_of_parameters)
        input = bsde.set_z_input()

        for volatility in [
            bsde.generate_idiosyncratic_noise_volatility(),
            bsde.generate_common_noise_volatility(),
        ]:
            assert volatility.shape == (
                NUMBER_OF_PATHS,
                NUMBER_OF_TIMESTEPS,
                bsde.number_of_dimensions,
            )

        assert input.shape[-1] == bsde.z_approximator.gru.input_size
        assert input.shape[-1] == bsde.z_zero_approximator.gru.input_size
