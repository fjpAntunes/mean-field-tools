from mean_field_tools.deep_bsde.filtration import CommonNoiseFiltration
from mean_field_tools.deep_bsde.forward_backward_sde import CommonNoiseBackwardSDE
from mean_field_tools.deep_bsde.utils import IDENTITY_TERMINAL
import torch

torch.manual_seed(0)

TIME_DOMAIN = torch.linspace(0, 1, 11)
RHO = 0.3
NUMBER_OF_PATHS = 50

FILTRATION = CommonNoiseFiltration(
    spatial_dimensions=1,
    time_domain=TIME_DOMAIN,
    number_of_paths=NUMBER_OF_PATHS,
    common_noise_coefficient=RHO,
    seed=0,
)

bsde = CommonNoiseBackwardSDE(
    terminal_condition_function=IDENTITY_TERMINAL,
    filtration=FILTRATION,
)

bsde.initialize_approximator()
bsde.initialize_z_approximator()

FILTRATION.forward_process = FILTRATION.brownian_process

Z_VAL = 2.0
Z_ZERO_VAL = 0.5


class ConstantApproximator:
    """Mock approximator that returns a constant value everywhere."""

    def __init__(self, value, shape_like):
        self.value = value
        self.shape_like = shape_like

    def detached_call(self, input):
        return self.value * torch.ones(
            input.shape[0], self.shape_like.shape[1], 1
        )


bsde.z_approximator = ConstantApproximator(Z_VAL, FILTRATION.brownian_process)
bsde.z_zero_approximator = ConstantApproximator(
    Z_ZERO_VAL, FILTRATION.brownian_process
)


def test_volatility_integral_shape():
    vol_integral = bsde.calculate_volatility_integral()
    num_timesteps = len(TIME_DOMAIN)
    assert vol_integral.shape == (NUMBER_OF_PATHS, num_timesteps, 1)


def test_volatility_integral_terminal_is_zero():
    vol_integral = bsde.calculate_volatility_integral()
    terminal = vol_integral[:, -1, :]
    assert torch.allclose(terminal, torch.zeros_like(terminal))


def test_volatility_integral_values():
    """With constant Z and Z^0, the backward volatility integral at index k
    (using the total - cumsum pattern from the parent class) is:
    Z * (W_T - W_{t_{k+1}}) + Z^0 * (W^0_T - W^0_{t_{k+1}})
    """
    vol_integral = bsde.calculate_volatility_integral()

    idiosyncratic = FILTRATION.idiosyncratic_noise
    common = FILTRATION.common_noise

    idiosyncratic_terminal = idiosyncratic[:, -1:, :]
    common_terminal = common[:, -1:, :]

    # At position k, the integral runs from t_{k+1} to T
    expected_inner = Z_VAL * (
        idiosyncratic_terminal - idiosyncratic[:, 1:, :]
    ) + Z_ZERO_VAL * (common_terminal - common[:, 1:, :])
    terminal = torch.zeros(NUMBER_OF_PATHS, 1, 1)
    expected = torch.cat([expected_inner, terminal], dim=1)

    assert torch.allclose(vol_integral, expected, atol=1e-5)
