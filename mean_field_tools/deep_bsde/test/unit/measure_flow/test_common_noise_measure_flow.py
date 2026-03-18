from mean_field_tools.deep_bsde.measure_flow import CommonNoiseMeasureFlow
from mean_field_tools.deep_bsde.filtration import CommonNoiseFiltration
from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator
from mean_field_tools.deep_bsde.utils import L_2_norm

import torch
import numpy as np


torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
NUMBER_OF_TIMESTEPS = 101
NUMBER_OF_PATHS = 10_000
TIME_DOMAIN = torch.linspace(0, 1, NUMBER_OF_TIMESTEPS)
SPATIAL_DIMENSIONS = 1

FILTRATION = CommonNoiseFiltration(
    spatial_dimensions=SPATIAL_DIMENSIONS,
    time_domain=TIME_DOMAIN,
    number_of_paths=NUMBER_OF_PATHS,
    common_noise_coefficient=0.3,
    seed=0,
)

measure_flow = CommonNoiseMeasureFlow(filtration=FILTRATION)
measure_flow.initialize_approximator(
    training_args={
        "training_strategy_args": {
            "batch_size": 512,
            "number_of_iterations": 100,
            "number_of_batches": 100,
        }
    },
)


def test_set_elicitability_input():
    input = measure_flow._set_elicitability_input()

    assert input.shape == (NUMBER_OF_PATHS, NUMBER_OF_TIMESTEPS, 1 + SPATIAL_DIMENSIONS)


def test_parameterize_time_process():
    "Should evaluate to t"

    FILTRATION.forward_process = FILTRATION.time_process
    conditional_mean = measure_flow.parameterize(FILTRATION)
    t = FILTRATION.time_process

    deviation = conditional_mean - t
    assert L_2_norm(deviation) < 1e-4


def test_parameterize():
    "Should evaluate to rho * common_noise"
    FILTRATION.forward_process = FILTRATION.brownian_process
    conditional_mean = measure_flow.parameterize(FILTRATION)
    rho = FILTRATION.common_noise_coefficient
    common_noise = FILTRATION.common_noise

    deviation = conditional_mean - rho * common_noise
    assert L_2_norm(deviation) < 1


# --- Multi-network tests ---


def _make_2d_filtration():
    """Helper to create a 2D CommonNoiseFiltration for multi-network tests."""
    return CommonNoiseFiltration(
        spatial_dimensions=2,
        time_domain=torch.linspace(0, 1, 11),
        number_of_paths=100,
        common_noise_coefficient=0.3,
        seed=42,
    )


TRAINING_ARGS_FAST = {
    "training_strategy_args": {
        "batch_size": 50,
        "number_of_iterations": 10,
        "number_of_batches": 1,
    }
}


def test_initialize_approximator_backward_compat():
    """initialize_approximator (singular) sets up single-network mode with list wrapper."""
    filtration = _make_2d_filtration()
    mf = CommonNoiseMeasureFlow(filtration=filtration)
    mf.initialize_approximator(training_args=TRAINING_ARGS_FAST)

    assert len(mf.mean_approximators) == 1
    assert mf._use_single_network is True
    assert mf.mean_approximators[0].output_dimension == 2

    # Verify parameterize produces correct shape
    filtration.forward_process = torch.randn(100, 11, 2)
    result = mf.parameterize(filtration)
    assert result.shape == (100, 11, 2)


def test_initialize_approximators_multi_network():
    """initialize_approximators (plural) creates separate networks."""
    filtration = _make_2d_filtration()
    mf = CommonNoiseMeasureFlow(filtration=filtration)
    mf.initialize_approximators(
        output_dimensions=[1, 1],
        training_args_list=[TRAINING_ARGS_FAST, TRAINING_ARGS_FAST],
    )

    assert len(mf.mean_approximators) == 2
    assert mf._use_single_network is False
    assert mf.mean_approximators[0].output_dimension == 1
    assert mf.mean_approximators[1].output_dimension == 1

    # Verify parameterize produces correct shape
    filtration.forward_process = torch.randn(100, 11, 2)
    result = mf.parameterize(filtration)
    assert result.shape == (100, 11, 2)


def test_multi_network_with_list_of_paths():
    """_elicit_mean_as_function_of_common_noise accepts a list of target tensors."""
    filtration = _make_2d_filtration()
    mf = CommonNoiseMeasureFlow(filtration=filtration)
    mf.initialize_approximators(
        output_dimensions=[1, 1],
        training_args_list=[TRAINING_ARGS_FAST, TRAINING_ARGS_FAST],
    )
    mf.elicitability_input = mf._set_elicitability_input()

    target1 = torch.randn(100, 11, 1)
    target2 = torch.randn(100, 11, 1)
    mf._elicit_mean_as_function_of_common_noise([target1, target2])

    out1 = mf.mean_approximators[0].detached_call(mf.elicitability_input)
    out2 = mf.mean_approximators[1].detached_call(mf.elicitability_input)
    assert out1.shape == (100, 11, 1)
    assert out2.shape == (100, 11, 1)

    combined = torch.cat([out1, out2], dim=2)
    assert combined.shape == (100, 11, 2)


def test_multi_network_custom_nn_args():
    """Each network can have different architecture via nn_args_list."""
    filtration = _make_2d_filtration()
    mf = CommonNoiseMeasureFlow(filtration=filtration)
    mf.initialize_approximators(
        nn_args_list=[{"number_of_nodes": 16}, {"number_of_nodes": 64}],
        output_dimensions=[1, 1],
    )

    assert mf.mean_approximators[0].input.out_features == 16
    assert mf.mean_approximators[1].input.out_features == 64


def test_multi_network_custom_training_args():
    """Each network can have different training args."""
    args1 = {
        "training_strategy_args": {
            "batch_size": 32,
            "number_of_iterations": 50,
            "number_of_batches": 1,
        }
    }
    args2 = {
        "training_strategy_args": {
            "batch_size": 64,
            "number_of_iterations": 100,
            "number_of_batches": 1,
        }
    }
    filtration = _make_2d_filtration()
    mf = CommonNoiseMeasureFlow(filtration=filtration)
    mf.initialize_approximators(
        output_dimensions=[1, 1],
        training_args_list=[args1, args2],
    )

    assert mf.training_args_list[0] == args1
    assert mf.training_args_list[1] == args2


def test_multi_network_prebuilt_approximators():
    """Pre-built FunctionApproximator instances can be passed directly."""
    filtration = _make_2d_filtration()
    domain_dim = 1 + filtration.spatial_dimensions

    approx1 = FunctionApproximator(domain_dimension=domain_dim, output_dimension=1)
    approx2 = FunctionApproximator(domain_dimension=domain_dim, output_dimension=1)

    mf = CommonNoiseMeasureFlow(filtration=filtration)
    mf.initialize_approximators(
        approximators=[approx1, approx2],
        output_dimensions=[1, 1],
    )

    assert mf.mean_approximators[0] is approx1
    assert mf.mean_approximators[1] is approx2
