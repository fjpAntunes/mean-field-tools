from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator
from mean_field_tools.deep_bsde.utils import tensors_are_close
import torch


def setup():
    function_approximator = FunctionApproximator(
        domain_dimension=1, output_dimension=1, device="cpu"
    )
    return function_approximator


class MockDevice:
    def __init__(self, type):
        self.type = type


class MockTensor:
    def __init__(self, device: MockDevice):
        self.device = device

    def to(self, device_type):
        self.device.type = device_type
        return self


def test_preprocess():
    """Preprocess should make input tensor be on same device as function_approximator."""
    function_approximator = setup()
    mock_device = MockDevice(type="cuda")
    mock_input = MockTensor(device=mock_device)
    processed_input = function_approximator.preprocess(mock_input)

    assert processed_input.device.type == function_approximator.device


def test_postprocess():
    """If function_approximator is not training, should return output on cpu device for interactivity."""
    function_approximator = setup()
    mock_device = MockDevice(type="cuda")
    mock_output = MockTensor(device=mock_device)
    processed_output = function_approximator.postprocess(
        mock_output, training_status=False
    )

    assert processed_output.device.type == "cpu"


def test_generate_sample_batch():
    approximator = setup()
    sample = torch.Tensor(
        [
            [[0, 0], [1, 0], [2, 0], [3, 0]],
            [[1, 1], [2, 1], [3, 1], [4, 1]],
            [[2, 2], [3, 2], [4, 2], [5, 2]],
        ]
    )
    target = sample**2
    batch, _ = approximator._generate_batch(
        batch_size=1, sample=sample, target=target, seed=0
    )
    benchmark = torch.Tensor([[[2.0, 2.0], [3.0, 2.0], [4.0, 2.0], [5.0, 2.0]]])

    assert tensors_are_close(batch, benchmark)


def test_generate_sample_batch_target():
    approximator = setup()
    sample = torch.Tensor(
        [
            [[0, 0], [1, 0], [2, 0], [3, 0]],
            [[1, 1], [2, 1], [3, 1], [4, 1]],
            [[2, 2], [3, 2], [4, 2], [5, 2]],
        ]
    )
    target = sample**2
    _, batch_target = approximator._generate_batch(
        batch_size=1, sample=sample, target=target, seed=0
    )
    benchmark = torch.Tensor([[[4.0, 4.0], [9.0, 4.0], [16.0, 4.0], [25.0, 4.0]]])

    assert tensors_are_close(batch_target, benchmark)
