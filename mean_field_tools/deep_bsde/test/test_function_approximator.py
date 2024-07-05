from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator
from types import SimpleNamespace
import torch


function_approximator = FunctionApproximator(
    domain_dimension=1, output_dimension=1, device="cpu"
)


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
    mock_device = MockDevice(type="cuda")
    mock_input = MockTensor(device=mock_device)
    processed_input = function_approximator.preprocess(mock_input)

    assert processed_input.device.type == function_approximator.device


def test_postprocess():
    """If function_approximator is not training, should return output on cpu device for interactivity."""
    mock_device = MockDevice(type="cuda")
    mock_output = MockTensor(device=mock_device)
    processed_output = function_approximator.postprocess(
        mock_output, training_status=False
    )

    assert processed_output.device.type == "cpu"
