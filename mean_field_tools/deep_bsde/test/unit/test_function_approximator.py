from mean_field_tools.deep_bsde.function_approximator import FunctionApproximator
from mean_field_tools.deep_bsde.utils import tensors_are_close
import torch


def setup():
    function_approximator = FunctionApproximator(
        number_of_layers=2,
        number_of_nodes=2,
        domain_dimension=2,
        output_dimension=1,
        device="cpu",
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


def test_single_training_step():
    approximator = setup()
    approximator.training_setup()
    sample = torch.Tensor(
        [
            [[0, 0], [1, 0], [2, 0], [3, 0]],
            [[1, 1], [2, 1], [3, 1], [4, 1]],
            [[2, 2], [3, 2], [4, 2], [5, 2]],
        ]
    )
    target = sample**2
    batch_sample, batch_target = approximator._generate_batch(
        batch_size=1, sample=sample, target=target, seed=0
    )

    approximator.single_gradient_descent_step(batch_sample, batch_target)

    benchmark = {
        "input.weight": [
            [-0.5869807600975037, -0.5253874659538269],
            [-0.2773452401161194, 0.18461589515209198],
        ],
        "input.bias": [-0.019010011106729507, 0.5556575059890747],
        "hidden.0.weight": [
            [-0.05775151774287224, 0.19210933148860931],
            [-0.21869690716266632, -0.14399270713329315],
        ],
        "hidden.0.bias": [-0.6805334091186523, -0.46330416202545166],
        "output.weight": [[-0.2964857518672943, 0.021193761378526688]],
        "output.bias": [0.2845441997051239],
    }

    output = {name: param.tolist() for name, param in approximator.named_parameters()}

    assert benchmark == output

def test_gradient_with_respect_to_input():
    """Tests simple point gradient.
    The variable point_sample should represent a state input point 
    """
    approximator = setup()
    approximator.forward = lambda x: torch.sum(x, axis = -1)
    point_sample = torch.Tensor([[0, 0]])
    point_sample.requires_grad = True
    test = approximator.grad(point_sample)
    test = test[0]

    benchmark = [1, 1]
    assert test.tolist() == benchmark

def test_batch_gradient_with_respect_to_input():
    """Tests batch point gradients.
    """
    approximator = setup()
    approximator.forward = lambda x: torch.sum(x, axis = -1) # Override forward to a simple derivative function.
    point_sample = torch.Tensor([[0.5,0.5],[0, 0]])
    point_sample.requires_grad = True
    test = approximator.grad(point_sample)
    benchmark = [[1,1],[1,1]]
    assert test.tolist() == benchmark
    