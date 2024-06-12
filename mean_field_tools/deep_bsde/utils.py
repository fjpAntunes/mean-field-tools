import torch


def tensors_are_close(a, b, tolerance):
    return torch.norm(a - b) < tolerance
