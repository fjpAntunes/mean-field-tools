import torch


def tensors_are_close(a, b, tolerance, norm=torch.norm):
    return norm(a - b) < tolerance
