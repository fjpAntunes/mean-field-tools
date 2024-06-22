import torch


def L_inf_norm(x):
    return torch.max(torch.abs(x))


def tensors_are_close(a, b, tolerance=1e-10, norm=torch.norm):
    return norm(a - b) < tolerance
