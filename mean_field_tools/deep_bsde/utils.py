from mean_field_tools.deep_bsde.filtration import Filtration
import torch


def L_inf_norm(x):
    return torch.max(torch.abs(x))


def L_2_norm(x):
    return torch.mean(x**2) + torch.var(x)


def tensors_are_close(a, b, tolerance=1e-10, norm=torch.norm):
    return norm(a - b) < tolerance


def QUADRATIC_TERMINAL(filtration: Filtration):
    terminal_brownian = filtration.brownian_process[:, -1, :]
    return terminal_brownian**2


def IDENTITY_TERMINAL(filtration: Filtration):
    terminal_brownian = filtration.brownian_process[:, -1, :]
    return terminal_brownian
