"""Tests Ornstein-Uhlenbeck as forward process"""

from mean_field_tools.deep_bsde.forward_backward_sde import (
    Filtration,
    BackwardSDE,
    ForwardSDE,
    ForwardBackwardSDE,
)
from mean_field_tools.deep_bsde.function_approximator import FunctionApproximatorArtist
import torch
import numpy as np


def test_analytical_forward_numerical_backward():
    TIME_DOMAIN = torch.linspace(0, 1, 101)
    NUMBER_OF_PATHS = 100
    SPATIAL_DIMENSIONS = 1

    K = 1

    def OU_FUNCTIONAL_FORM(filtration):
        dummy_time = filtration.time_process[:, 1:, 0].unsqueeze(-1)
        integrand = torch.exp(K * dummy_time) * filtration.brownian_increments

        initial = torch.zeros(
            size=(filtration.number_of_paths, 1, filtration.spatial_dimensions)
        )
        integral = torch.cat([initial, torch.cumsum(integrand, dim=1)], dim=1)

        time = filtration.time_process[:, :, 0].unsqueeze(-1)
        path = torch.exp(-K * time) * integral
        return path

    def BACKWARD_DRIFT(filtration: Filtration):
        X_t = filtration.forward_process

        return 2 * X_t

    def TERMINAL_CONDITION(filtration: Filtration):
        X_T = filtration.forward_process[:, -1, :]

        return X_T**2

    def ANALYTICAL_SOLUTION(x, t, T):
        return (
            x**2 * np.exp(-2 * K * (T - t))
            + ((1 - np.exp(-2 * K * (T - t))) / (2 * K))
            + 2 * x * ((1 - np.exp(-K * (T - t))) / K)
        )

    FILTRATION = Filtration(
        spatial_dimensions=SPATIAL_DIMENSIONS,
        time_domain=TIME_DOMAIN,
        number_of_paths=NUMBER_OF_PATHS,
        seed=0,
    )

    forward_sde = ForwardSDE(
        filtration=FILTRATION,
        functional_form=OU_FUNCTIONAL_FORM,
    )

    backward_sde = BackwardSDE(
        terminal_condition_function=TERMINAL_CONDITION,
        filtration=FILTRATION,
        exogenous_process=["time_process", "forward_process"],
        drift=BACKWARD_DRIFT,
    )
    backward_sde.initialize_approximator()

    forward_backward_sde = ForwardBackwardSDE(
        filtration=FILTRATION, forward_sde=forward_sde, backward_sde=backward_sde
    )

    artist = FunctionApproximatorArtist(
        save_figures=False, analytical_solution=ANALYTICAL_SOLUTION
    )

    APPROXIMATOR_ARGS = {
        "batch_size": 100,
        "number_of_iterations": 500,
        "number_of_epochs": 5,
        "number_of_plots": 5,
        "plotter": artist,
    }

    forward_backward_sde.backward_solve(approximator_args=APPROXIMATOR_ARGS)

    output = backward_sde.generate_paths()[0, :, :].tolist()

    benchmark = [
        [0.3472933769226074],
        [0.21301814913749695],
        [0.08738366514444351],
        [0.06587158888578415],
        [0.027653194963932037],
        [0.11833702772855759],
        [0.19727928936481476],
        [0.16492518782615662],
        [-0.03479895740747452],
        [-0.0025067944079637527],
        [-0.09434111416339874],
        [-0.06395888328552246],
        [-0.0356699675321579],
        [-0.02142864279448986],
        [0.09132206439971924],
        [0.20679111778736115],
        [0.18252843618392944],
        [0.05483339726924896],
        [-0.07021968811750412],
        [-0.0272169578820467],
        [0.0388026162981987],
        [0.09422904998064041],
        [-0.02760639227926731],
        [-0.04627589136362076],
        [0.10107193142175674],
        [0.17182022333145142],
        [0.11947363615036011],
        [0.10623624920845032],
        [0.12416332960128784],
        [0.2532872259616852],
        [0.4091138541698456],
        [0.5026235580444336],
        [0.410480260848999],
        [0.3457365334033966],
        [0.34561946988105774],
        [0.29583540558815],
        [0.3164914548397064],
        [0.35469764471054077],
        [0.3614395260810852],
        [0.4181967079639435],
        [0.455602765083313],
        [0.43953937292099],
        [0.5112019777297974],
        [0.47517848014831543],
        [0.4733372926712036],
        [0.5175350308418274],
        [0.7578842043876648],
        [0.5833508968353271],
        [0.41953104734420776],
        [0.35351771116256714],
        [0.42515847086906433],
        [0.517522394657135],
        [0.5260401368141174],
        [0.49452999234199524],
        [0.449307382106781],
        [0.4917576014995575],
        [0.4463088810443878],
        [0.3990602195262909],
        [0.45718199014663696],
        [0.5967037677764893],
        [1.0180854797363281],
        [0.7849396467208862],
        [0.6240378618240356],
        [0.8303413987159729],
        [0.742106020450592],
        [0.6588395237922668],
        [0.7547087073326111],
        [0.8877749443054199],
        [1.0672621726989746],
        [0.8205132484436035],
        [1.2107858657836914],
        [1.1008551120758057],
        [1.1342735290527344],
        [0.8466917872428894],
        [0.750714898109436],
        [0.6719896197319031],
        [0.5978546738624573],
        [0.47579795122146606],
        [0.5742063522338867],
        [0.5446513891220093],
        [0.6178300380706787],
        [0.5908410549163818],
        [0.6547180414199829],
        [0.5424653887748718],
        [0.5276397466659546],
        [0.6040235757827759],
        [0.5428470969200134],
        [0.6458611488342285],
        [0.6721121072769165],
        [0.6213446855545044],
        [0.9423855543136597],
        [0.6323975920677185],
        [0.606735110282898],
        [0.47572246193885803],
        [0.3790335953235626],
        [0.3702797293663025],
        [0.4158076047897339],
        [0.5660786032676697],
        [0.4126059412956238],
        [0.42913419008255005],
        [0.4628603458404541],
    ]

    assert output == benchmark
