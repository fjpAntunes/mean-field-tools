r"""Tests quadratic with drift
Equation:
$$
dY_t = -B_tdt + Z_t dB_t, \quad Y_T = B^2_T, \\
$$
Where $W_t$ is the standard brownian motion.
Writing the equation in the form $dY_t = - f(t,B_t)dt + Z_t dB_t$,
we have $f(t,B_t) = B_t$.
"""

from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, BackwardSDE
from mean_field_tools.deep_bsde.function_approximator import FunctionApproximatorArtist
import torch


def test_terminal_quadratic_with_stochastic_drift():

    TIME_DOMAIN = torch.linspace(0, 1, 101)
    NUMBER_OF_PATHS = 100
    SPATIAL_DIMENSIONS = 1

    def TERMINAL_CONDITION(filtration: Filtration):
        B_T = filtration.brownian_process[:, -1, :]
        return B_T**2

    def DRIFT(filtration: Filtration):
        B_t = filtration.brownian_process

        return B_t

    def ANALYTICAL_SOLUTION(x, t, T):
        return x**2 + (T - t) + x * (T - t)

    filtration = Filtration(SPATIAL_DIMENSIONS, TIME_DOMAIN, NUMBER_OF_PATHS, seed=0)

    bsde = BackwardSDE(
        terminal_condition_function=TERMINAL_CONDITION,
        drift=DRIFT,
        filtration=filtration,
    )

    bsde.initialize_approximator()

    artist = FunctionApproximatorArtist(
        save_figures=False, analytical_solution=ANALYTICAL_SOLUTION
    )

    bsde.solve(
        approximator_args={
            "batch_size": 100,
            "number_of_iterations": 500,
            "number_of_epochs": 5,
            "number_of_plots": 5,
            "plotter": artist,
        }
    )

    forward_path = bsde.filtration.get_paths()[:1, :, :]

    output = bsde.y_approximator(forward_path).tolist()

    benchmark = [
        [
            [0.9830755591392517],
            [0.8605413436889648],
            [0.7792044878005981],
            [0.7604942321777344],
            [0.7423194646835327],
            [0.7554470896720886],
            [0.7825713753700256],
            [0.7530415654182434],
            [0.7176299691200256],
            [0.7007310390472412],
            [0.7558783888816833],
            [0.7231760025024414],
            [0.6978869438171387],
            [0.6844171285629272],
            [0.6534522771835327],
            [0.6755509376525879],
            [0.655299961566925],
            [0.6299310326576233],
            [0.7138860821723938],
            [0.6612969636917114],
            [0.6143127083778381],
            [0.5944817662239075],
            [0.6521489024162292],
            [0.6744847893714905],
            [0.5710240602493286],
            [0.5642073750495911],
            [0.5536953806877136],
            [0.5479097962379456],
            [0.5380285382270813],
            [0.5512526035308838],
            [0.6355940699577332],
            [0.7093220949172974],
            [0.6223718523979187],
            [0.5684196949005127],
            [0.5605968236923218],
            [0.523625910282135],
            [0.5271946787834167],
            [0.5436842441558838],
            [0.5412006378173828],
            [0.5775149464607239],
            [0.6036620140075684],
            [0.5840500593185425],
            [0.6459805369377136],
            [0.6066826581954956],
            [0.600407063961029],
            [0.6399012804031372],
            [0.9166895151138306],
            [0.7069821357727051],
            [0.5371138453483582],
            [0.4767749011516571],
            [0.532813310623169],
            [0.621388852596283],
            [0.6283804178237915],
            [0.5927056074142456],
            [0.5437761545181274],
            [0.5850329995155334],
            [0.5356091260910034],
            [0.48626312613487244],
            [0.5415833592414856],
            [0.6973958015441895],
            [1.2320770025253296],
            [0.9395639300346375],
            [0.7412846088409424],
            [1.0060471296310425],
            [0.8987520337104797],
            [0.7972004413604736],
            [0.9245228171348572],
            [1.1029034852981567],
            [1.342725396156311],
            [1.03464674949646],
            [1.5468831062316895],
            [1.419365406036377],
            [1.4752706289291382],
            [1.115024447441101],
            [0.9966011643409729],
            [0.8980920910835266],
            [0.8031572699546814],
            [0.638849139213562],
            [0.7810178995132446],
            [0.7455794811248779],
            [0.855419933795929],
            [0.8250831961631775],
            [0.9235996007919312],
            [0.7715939879417419],
            [0.7570815682411194],
            [0.8765464425086975],
            [0.7958219051361084],
            [0.9555879831314087],
            [1.004379153251648],
            [0.9427197575569153],
            [1.394758701324463],
            [0.9860992431640625],
            [0.9601462483406067],
            [0.7663520574569702],
            [0.6121648550033569],
            [0.6033919453620911],
            [0.6900011301040649],
            [0.9477577209472656],
            [0.7041655778884888],
            [0.7427859902381897],
            [0.812171220779419],
        ]
    ]

    assert benchmark == output
