from mean_field_tools.deep_bsde.forward_backward_sde import Filtration, BackwardSDE
from mean_field_tools.deep_bsde.utils import (
    QUADRATIC_TERMINAL,
    tensors_are_close,
    L_2_norm,
)
import torch


NUMBER_OF_TIMESTEPS = 101
TIME_DOMAIN = torch.linspace(0, 1, NUMBER_OF_TIMESTEPS)
NUMBER_OF_PATHS = 100
SPATIAL_DIMENSIONS = 1


filtration = Filtration(SPATIAL_DIMENSIONS, TIME_DOMAIN, NUMBER_OF_PATHS, seed=0)

bsde = BackwardSDE(
    terminal_condition_function=QUADRATIC_TERMINAL,
    filtration=filtration,
)

bsde.initialize_approximator()

bsde.solve(
    approximator_args={
        "training_strategy_args": {
            "batch_size": 100,
            "number_of_iterations": 500,
            "number_of_batches": 5,
            "number_of_plots": 5,
        },
    }
)


def test_backward_process_shape():
    backward_process = bsde.generate_backward_process()
    assert backward_process.shape == (
        NUMBER_OF_PATHS,
        NUMBER_OF_TIMESTEPS,
        SPATIAL_DIMENSIONS,
    )


def test_backward_volatility_shape():
    backward_volatility = bsde.generate_backward_volatility()

    assert backward_volatility.shape == (
        NUMBER_OF_PATHS,
        NUMBER_OF_TIMESTEPS,
        SPATIAL_DIMENSIONS,
    )


def test_backward_volatility_value():
    z = filtration.brownian_process * 2
    backward_volatility = bsde.generate_backward_volatility()
    benchmark = [
        [0.19223235547542572],
        [-0.04295133426785469],
        [-0.3024246096611023],
        [-0.36063510179519653],
        [-0.46712979674339294],
        [-0.2569470703601837],
        [-0.09816814213991165],
        [-0.16655416786670685],
        [-0.7054838538169861],
        [-0.6106205582618713],
        [-1.0205209255218506],
        [-0.8957747220993042],
        [-0.7927541732788086],
        [-0.7545130848884583],
        [-0.40177851915359497],
        [-0.13674141466617584],
        [-0.19084274768829346],
        [-0.5282365679740906],
        [-1.1349833011627197],
        [-0.9012465476989746],
        [-0.6290470957756042],
        [-0.4557141065597534],
        [-0.9873480796813965],
        [-1.158875584602356],
        [-0.4699849784374237],
        [-0.27582526206970215],
        [-0.4268684983253479],
        [-0.47733309864997864],
        [-0.42736750841140747],
        [-0.09339283406734467],
        [0.21969303488731384],
        [0.3926367461681366],
        [0.23543959856033325],
        [0.11831178516149521],
        [0.12259438633918762],
        [0.025275709107518196],
        [0.07201085984706879],
        [0.1532703936100006],
        [0.17177726328372955],
        [0.28562286496162415],
        [0.3613656759262085],
        [0.33988508582115173],
        [0.47686225175857544],
        [0.42168375849723816],
        [0.4270651340484619],
        [0.5153017044067383],
        [0.9380822777748108],
        [0.6557795405387878],
        [0.37076881527900696],
        [0.2507031261920929],
        [0.39681094884872437],
        [0.5760132074356079],
        [0.602522075176239],
        [0.5564913749694824],
        [0.4833090603351593],
        [0.5726630091667175],
        [0.49879127740859985],
        [0.4183090329170227],
        [0.5396924614906311],
        [0.8097812533378601],
        [1.5174967050552368],
        [1.1779274940490723],
        [0.9163206815719604],
        [1.2924587726593018],
        [1.1661295890808105],
        [1.0388253927230835],
        [1.2295026779174805],
        [1.4732370376586914],
        [1.765851616859436],
        [1.4184985160827637],
        [2.011117696762085],
        [1.8988817930221558],
        [1.9747931957244873],
        [1.5835036039352417],
        [1.446982979774475],
        [1.3278687000274658],
        [1.2062128782272339],
        [0.9670677185058594],
        [1.2036141157150269],
        [1.1649013757705688],
        [1.3458575010299683],
        [1.3170976638793945],
        [1.4780194759368896],
        [1.268105387687683],
        [1.2617990970611572],
        [1.4611730575561523],
        [1.3557045459747314],
        [1.6131131649017334],
        [1.701775312423706],
        [1.6319743394851685],
        [2.2441227436065674],
        [1.734273076057434],
        [1.7162692546844482],
        [1.4340853691101074],
        [1.1909732818603516],
        [1.1931949853897095],
        [1.3626081943511963],
        [1.8016726970672607],
        [1.4256318807601929],
        [1.5120831727981567],
        [1.6500576734542847],
    ]

    assert L_2_norm(z - backward_volatility) < 0.5


def test_calculate_backward_volatility_integral_shape():
    volatility_integral = bsde.calculate_volatility_integral()

    assert volatility_integral.shape == (
        NUMBER_OF_PATHS,
        NUMBER_OF_TIMESTEPS,
        SPATIAL_DIMENSIONS,
    )


def test_calculate_backward_volatility_integral_value():
    volatility_integral = bsde.calculate_volatility_integral()

    backward_volatility = bsde.generate_backward_volatility()

    dt = filtration.dt
    quadratic_variation = torch.sum((backward_volatility**2) * dt, dim=1).unsqueeze(
        -1
    ) - torch.cumsum((backward_volatility**2) * dt, dim=1)
    ito_isometry_deviation = torch.mean(volatility_integral**2, dim=0) - torch.mean(
        quadratic_variation, dim=0
    )

    assert L_2_norm(ito_isometry_deviation) < 0.3


def test_calculate_picard_operator():
    picard_operator = bsde.calculate_picard_operator()
    y = bsde.generate_backward_process()
    deviations = picard_operator - y

    assert torch.mean(deviations**2) < 0.4
