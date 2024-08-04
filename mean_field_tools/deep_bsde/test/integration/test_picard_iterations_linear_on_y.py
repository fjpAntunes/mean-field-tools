r"""
Test Setup:

* Forward component:
  $$ dX_t = -kX_t dt + dW_t $$

* Backwards component:
$$ dY_t= - ( Y_t + 2X_t)dt + Z_t dW_t $$
$$ Y_T = X_T^2 $$
"""

from mean_field_tools.deep_bsde.forward_backward_sde import (
    Filtration,
    BackwardSDE,
    ForwardSDE,
    ForwardBackwardSDE,
)
from mean_field_tools.deep_bsde.function_approximator import FunctionApproximatorArtist
import torch
import numpy as np


def test_picard_iterations_linear_on_y():
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def LINEAR_BACKWARD_DRIFT(filtration: Filtration):
        X_t = filtration.forward_process
        Y_t = filtration.backward_process
        return Y_t + 2 * X_t

    def LINEAR_TERMINAL_CONDITION(filtration: Filtration):
        X_T = filtration.forward_process[:, -1, :]

        return X_T**2

    backward_sde = BackwardSDE(
        terminal_condition_function=LINEAR_TERMINAL_CONDITION,
        filtration=FILTRATION,
        exogenous_process=["time_process", "forward_process", "backward_process"],
        drift=LINEAR_BACKWARD_DRIFT,
    )
    backward_sde.initialize_approximator(nn_args={"device": device})

    forward_backward_sde = ForwardBackwardSDE(
        filtration=FILTRATION, forward_sde=forward_sde, backward_sde=backward_sde
    )

    PICARD_ITERATION_ARGS = {
        "training_strategy_args": {
            "batch_size": 100,
            "number_of_iterations": 400,
            "number_of_batches": 4,
        },
    }

    forward_backward_sde.backward_solve(
        number_of_iterations=5, approximator_args=PICARD_ITERATION_ARGS
    )

    output = forward_backward_sde.filtration.backward_process[0, :, 0].tolist()
    benchmark = [
        1.1250741481781006,
        0.9509891271591187,
        0.6057495474815369,
        0.5665613412857056,
        0.5020486116409302,
        0.7211928963661194,
        0.8350042700767517,
        0.7591047883033752,
        0.5002862811088562,
        0.4779666066169739,
        -0.04842851683497429,
        0.5005406141281128,
        0.4196411073207855,
        0.3832111656665802,
        0.5689937472343445,
        0.7198563814163208,
        0.6795598864555359,
        0.5497263669967651,
        0.517410159111023,
        0.556255042552948,
        0.5780248641967773,
        0.589715301990509,
        0.5966103672981262,
        0.595687747001648,
        0.5461584329605103,
        0.5364422798156738,
        0.4854357838630676,
        0.42476886510849,
        0.41269651055336,
        0.46535277366638184,
        0.7416397333145142,
        0.9420880675315857,
        0.7804662585258484,
        0.6327571272850037,
        0.6527754664421082,
        0.5204687118530273,
        0.5926328897476196,
        0.7074589133262634,
        0.7156996726989746,
        0.8384636640548706,
        0.8862472176551819,
        0.8873414397239685,
        0.9576880931854248,
        0.966304361820221,
        0.9975845813751221,
        1.0361530780792236,
        0.9713351130485535,
        1.0666558742523193,
        1.1082632541656494,
        0.8143969774246216,
        1.094415307044983,
        1.1046314239501953,
        1.0733212232589722,
        1.0347414016723633,
        0.9409101605415344,
        0.9297773838043213,
        0.8296433687210083,
        0.7166968584060669,
        0.7627468705177307,
        0.8113058805465698,
        1.550317645072937,
        0.9625598192214966,
        0.780507504940033,
        1.0369083881378174,
        0.8830531239509583,
        0.8020057082176208,
        0.9075334668159485,
        1.127750039100647,
        1.754964828491211,
        1.0048249959945679,
        1.9136745929718018,
        1.652431845664978,
        1.6724143028259277,
        1.0534104108810425,
        0.9483596086502075,
        0.840809166431427,
        0.7311545014381409,
        0.573782742023468,
        0.6948237419128418,
        0.6549240350723267,
        0.7488437294960022,
        0.708591878414154,
        0.7903176546096802,
        0.6416157484054565,
        0.620368242263794,
        0.7025060057640076,
        0.6245532631874084,
        0.7358950972557068,
        0.757916271686554,
        0.6816139817237854,
        1.065155029296875,
        0.6653161644935608,
        0.6207394003868103,
        0.475166916847229,
        0.3545059859752655,
        0.3296808898448944,
        0.3679199814796448,
        0.4966379404067993,
        0.33488228917121887,
        0.33791157603263855,
        0.3574211001396179,
    ]

    assert benchmark == output
