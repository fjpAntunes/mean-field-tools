"""Deep BSDE solvers for mean field problems.

The names re-exported here are the supported public API. Anything reachable
through a module path but absent from this list is an implementation detail and
may change without a major version bump.
"""

from mean_field_tools.deep_bsde.artist import (
    FunctionApproximatorArtist,
    PicardIterationsArtist,
)
from mean_field_tools.deep_bsde.filtration import CommonNoiseFiltration, Filtration
from mean_field_tools.deep_bsde.forward_backward_sde import (
    AnalyticForwardSDE,
    BackwardSDE,
    CommonNoiseBackwardSDE,
    ForwardBackwardSDE,
    NumericalForwardSDE,
)
from mean_field_tools.deep_bsde.function_approximator import (
    FunctionApproximator,
    HybridApproximator,
    PathDependentApproximator,
)
from mean_field_tools.deep_bsde.measure_flow import (
    CommonNoiseMeasureFlow,
    MeasureFlow,
)
from mean_field_tools.deep_bsde.utils import (
    IDENTITY_TERMINAL,
    L_2_norm,
    L_inf_norm,
    QUADRATIC_TERMINAL,
)

__all__ = [
    # Filtration
    "Filtration",
    "CommonNoiseFiltration",
    # Forward / backward SDEs
    "NumericalForwardSDE",
    "AnalyticForwardSDE",
    "BackwardSDE",
    "CommonNoiseBackwardSDE",
    "ForwardBackwardSDE",
    # Measure flow
    "MeasureFlow",
    "CommonNoiseMeasureFlow",
    # Function approximators
    "FunctionApproximator",
    "PathDependentApproximator",
    "HybridApproximator",
    # Plotting and diagnostics
    "FunctionApproximatorArtist",
    "PicardIterationsArtist",
    # Terminal conditions and norms
    "IDENTITY_TERMINAL",
    "QUADRATIC_TERMINAL",
    "L_2_norm",
    "L_inf_norm",
]
