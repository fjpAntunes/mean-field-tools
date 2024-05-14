import numpy as np
from finite_differences.fokker_planck import FokkerPlanckEquation

ONE_DIM_DOMAIN = np.linspace(0, 1, 1_000)
TIME_DOMAIN = np.linspace(0, 1, 1_000)
TOLERANCE = TIME_DOMAIN[1] - TIME_DOMAIN[0]
one_dim_fokker_planck = FokkerPlanckEquation(
    space_domain=[ONE_DIM_DOMAIN],
    time_domain=TIME_DOMAIN,
    vector_field=[ONE_DIM_DOMAIN],
    initial_condition=np.array([0]*498 + [0.5,1,1,0.5] + [0]*498),
    volatility=1,
    number_of_dimensions=1,
    tolerance=TOLERANCE,
)

ONE_DIM_LINEAR_SCALAR = ONE_DIM_DOMAIN


def test_calculate_divergence_term_one_dim_linear():
    output = one_dim_fokker_planck._calculate_divergence_term(ONE_DIM_LINEAR_SCALAR)
    np.testing.assert_array_almost_equal(output, 2 * ONE_DIM_DOMAIN, decimal=2)


def test_calculate_second_derivative_term_one_dim_linear():
    output = one_dim_fokker_planck._calculate_second_derivative_term(
        ONE_DIM_LINEAR_SCALAR
    )
    np.testing.assert_array_almost_equal(
        output, np.zeros_like(ONE_DIM_DOMAIN), decimal=2
    )


def test_calculate_time_derivative_one_dim_linear():
    output = one_dim_fokker_planck._calculate_time_derivative(ONE_DIM_LINEAR_SCALAR)
    np.testing.assert_almost_equal(output, -2 * ONE_DIM_DOMAIN, decimal=2)

ONE_DIM_QUADRATIC_SCALAR = ONE_DIM_DOMAIN**2

def test_calculate_divergence_term_one_dim_test_quadratic():
    output = one_dim_fokker_planck._calculate_divergence_term(ONE_DIM_QUADRATIC_SCALAR)
    np.testing.assert_almost_equal(output, 3 * ONE_DIM_DOMAIN**2, decimal=2)


def test_calculate_second_derivative_term_one_dim_quadratic():
    output = one_dim_fokker_planck._calculate_second_derivative_term(
        ONE_DIM_QUADRATIC_SCALAR
    )
    np.testing.assert_array_almost_equal(
        output[2:-2], 2 * np.ones_like(ONE_DIM_DOMAIN)[2:-2], decimal=2
    )


def test_calculate_time_derivative_one_dim_quadratic():
    output = one_dim_fokker_planck._calculate_time_derivative(ONE_DIM_QUADRATIC_SCALAR)
    np.testing.assert_array_almost_equal(
        output[2:-2],
        -3 * ONE_DIM_DOMAIN[2:-2] ** 2 + 2 * np.ones_like(ONE_DIM_DOMAIN)[2:-2],
        decimal=2,
    )


X_AXIS = np.linspace(0, 1, 1_000)
Y_AXIS = np.linspace(0, 1, 1_000)
X, Y = np.meshgrid(X_AXIS, Y_AXIS, indexing="ij")
TWO_DIM_DOMAIN = [X, Y]
two_dim_fokker_planck = FokkerPlanckEquation(
    space_domain=TWO_DIM_DOMAIN,
    time_domain=TIME_DOMAIN,
    vector_field=TWO_DIM_DOMAIN,
    initial_condition=np.zeros_like(TWO_DIM_DOMAIN[0]),
    volatility=np.array([1, 1]),
    number_of_dimensions=2,
    tolerance=TOLERANCE,
)


def test_calculate_divergence_term_two_dim_test():
    output = two_dim_fokker_planck._calculate_divergence_term(np.ones((1_000, 1_000)))
    np.testing.assert_array_almost_equal(output, 2 * np.ones((1_000, 1_000)), decimal=2)


def test_calculate_divergence_term_two_dim_test_2():
    output = two_dim_fokker_planck._calculate_divergence_term(scalar_field=X * Y)
    np.testing.assert_array_almost_equal(output, 4 * X * Y, decimal=2)


def test_calculate_second_derivative_term_two_dim_linear():
    output = two_dim_fokker_planck._calculate_second_derivative_term(X + Y)
    np.testing.assert_array_almost_equal(
        output, np.zeros(shape=(2, 2, 1_000, 1_000)), decimal=2
    )


def test_calculate_second_derivative_term_two_dim_quadratic():
    output = two_dim_fokker_planck._calculate_second_derivative_term(
        X**2 + 2 * X * Y + Y**2
    )
    np.testing.assert_array_almost_equal(
        output[:, :, 2:-2, 2:-2],
        2 * np.ones(shape=(2, 2, 1_000, 1_000))[:, :, 2:-2, 2:-2],
        decimal=2,
    )


def test_calculate_time_derivative_term_two_dim():
    output = two_dim_fokker_planck._calculate_time_derivative(scalar_field=X + Y)
    result = -3 * (X + Y)
    np.testing.assert_array_almost_equal(output, result, decimal=2)


def test_calculate_time_derivative_term_two_dim_2():
    output = two_dim_fokker_planck._calculate_time_derivative(
        scalar_field=X**2 + 2 * X * Y + Y**2
    )
    result = -4 * (X**2 + 2 * X * Y + Y**2) + 4 * np.ones_like(X)
    np.testing.assert_array_almost_equal(
        output[2:-2, 2:-2], result[2:-2, 2:-2], decimal=2
    )
