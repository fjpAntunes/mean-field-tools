import numpy as np
from typing import List


class FokkerPlanckEquation:
    def __init__(
        self,
        space_domain: List[np.array],  # meshgrid-like
        time_domain: np.array,  # linspace-like
        vector_field: List[np.array],
        initial_condition: np.array,
        volatility: np.array,
        number_of_dimensions: int,
        tolerance: float,
    ):

        self.space_domain = space_domain
        self.time_domain = time_domain
        self.vector_field = vector_field
        self.initial_condition = initial_condition
        self.volatility = volatility
        self.number_of_dimensions = number_of_dimensions
        self.tolerance = tolerance

        self._sanity_check()

    def _sanity_check(self):
        self._check_number_of_dimensions_and_space_domain_compatibility()

    def _check_number_of_dimensions_and_space_domain_compatibility(self):
        if len(self.space_domain) != self.number_of_dimensions:
            raise Exception(
                "Number of arrays in space domain does not match number of dimensions"
            )

    def _calculate_divergence_term(self, scalar_field):
        partials = []
        for dimension, field in enumerate(self.vector_field):
            dislocation = scalar_field * field
            partial = np.gradient(dislocation, self.tolerance, edge_order=2)
            if self.number_of_dimensions > 1:
                partial = partial[dimension]
            partials.append(partial)

        return sum(partials)

    def _calculate_second_derivative_term(self, scalar_field):
        grad = np.gradient(scalar_field, self.tolerance)
        if self.number_of_dimensions > 1:
            hessian = []
            for partial in grad:
                hessian.append(np.gradient(partial, self.tolerance, edge_order=2))
            hessian = np.array(hessian)
        else:
            hessian = np.gradient(grad, self.tolerance, edge_order=2)
        return hessian

    def _calculate_time_derivative(self, scalar_field):
        divergence = self._calculate_divergence_term(scalar_field)
        hessian = self._calculate_second_derivative_term(scalar_field)
        if self.number_of_dimensions == 1:
            time_derivative = -divergence + self.volatility * hessian
        else:
            time_derivative = -divergence
            for i in range(self.number_of_dimensions):
                time_derivative = (
                    time_derivative + self.volatility[i] * hessian[i, i, :, :]
                )
        return time_derivative, divergence, hessian

    def _calculate_updated_scalar_field(self, scalar_field, time_derivative, step):
        return scalar_field + time_derivative * step

    def _integrate_over_time(self):
        timestep = self.time_domain[1] - self.time_domain[0]
        scalar_field = self.initial_condition
        solution = [scalar_field]
        time_derivatives = []
        divergences = []
        hessians = []
        for _ in self.time_domain:
            time_derivative, divergence, hessian = self._calculate_time_derivative(
                scalar_field
            )
            time_derivatives.append(time_derivative)
            divergences.append(divergence)
            hessians.append(hessian)
            scalar_field = self._calculate_updated_scalar_field(
                scalar_field, time_derivative, timestep
            )
            solution.append(scalar_field)

        return (
            np.array(solution),
            np.array(time_derivatives),
            np.array(divergences),
            np.array(hessians),
        )
