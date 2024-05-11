import numpy as np
from typing import List

class FokkerPlanckEquation():
    def __init__(
            self,
            space_domain : List[np.array], # meshgrid-like
            time_domain : np.array, # linspace-like
            vector_field : List[np.array], 
            initial_condition : np.array,
            volatility : np.array,
            number_of_dimensions : int,
            tolerance : float
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
             raise Exception('Number of arrays in space domain does not match number of dimensions')        

    def _calculate_divergence_term(self, scalar_field):
        partials = []
        for dimension,field in enumerate(self.vector_field):
            dislocation = scalar_field * field
            partial = np.gradient(dislocation,self.tolerance)
            if self.number_of_dimensions > 1:
                partial = partial[dimension] 
            partials.append(partial)

        return sum(partials)

    def _calculate_second_derivative_term(scalar_field):
         pass