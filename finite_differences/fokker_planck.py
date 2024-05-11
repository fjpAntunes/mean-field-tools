import numpy as np

class FokkerPlanckEquation():
    def __init__(
            self,
            space_domain ,
            time_domain,
            vector_field, # tuple(n_dim over meshgrid) 
            initial_condition,
            volatility,
            number_of_dimensions,
            tolerance = 1e-3
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
        self._check_space_domain_and_tolerance_compatibility()
        self._check_time_domain_and_tolerance_compatibility()
        pass

    def _check_number_of_dimensions_and_space_domain_compatibility(self):
        pass

    def _check_space_domain_and_tolerance_compatibility(self):
        pass

    def _check_time_domain_and_tolerance_compatibility(self):
        pass

    def _calculate_divergence_term(self, scalar_field):
        partials = []
        for dimension,field in enumerate(self.vector_field):
            #if self.number_of_dimensions == 2: import pdb; pdb.set_trace()
            dislocation = scalar_field * field
            partial = np.gradient(dislocation,self.tolerance)
            if self.number_of_dimensions > 1:
                partial = partial[dimension] 
            partials.append(partial)

        return sum(partials)

    def _calculate_second_derivative_term(scalar_field):
         pass