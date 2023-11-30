from scipy.integrate import odeint
from scipy.interpolate import interp1d
import numpy as np

class EconomicModelMFG():

    def __init__(self,
                 time_domain = [],
                 initial_population_sample = [],
                 feedback_control = lambda x : x,
                 mean_field_interest_rate = lambda x : x,
                 costate_terminal_condition = lambda x : x,
                 minimum_capital = -np.inf,
                 **kwargs):

        self.time_domain = time_domain
        self.feedback_control = feedback_control
        self.mean_field_interest_rate = mean_field_interest_rate
        self.costate_terminal_condition = costate_terminal_condition
        self.minimum_capital = minimum_capital

        self.arg_list = ['delta','w']
        self.delta, self.w  =  (kwargs[name] for name in self.arg_list)

        self.initial_population_sample =initial_population_sample
        self.number_of_samples = len(self.initial_population_sample)


    '''
    Utility functions
    '''
    def _backward_odeint(self, backward_dot, terminal_condition, time_domain, args = ()):
        """backward ode solver. Returns vector in the same shape as time_domain
        """
        backward_solution = odeint(backward_dot, terminal_condition, time_domain[::-1], args = args)
        return backward_solution[::-1].T[0]

    def _forward_odeint(self, forward_dot, initial_condition, time_domain, args = ()):
        """Forward ode solver. Returns vector in the same shape as time_domain
        """
        forward_solution = odeint(forward_dot, initial_condition, time_domain, args = args)
        return forward_solution.T[0]

    def _create_time_function(self, vector):
        return interp1d(self.time_domain, vector, bounds_error=False, fill_value="extrapolate")

    '''
    Differential functions
    '''

    def _forward_dot_k(self,k,t):
        """Differential for k
        """
        if k >= self.minimum_capital:
          dot_k = (self.bar_rfunc(t) - self.delta)*k + self.w - self.feedback_control( self.pfunc(t) )
        else:
          dot_k = 0
        return dot_k

    def _backward_dot_p(self, p, t):
        """Differential for p
        """
        dot_p = -( self.bar_rfunc(t) - self.delta)*p

        return dot_p


    """Integration functions"""

    def _integrate_p(self, kT):
        self.pT = self.costate_terminal_condition(kT)
        p = self._backward_odeint(self._backward_dot_p, self.pT, self.time_domain)
        self.pfunc = self._create_time_function(p)
        return p

    def _integrate_k(self, k0):
        k = self._forward_odeint(self._forward_dot_k, k0, self.time_domain)
        self.kfunc = self._create_time_function(k)
        return k

    """Methods for iteratively solving k and p"""

    def shooting_iteration(self):
        for i,k0  in enumerate(self.initial_population_sample):
          k0 = self.initial_population_sample[i]
          self.pfunc = self._create_time_function(self.p_paths[i])
          self.k_paths[i] = self._integrate_k(k0)
          kT = self.k_paths[i][-1]
          self.p_paths[i] = self._integrate_p( kT)

    def update_bar_rfunc(self):
      self.hat_k = 0
      for k in self.k_paths:
        self.hat_k = self.hat_k + k / self.number_of_samples

      self.bar_r = self.mean_field_interest_rate(self.hat_k)
      self.bar_rfunc = self._create_time_function(self.bar_r)

    def monte_carlo_shooting(self, number_of_iterations, initial_p):


      self.bar_rfunc = lambda t: self.initial_population_sample.mean()

      self.k_paths = [sample for sample in self.initial_population_sample]
      self.p_paths = [initial_p for _ in self.initial_population_sample]

      for i in range(number_of_iterations):
        print(f'loop { i+ 1}')
        self.shooting_iteration()
        self.update_bar_rfunc()