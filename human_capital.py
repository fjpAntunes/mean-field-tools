from collections.abc import Callable
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import numpy as np


class HumanCapitalEconomicModelMFG():

    def __init__(self,
                 time_domain,
                 initial_capital_sample,
                 initial_education_sample,
                 feedback_consumption : Callable[[float], float],
                 feedback_work_effort :  Callable[[float, float], float],
                 mean_field_interest_rate : Callable[[float, float], float],
                 mean_field_wage_rate : Callable[[float, float], float],
                 education_efficiency: Callable[[float], float],
                 capital_costate_terminal_condition : Callable[[float], float],
                 minimum_capital = -np.inf,
                 **kwargs):

        self.time_domain = time_domain
        self.initial_capital_sample =initial_capital_sample
        self.initial_education_sample =initial_education_sample

        self.feedback_consumption = feedback_consumption
        self.feedback_work_effort = feedback_work_effort

        self.mean_field_interest_rate = mean_field_interest_rate
        self.mean_field_wage_rate = mean_field_wage_rate

        self.education_efficiency = education_efficiency
        self.capital_costate_terminal_condition = capital_costate_terminal_condition
        self.minimum_capital = minimum_capital

        self.arg_list = ['delta','xi']
        self.delta, self.xi  =  (kwargs[name] for name in self.arg_list)

        self.number_of_samples = len(self.initial_capital_sample)


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
        capital_gains = (self.bar_r_func(t) - self.delta)*k
        work_effort = self.feedback_work_effort(self.p_func(t), self.q_func(t), self.bar_w_func(t))
        wage = self.bar_w_func(t)*self.h_func(t)*work_effort
        consumption = self.feedback_consumption(self.p_func(t))

        if k >= self.minimum_capital:
          dot_k = capital_gains + wage - consumption
        else:
          dot_k = 0
        return dot_k

    def _forward_dot_h(self,h,t):
        work_effort = self.feedback_work_effort(self.p_func(t), self.q_func(t), self.bar_w_func(t))
        dot_h = np.power(h,self.xi)*self.education_efficiency(1 - work_effort)

        return dot_h

    def _backward_dot_p(self, p, t):
        """Differential for p
        """
        dot_p = -( self.bar_r_func(t) - self.delta)*p

        return dot_p

    def _backward_dot_q(self, q, t):
        work_effort = self.feedback_work_effort(self.p_func(t), self.q_func(t), self.bar_w_func(t))
        p_term = self.bar_w_func(t)*work_effort
        q_term = self.xi * np.power(self.h_func(t), self.xi - 1) * self.education_efficiency( 1 - work_effort)

        return - (p_term*self.p_func(t) + q_term*q)

    """Integration functions"""

    def _integrate_p(self, kT):
        pT = self.capital_costate_terminal_condition(kT)
        p = self._backward_odeint(self._backward_dot_p, pT, self.time_domain)
        self.p_func = self._create_time_function(p)
        return p

    def _integrate_q(self, qT):
        qT = 0
        q = self._backward_odeint(self._backward_dot_q, qT, self.time_domain)
        self.q_func = self._create_time_function(q)
        return q

    def _integrate_k(self, k0):
        k = self._forward_odeint(self._forward_dot_k, k0, self.time_domain)
        self.k_func = self._create_time_function(k)
        return k

    def _integrate_h(self, h0):
        h = self._forward_odeint(self._forward_dot_h, h0, self.time_domain)
        self.h_func = self._create_time_function(h)
        return h

    """Methods for iteratively solving k and p"""
    #
    # STILL NEED TO UPDATE THIS SECTION
    #
    def shooting_forward_step(self, i):
        k0 = self.initial_capital_sample[i]
        h0 = self.initial_education_sample[i]
        self.p_func = self._create_time_function(self.p_paths[i])
        self.q_func = self._create_time_function(self.q_paths[i])
        self.h_func = self._create_time_function(self.h_paths[i])
        self.k_func = self._create_time_function(self.k_paths[i])
        
        self.k_paths[i] = self._integrate_k(k0)
        self.h_paths[i] = self._integrate_h(h0)

    def shooting_backwards_step(self,i):
        kT = self.k_paths[i][-1]
        self.k_func = self._create_time_function(self.k_paths[i])
        self.h_func = self._create_time_function(self.h_paths[i]) 
        self.p_paths[i] = self._integrate_p( kT)
        self.q_paths[i] = self._integrate_q(0)


    def shooting_iteration(self):
        for i  in range(self.number_of_samples):
            self.shooting_forward_step(i)
            self.shooting_backwards_step(i)


    def update_mean_field_functions(self):
        self.hat_k = 0
        for k in self.k_paths:
            self.hat_k = self.hat_k + k / self.number_of_samples
        
        self.hat_h = 0
        for i in range(self.number_of_samples):
            p = self.p_paths[i]
            q = self.q_paths[i]
            h = self.h_paths[i]
            effective_h = self.feedback_work_effort(p,q, self.bar_w) * h
            self.hat_h = self.hat_h + effective_h / self.number_of_samples
        
        self.bar_r = self.mean_field_interest_rate(self.hat_k, self.hat_h)
        self.bar_r_func = self._create_time_function(self.bar_r)
        
        self.bar_w = self.mean_field_wage_rate(self.hat_k, self.hat_h)
        self.bar_w_func = self._create_time_function(self.bar_w)

    def initialize_parameters(self, initial_p = [], initial_q = []):
        size = len(self.time_domain)
        if not initial_p:
            initial_p = np.ones(size)
        if not initial_q:
            initial_q = np.ones(size)

        initial_hat_k = self.initial_capital_sample.mean()
        initial_hat_h = self.initial_education_sample.mean()

        self.bar_r = np.ones(size) * self.mean_field_interest_rate(initial_hat_k, initial_hat_h)
        self.bar_w = np.ones(size) * self.mean_field_wage_rate(initial_hat_k,initial_hat_h)  
        self.bar_r_func = self._create_time_function(self.bar_r)
        self.bar_w_func = self._create_time_function(self.bar_w)

        self.k_paths = [sample*np.ones(size) for sample in self.initial_capital_sample]
        self.h_paths = [sample*np.ones(size) for sample in self.initial_education_sample]        
        self.p_paths = [initial_p for _ in self.initial_capital_sample]
        self.q_paths = [initial_q for _ in self.initial_capital_sample]
 
    def monte_carlo_shooting(self, number_of_iterations, initial_p = [], initial_q = []):
        self.initialize_parameters(initial_p=initial_p,initial_q=initial_q)
        for i in range(number_of_iterations):
          print(f'loop { i+ 1}')
          self.shooting_iteration()
          self.update_mean_field_functions()
