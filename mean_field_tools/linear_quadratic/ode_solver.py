from scipy.integrate import odeint
from scipy.interpolate import interp1d
import numpy as np


class LinearQuadraticMFG:

    def __init__(self, time_domain, **kwargs):

        self.time_domain = time_domain
        self.arg_list = [
            "A",
            "barA",
            "B",
            "C",
            "Q",
            "barQ",
            "S",
            "QT",
            "barQT",
            "ST",
            "x0",
            "sigma0",
            "nu",
        ]
        (
            self.A,
            self.barA,
            self.B,
            self.C,
            self.Q,
            self.barQ,
            self.S,
            self.QT,
            self.barQT,
            self.ST,
            self.x0,
            self.sigma0,
            self.nu,
        ) = (kwargs[name] for name in self.arg_list)

        self.z0 = self.x0

    """
    Utility functions
    """

    def _backward_odeint(self, backward_dot, terminal_condition, time_domain, args=()):
        """backward ode solver. Returns vector in the same shape as time_domain"""
        backward_solution = odeint(
            backward_dot, terminal_condition, time_domain[::-1], args=args
        )
        return backward_solution[::-1].T[0]

    def _forward_odeint(self, forward_dot, initial_condition, time_domain, args=()):
        """Forward ode solver. Returns vector in the same shape as time_domain"""
        forward_solution = odeint(
            forward_dot, initial_condition, time_domain, args=args
        )
        return forward_solution.T[0]

    def _create_time_function(self, vector):
        return interp1d(
            self.time_domain, vector, bounds_error=False, fill_value="extrapolate"
        )

    """
    Differential functions
    """

    def _forward_dot_z(self, z, t):
        """Differential for z"""
        dot_z = (self.A + self.barA - (self.B**2 / self.C) * self.pfunc(t)) * z - (
            self.B**2 / self.C
        ) * self.rfunc(t)
        return dot_z

    def _backward_dot_p(self, p, t):
        """Differential for p"""
        dot_p = -(2 * self.A * p - (self.B**2 / self.C) * p**2 + self.Q + self.barQ)

        return dot_p

    def _backward_dot_r(self, r, t):
        """Differential for r"""
        dot_r = -(
            (self.A - (self.B**2 / self.C) * self.pfunc(t)) * r
            + (self.pfunc(t) * self.barA - self.barQ * self.S) * self.zfunc(t)
        )
        return dot_r[0]

    def _backward_dot_s(self, s, t):
        """Differential for s."""
        dot_s = -(
            self.nu * self.pfunc(t)
            - 0.5 * (self.B**2 / self.C) * self.rfunc(t)
            + self.rfunc(t) * self.barA * self.zfunc(t)
            + 0.5 * (self.S**2 * self.barQ) * self.zfunc(t) ** 2
        )

        return dot_s

    """Integration functions"""

    def _integrate_p(self):
        self.pT = self.QT + self.barQT
        p = self._backward_odeint(self._backward_dot_p, self.pT, self.time_domain)
        self.pfunc = self._create_time_function(p)
        return p

    def _integrate_r(self):
        self.rT = -self.barQT * self.ST * self.z[-1]
        r = self._backward_odeint(self._backward_dot_r, self.rT, self.time_domain)
        self.rfunc = self._create_time_function(r)
        return r

    def _integrate_z(self):
        self.z0 = self.x0
        z = self._forward_odeint(self._forward_dot_z, self.z0, self.time_domain)
        self.zfunc = self._create_time_function(z)
        return z

    def _integrate_s(self):
        self.sT = 0.5 * self.barQT * self.ST**2 * self.z[-1] ** 2
        s = self._backward_odeint(self._backward_dot_s, self.sT, self.time_domain)
        self.sfunc = self._create_time_function(s)
        return s

    """Methods for iteratively solving r and z"""

    def _abstract_iterations(self, damping_func, num_iterations):
        for i in range(num_iterations):
            self.r = self._integrate_r()
            update_z = self._integrate_z()
            self.z = damping_func(i) * self.z + (1 - damping_func(i)) * update_z
        return (self.z, self.r)

    def _fictitious_play(self, num_iterations):
        damping_func = lambda i: i / (i + 1)
        result = self._abstract_iterations(*(damping_func, num_iterations))
        return result

    def _picard_iterations(self, damping, num_iterations):
        damping_func = lambda i: damping
        result = self._abstract_iterations(*(damping_func, num_iterations))
        return result

    def solve_nash_equilibrium(self, initial_z, initial_r, **kwargs):
        self.p = self._integrate_p()
        self.z, self.r = (initial_z, initial_r)
        self.zfunc, self.rfunc = (
            self._create_time_function(v) for v in [self.z, self.r]
        )
        self.z, self.r = self._picard_iterations(**kwargs)
        self.s = self._integrate_s()
        return (self.z, self.p, self.r, self.s)

    """Descriptive methods"""

    def representative_agent_optimal_control(self, t, x):
        return -self.B * (self.pfunc(t) * x + self.rfunc(t)) / self.C

    def representative_agent_optimal_cost(self):
        cost = (
            0.5 * self.pfunc(0) * (self.sigma0**2 + self.x0**2)
            + self.rfunc(0) * self.x0
            + self.sfunc(0)
        )
        return cost

    def representative_agent_value_function(self, t, x):
        value = 0.5 * self.pfunc(t) * x**2 + self.rfunc(t) * x + self.sfunc(t)
        return value
