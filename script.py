from human_capital import HumanCapitalEconomicModelMFG
import numpy as np

from human_capital import HumanCapitalEconomicModelMFG
import numpy as np


arg_dict = {'delta': 0.05,'xi':0}


time_domain =  np.linspace(0, 1, 101)

num_samples = 50
initial_capital_sample = np.power(np.random.power(10, num_samples), -1) - 1
initial_education_sample = np.power(np.random.power(5, num_samples), -1) - 1

def feedback_consumption(p):
    c0 = 0
    return np.power(p,-1)#c0 - p

def feedback_work_effort(h,p,q,bar_w, xi):
    value = bar_w * h * p - np.power(h, xi) * q
    return np.maximum(np.minimum(value,1),0)

alpha = 0.5
beta = 1 - alpha
C = 0.5
tolerance = 1e-3

def mean_field_interest_rate(k, h):
  safe_k = np.maximum(k, tolerance)
  safe_h = np.maximum(h, tolerance)
  return C * alpha * np.power(safe_k, alpha - 1) * np.power(safe_h, beta)

def mean_field_wage_rate(k, h):
    safe_k = np.maximum(k, tolerance)
    safe_h = np.maximum(h, tolerance)
    return C * beta * np.power(safe_k, alpha) * np.power(safe_h, beta-1)

def education_efficiency(effort):
   return effort

def capital_costate_terminal_condition(k):
  return 1


model = HumanCapitalEconomicModelMFG(
    time_domain,
    initial_capital_sample,
    initial_education_sample,
    feedback_consumption,
    feedback_work_effort,
    mean_field_interest_rate,
    mean_field_wage_rate,
    education_efficiency,
    capital_costate_terminal_condition,
    damping_function = lambda k: 1/(k + 1),
    **arg_dict)

model.monte_carlo_shooting(number_of_iterations=10)

if __name__ == '__main__':
   model.monte_carlo_shooting(number_of_iterations=10)