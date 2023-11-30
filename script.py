from human_capital import HumanCapitalEconomicModelMFG
import numpy as np

arg_dict = {'delta': 2,'xi':1}


time_domain =  np.linspace(0, 1, 101)

initial_capital_sample = np.power(np.random.power(5, 50), -1) - 1
initial_education_sample = np.power(np.random.power(5, 50), -1) - 1

def feedback_consumption(p):
  c_0 = 0
  return c_0 - p

def feedback_work_effort(p,q,bar_w):
    threshold = bar_w*p - q
    return 1*(threshold > 0)

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
  return -0.05*k


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
    **arg_dict)

if __name__ == '__main__':
   model.monte_carlo_shooting(number_of_iterations=10)