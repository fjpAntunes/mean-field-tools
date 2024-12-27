Mean Field Tools: Deep Backward Stochastic Differential Equation (BSDE) Solver
Project Overview
This project provides a sophisticated computational framework for solving forward-backward stochastic differential equations (FBSDEs) with advanced numerical methods, particularly focusing on mean-field interaction scenarios and systemic risk modeling.
Key Components
1. Filtration Module (filtration.py)

Manages stochastic process state tracking
Generates Brownian motion and increments
Supports common noise filtration
Tracks time-dependent system states

2. Forward-Backward SDE Module (forward_backward_sde.py)

Implements numerical solution strategies for SDEs
Supports Forward and Backward SDE classes
Uses Picard iterations for numerical convergence
Handles complex stochastic process interactions

3. Function Approximator (function_approximator.py)

Neural network-based function approximation
Implements stochastic gradient descent training
Enables flexible function representation

4. Measure Flow Module (measure_flow.py)

Calculates conditional mean and flow of stochastic processes
Supports common noise measure flow
Parametrizes process mean fields

5. Systemic Risk Example (systemic_risk_common_noise.py)

Demonstrates practical application of the framework
Models a specific systemic risk scenario with common noise
Provides analytical and numerical solution comparisons

Methodology

Picard Iterations

Iteratively refine forward and backward processes
Uses neural network approximation
Converges to solution through damped updates


Common Noise Modeling

Introduces correlated noise components
Enables modeling of systemic interactions
Separates common and idiosyncratic noise sources


Stochastic Differential Equation Solving

Numerical approximation of SDEs
Supports complex drift and volatility functions
Handles multi-dimensional processes



Key Features

Flexible stochastic process modeling
Neural network-based function approximation
Supports multiple noise generation strategies
Comprehensive plotting and visualization tools

Use Cases

Financial risk modeling
Systemic risk assessment
Complex stochastic process simulation
Machine learning-enhanced SDE solving

Dependencies

PyTorch
NumPy
Matplotlib
tqdm

Usage Example
The systemic_risk_common_noise.py script provides a comprehensive example of setting up and solving a mean-field BSDE with common noise, demonstrating the framework's capabilities.
Note
This framework is particularly powerful for modeling complex stochastic systems with intricate interactions and mean-field effects.