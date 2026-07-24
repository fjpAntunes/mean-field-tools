# Mean Field Tools

A Python library for numerically solving mean field problems using the Deep BSDE method, focusing on forward-backward stochastic differential equations (FBSDEs) and mean field games.

## Features

- Deep BSDE (Backward Stochastic Differential Equation) solvers
- Forward-Backward SDE implementations with Picard iteration
- Neural-network function approximators (including ResNet architectures)
- Mean field flow measure approximations (with common-noise support)
- Filtration tools for Brownian motion and stochastic state tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/fjpAntunes/mean-field-tools.git
cd mean-field-tools

# Install with Poetry
poetry install
```

## Usage

See `mean_field_tools/deep_bsde/README.md` for a detailed overview of the components, and `mean_field_tools/deep_bsde/script/experiments/` for runnable examples (systemic risk, portfolio hedging, economic growth, and more).

## Testing

The project uses pytest for testing. Tests are organized into unit and integration tests within the `mean_field_tools/deep_bsde/test/` directory.

```bash
# Run all tests
pytest

# Run specific test categories
pytest mean_field_tools/deep_bsde/test/unit/
pytest mean_field_tools/deep_bsde/test/integration/

# Run a specific test file
pytest mean_field_tools/deep_bsde/test/unit/test_function_approximator.py
```

## Structure

- `mean_field_tools/deep_bsde/`: Core library — the Deep BSDE solver and its components
  - `filtration.py`: Brownian motion generation and stochastic state tracking
  - `forward_backward_sde.py`: Forward/Backward SDE classes and Picard iteration
  - `function_approximator.py`: Neural-network approximators
  - `measure_flow.py`: Mean field flow measure approximations
  - `artist.py`: Plotting and diagnostics
  - `script/experiments/`: Example applications
  - `test/`: Unit and integration tests

## License

MIT
