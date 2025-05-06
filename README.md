# Mean Field Tools

A Python library providing tools for numerical solutions of mean field problems, focusing on stochastic differential equations, forward-backward SDEs, and mean field games.

## Features

- Deep BSDE (Backward Stochastic Differential Equations) solvers
- Forward-Backward SDE implementations
- Mean field flow measure approximations
- Economic model implementations
- Human capital models
- Linear-quadratic solvers
- Finite difference methods for Fokker-Planck equations

## Installation

```bash
# Clone the repository
git clone https://github.com/fjpAntunes/mean-field-tools.git
cd mean-field-tools

# Install with Poetry
poetry install
```

## Usage

Check the `experiments` directory and individual module README files for examples and usage instructions.

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

- `mean_field_tools/deep_bsde/`: Core implementations for deep BSDE methods
- `mean_field_tools/economic_model/`: Economic modeling tools
- `mean_field_tools/finite_differences/`: Finite difference methods
- `mean_field_tools/human_capital/`: Human capital modeling
- `mean_field_tools/linear_quadratic/`: Linear-quadratic problem solvers

## License

MIT
