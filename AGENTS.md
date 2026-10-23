# AutoEmulate — Agent & Contributor Guide

This file provides context for AI coding agents (GitHub Copilot, Claude, Codex, etc.) and new contributors working on this codebase.

## Repository Overview

AutoEmulate is a Python package that automatically fits and compares emulators (surrogate models) to replace slow simulations.

**Key concepts:**
- **Emulator** — a fast, data-driven proxy for a slow simulation
- **AutoEmulate core** — fits multiple emulators, cross-validates, and surfaces the best one
- **Calibration** — infers simulation parameters from data
- **Sensitivity analysis** — finds which inputs drive simulation outputs
- **Active learning** — chooses which simulations to run next

## Build & Environment

This project uses [`uv`](https://docs.astral.sh/uv/) as the package manager.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install all dependencies (including dev extras)
uv sync --extra dev

# Activate the environment
source .venv/bin/activate
```

## Running Tests

```bash
# Run the full test suite
uv run pytest

# Run a specific test file
uv run pytest tests/core/

# Run with coverage
uv run pytest --cov=autoemulate

# Apple Silicon (M-series) — some GP tests require MPS fallback to CPU
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run pytest -q
```

## Code Style & Linting

```bash
# Run pre-commit hooks (ruff, pyright, nbstripout)
uv run pre-commit run --all-files
```

Pre-commit is enforced in CI on every PR — run it locally before pushing to catch failures early.

## Repository Structure

```
autoemulate/
├── autoemulate/
│   ├── core/           # AutoEmulate main class and cross-validation logic
│   ├── emulators/      # Individual emulator implementations (GP, SVM, RBF, etc.)
│   ├── calibration/    # Calibration methods
│   ├── data/           # Data handling and splitting utilities
│   ├── transforms/     # Input/output transforms (normalisation, PCA, etc.)
│   ├── learners/       # Active learning strategies
│   ├── datasets/       # Built-in simulation datasets for testing/demos
│   ├── simulations/    # Simple simulation wrappers
│   ├── callbacks/      # Training/fitting callbacks
│   └── feature_generation/ # Feature engineering utilities
├── tests/              # pytest test suite, mirrors autoemulate/ structure
├── docs/               # Sphinx + MyST documentation
└── case_studies/       # Jupyter notebooks for real-world use cases
```

## Adding a New Emulator

See the [adding emulators guide](https://alan-turing-institute.github.io/autoemulate/tutorials/advanced/01_add_emulators.html). In short:

1. Create a new file in `autoemulate/emulators/`
2. Subclass `Emulator` (or one of its subclasses, e.g. `DeterministicEmulator`/`ProbabilisticEmulator`, in `autoemulate/emulators/base.py`) and implement the required methods (`__init__`, `_fit`, `_predict`, etc.)
3. Register it via the `Registry`/`register` mechanism in `autoemulate/emulators/registry.py`
4. Add tests in `tests/emulators/`

## Docstring Style

Use **NumPy-style docstrings** throughout:

```python
def fit(self, X: np.ndarray, y: np.ndarray) -> "MyEmulator":
    """Fit the emulator to training data.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input simulation parameters.
    y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Simulation outputs.

    Returns
    -------
    self : MyEmulator
        Fitted estimator.
    """
```

## Pull Request Checklist

- [ ] Tests pass: `uv run pytest -q`
- [ ] Pre-commit hooks pass: `uv run pre-commit run --all-files`
- [ ] New public functions have NumPy docstrings
- [ ] If adding an emulator: registered via the `Registry`/`register` mechanism and tested

## Contact

Questions? Reach the team at [ai4physics@turing.ac.uk](mailto:ai4physics@turing.ac.uk) or open a GitHub issue.
