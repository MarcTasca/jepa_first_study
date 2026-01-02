# JEPA for Dynamics Forecasting

A Joint Embedding Predictive Architecture (JEPA) implementation for forecasting complex dynamics (e.g., Chaos, Lissajous curves).

## Installation

### Prerequisites
- Python 3.8+
- PyTorch

### Setup
Install dependencies:
```bash
pip install torch numpy matplotlib imageio
```
Or using `uv`:
```bash
uv sync
```

## Usage

Run the main training and visualization loop:
```bash
python run.py --mode pendulum
```

### Command Line Arguments
- `--mode`: Dataset to use (`pendulum`, `circle`, `spiral`, `lissajous`). Default: `pendulum`.
- `--epochs`: Number of epochs for training. Default: `100`.
- `--batch_size`: Batch size. Default: `64`.
- `--lr`: Learning rate. Default: `1e-3`.
- `--seed`: Random seed. Default: `42`.
- `--vis_points`: Number of points for visualization. Default: `1000`.

Example:
```bash
python run.py --mode lissajous --epochs 50 --batch_size 32
```

## Project Structure

```
jepa/
├── .github/          # CI/CD Workflows
├── forecasting/      # Analysis & Comparison scripts
├── run.py            # Entry point
├── src/
│   ├── config.py     # Configuration dataclasses
│   ├── dataset.py    # Datasets and Factory
│   ├── models.py     # Neural Network Modules
│   ├── runner.py     # Experiment Runner
│   ├── trainer.py    # Training Logic
│   ├── utils.py      # Utilities (Logging)
│   └── visualization.py # Plotting & Animation
├── tests/            # Unit Tests

```

## Components

### Configuration
Configuration is handled via `src/config.py`. It uses dataclasses to define default hyperparameters for datasets, models, and training.

### Datasets
`src/dataset.py` contains:
- `DoublePendulumDataset`: Chaotic dynamics via RK4 integration.
- `TrajectoryDataset`: Simple circular motion.
- `SpiralTrajectoryDataset`: Non-linear manifold.
- `LissajousTrajectoryDataset`: Expanding Lissajous curves.

### Models
`src/models.py` defines:
- `Encoder`: Maps obervations to latent space.
- `Predictor`: Predicts future latent states.
- `Decoder`: Maps latent states back to observations (verification only).

### Trainer
`src/trainer.py` implements the JEPA self-supervised training loop and a separate decoder training loop for verification.

## Development

We follow standard engineering practices to ensure code quality and reproducibility.

### Dependency Management
This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.
```bash
uv sync                 # Install dependencies from lock file
uv add <package_name>   # Add a new dependency
```

### Testing
Unit tests are located in `tests/` and cover models, datasets, and configuration. We use `pytest`.
```bash
uv run python -m pytest tests/
```

### Linting & Formatting
We use `ruff` for linting and formatting, configured in `pyproject.toml`.
```bash
uv run ruff check .     # Check for lint errors
uv run ruff format .    # Fix formatting issues
```

### Pre-commit Hooks
We use `pre-commit` to ensure code quality before every commit.
```bash
# Install hooks (run once)
uv run pre-commit install

# Run manually (optional)
uv run pre-commit run --all-files
```

### CI/CD
A GitHub Actions workflow (`.github/workflows/ci.yml`) is configured to run tests and lint checks automatically on every `push` and `pull_request` to the `main` branch.
