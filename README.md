# JEPA for Dynamics Forecasting

A **Joint Embedding Predictive Architecture (JEPA)** implementation for learning and forecasting complex dynamics from both **vector** and **image** observations.

![JEPA Architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/JEPA_overview_diagram.svg/500px-JEPA_overview_diagram.svg.png)

## Features

- **Vector Mode**: Learn dynamics from coordinate trajectories (circle, spiral, lissajous, double pendulum)
- **Image Mode**: Learn dynamics from rendered pixel frames (`pendulum_image`)
- **Variance Regularization**: Prevents representation collapse (VICReg-inspired)
- **EMA Target Encoder**: Coupled with learning rate for stable training
- **Autoregressive Forecasting**: Visualize model predictions vs ground truth physics
- **Dataset Caching**: Pre-compute and cache rendered images for fast iteration

## Installation

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install torch numpy matplotlib imageio pillow tqdm
```

## Quick Start

### Vector Mode (Double Pendulum)
```bash
uv run python run.py --mode pendulum --epochs 100 --size 100000
```

### Image Mode (Pixel Pendulum)
```bash
uv run python run.py --mode pendulum_image --size 10000 --image_size 64 --epochs 50
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `pendulum` | Dataset: `circle`, `spiral`, `lissajous`, `pendulum`, `pendulum_image` |
| `--size` | `100000` | Number of trajectory sequences |
| `--image_size` | `64` | Image resolution for `pendulum_image` mode |
| `--epochs` | `100` | Training epochs (applies to both JEPA and Decoder phases) |
| `--batch_size` | `64` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--history_length` | `2` | Number of frames in context window |
| `--workers` | `4` | DataLoader workers |
| `--vis_points` | `1000` | Frames in forecast visualization |

## Architecture

### Models (`src/models.py`)

| Model | Input | Output | Description |
|-------|-------|--------|-------------|
| `Encoder` | Vector `(B, D)` | Latent `(B, 32)` | 3-layer MLP encoder |
| `Predictor` | Latent `(B, 32)` | Latent `(B, 32)` | Predicts next-step embedding |
| `Decoder` | Latent `(B, 32)` | Vector `(B, D)` | Reconstructs observations |
| `VisionEncoder` | Image `(B, C, H, W)` | Latent `(B, 32)` | 3-layer CNN encoder |
| `VisionDecoder` | Latent `(B, 32)` | Image `(B, C, H, W)` | Transposed CNN decoder |

### Training Pipeline (`src/trainer.py`)

**Phase 1: JEPA Training (Self-Supervised)**
```
Context[t:t+H] → Encoder → z_t → Predictor^gap → z_pred
Target[t+gap:t+gap+H] → Target Encoder (EMA) → z_target
Loss = MSE(z_pred, z_target) + λ·Variance_Regularization
```

**Phase 2: Decoder Training (Verification)**
```
Context → Encoder (frozen) → z → Decoder → Reconstruction
Loss = MSE(Reconstruction, Context)
```

### Datasets (`src/dataset.py`)

| Dataset | Type | Description |
|---------|------|-------------|
| `TrajectoryDataset` | Vector | Circular motion |
| `SpiralTrajectoryDataset` | Vector | Expanding spiral |
| `LissajousTrajectoryDataset` | Vector | Lissajous curves |
| `DoublePendulumDataset` | Vector | Chaotic double pendulum (RK4 physics) |
| `PixelPendulumDataset` | Image | Rendered double pendulum frames |

## Outputs

Each run creates a timestamped folder in `results/` containing:

```
results/pendulum_image_20260102_173937/
├── models/
│   ├── encoder.pth
│   ├── predictor.pth
│   └── decoder.pth
├── reconstruction.png    # Original vs Decoded comparison
├── forecast.mp4          # Ground Truth vs JEPA prediction video
└── run.log               # Training logs
```

## Key Implementation Details

### Variance Regularization (Anti-Collapse)
Without regularization, the encoder can collapse to outputting constant embeddings. We add a VICReg-inspired variance loss:
```python
var_loss = mean(relu(1.0 - var(z_pred))) + mean(relu(1.0 - var(z_target)))
loss = mse_loss + 1.0 * var_loss
```

### Dynamic EMA-LR Coupling
The target encoder's EMA decay increases as learning rate decreases:
```python
ema_decay = 1.0 - (1.0 - base_ema) * (current_lr / initial_lr)
```

### Image Precomputation
For `pendulum_image` mode, images are pre-rendered and cached to disk for fast reloading:
```
[PixelPendulumDataset] Pre-computing 10000 sequences of 64x64 images into RAM...
[DatasetFactory] Saving dataset to cache: data/pendulum_image/<hash>/dataset.pt
```

## Development

### Testing
```bash
uv run python -m pytest tests/
```

### Linting
```bash
uv run ruff check .
uv run ruff format .
```

### Pre-commit Hooks
```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Project Structure

```
jepa/
├── run.py                 # Entry point
├── src/
│   ├── config.py          # CLI args & dataclasses
│   ├── dataset.py         # Datasets & DatasetFactory
│   ├── models.py          # Encoder, Predictor, Decoder (MLP & CNN)
│   ├── runner.py          # Experiment orchestration
│   ├── trainer.py         # JEPA & Decoder training loops
│   ├── utils.py           # Logging utilities
│   └── visualization.py   # Plots & animations
├── tests/                 # Unit tests
├── .github/workflows/     # CI/CD
└── pyproject.toml         # Dependencies
```

## Citation

Based on the JEPA architecture from:
- [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- [VICReg: Variance-Invariance-Covariance Regularization](https://arxiv.org/abs/2105.04906)
