# JEPA for Dynamics Forecasting

A **Joint Embedding Predictive Architecture (JEPA)** implementation for learning dynamics from trajectories and images.

## Quick Start

```bash
# Install
uv sync

# Run (vector mode - double pendulum coordinates)
uv run python run.py --mode pendulum --epochs 100

# Run (image mode - rendered pendulum frames)
uv run python run.py --mode pendulum_image --size 10000 --image_size 32 --epochs 50 --batch_size 256
```

## Modes

| Mode | Input | Description |
|------|-------|-------------|
| `circle` | Coordinates | Point on circle |
| `spiral` | Coordinates | Expanding spiral |
| `lissajous` | Coordinates | Lissajous curves |
| `pendulum` | Coordinates | Double pendulum (chaotic) |
| `pendulum_image` | RGB Images | Rendered pendulum frames |

## Architecture

```
Context[t:t+H] ──► Encoder ──► z_t ──► Predictor ──► z_pred
                                                       │
Target[t+gap:t+gap+H] ──► Target Encoder (EMA) ──► z_target
                                                       │
                              Loss = MSE(z_pred, z_target)
```

**Models:**
- `Encoder` / `VisionEncoder`: Maps observations to 32-dim latent space
- `Predictor`: Predicts next latent state
- `Decoder` / `VisionDecoder`: Reconstructs observations (verification only)

## Command Line

```bash
uv run python run.py \
  --mode pendulum_image \
  --size 10000 \
  --image_size 64 \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.001 \
  --history_length 2 \
  --workers 0
```

## Outputs

Each run creates:
```
results/<mode>_<timestamp>/
├── models/           # Saved weights
├── reconstruction.png  # Original vs decoded
├── forecast.mp4      # GT vs prediction video
└── run.log
```

## Known Behavior: Representation Collapse

With pure MSE loss, JEPA tends to collapse:
- Encoder outputs constant embeddings
- Loss → 0 (trivial solution: `z = 0` for all inputs)
- Decoder can't reconstruct (loss stays ~0.02)

This is a fundamental challenge in self-supervised learning. Solutions include:
- Contrastive loss (needs negative samples)
- VICReg (variance + invariance + covariance)
- Asymmetric architecture (BYOL/SimSiam)

## Development

```bash
uv run python -m pytest tests/    # Tests
uv run ruff check .               # Lint
uv run pre-commit run --all-files # Pre-commit
```

## Project Structure

```
src/
├── config.py      # CLI args & dataclasses
├── dataset.py     # TrajectoryDataset, PixelPendulumDataset, etc.
├── models.py      # Encoder, Predictor, Decoder (MLP & CNN)
├── runner.py      # Experiment orchestration
├── trainer.py     # JEPA training loop
└── visualization.py
```
