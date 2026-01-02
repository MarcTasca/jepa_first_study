import argparse
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    name: str = "pendulum"
    size: int = 100000
    history_length: int = 2
    dt: float = 0.05
    sequence_length: int = 30
    # Specific to some datasets
    spiral_loops: int = 3
    lissajous_a: int = 3
    lissajous_b: int = 5
    # Caching
    cache_dir: str = "data"
    use_cache: bool = True
    regenerate: bool = False


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    embedding_dim: int = 32
    # Input/Output dims are derived from dataset at runtime, but can be overridden if needed
    input_dim: int = 0
    output_dim: int = 0


@dataclass
class TrainingConfig:
    batch_size: int = 64
    jepa_epochs: int = 100
    decoder_epochs: int = 100
    lr: float = 1e-3
    device: str = "auto"


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    seed: int = 42
    results_dir: str = "results"
    models_dir: str = "models"
    num_vis_points: int = 1000

    @classmethod
    def from_args(cls) -> "ExperimentConfig":
        parser = argparse.ArgumentParser(description="JEPA Experiment Runner")

        # Dataset
        parser.add_argument(
            "--mode", type=str, default="pendulum", help="Dataset mode: circle, spiral, lissajous, pendulum"
        )
        parser.add_argument("--size", type=int, default=100000, help="Dataset size")
        parser.add_argument("--history_length", type=int, default=2, help="Number of history frames")

        # Training
        parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
        parser.add_argument("--epochs", type=int, default=100, help="Epochs for JEPA training (and Decoder)")
        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")

        # Visualization
        parser.add_argument("--vis_points", type=int, default=1000, help="Number of points for visualization")

        args = parser.parse_args()

        # Map args to config structure
        ds_config = DatasetConfig(name=args.mode, size=args.size, history_length=args.history_length)

        # Use same epochs count for both phases for simplicity, or we could add separating flags
        tr_config = TrainingConfig(
            batch_size=args.batch_size, jepa_epochs=args.epochs, decoder_epochs=args.epochs, lr=args.lr
        )

        return cls(dataset=ds_config, training=tr_config, seed=args.seed, num_vis_points=args.vis_points)
