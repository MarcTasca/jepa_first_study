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
    lissajous_a: int = 3
    lissajous_b: int = 5
    # Visual datasets
    image_size: int = 64
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
    ema_start: float = 0.99
    device: str = "auto"
    # Dataloader optimization
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    # Temporal masking (I-JEPA style)
    use_temporal_masking: bool = False
    mask_ratio: float = 0.4
    min_context_frames: int = 3


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

        # Get defaults from the dataclasses
        default_ds = DatasetConfig()
        default_tr = TrainingConfig()

        # Dataset
        parser.add_argument(
            "--mode",
            type=str,
            default=default_ds.name,
            help=f"Dataset mode: circle, spiral, lissajous, pendulum (default: {default_ds.name})",
        )
        parser.add_argument(
            "--size",
            type=int,
            default=default_ds.size,
            help=f"Dataset size (default: {default_ds.size})",
        )
        parser.add_argument(
            "--history_length",
            type=int,
            default=default_ds.history_length,
            help=f"Number of history frames (default: {default_ds.history_length})",
        )
        parser.add_argument(
            "--image_size",
            type=int,
            default=default_ds.image_size,
            help=f"Image size for visual datasets (default: {default_ds.image_size})",
        )

        # Training
        parser.add_argument(
            "--batch_size",
            type=int,
            default=default_tr.batch_size,
            help=f"Batch size (default: {default_tr.batch_size})",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=default_tr.jepa_epochs,
            help=f"Epochs for JEPA training (and Decoder) (default: {default_tr.jepa_epochs})",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=default_tr.lr,
            help=f"Learning rate (default: {default_tr.lr})",
        )
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument(
            "--workers",
            type=int,
            default=default_tr.num_workers,
            help=f"Num workers (default: {default_tr.num_workers})",
        )

        # Visualization
        parser.add_argument("--vis_points", type=int, default=1000, help="Number of points for visualization")

        args = parser.parse_args()

        ds_config = DatasetConfig(
            name=args.mode, size=args.size, history_length=args.history_length, image_size=args.image_size
        )

        # Use same epochs count for both phases for simplicity, or we could add separating flags
        tr_config = TrainingConfig(
            batch_size=args.batch_size,
            jepa_epochs=args.epochs,
            decoder_epochs=args.epochs,
            lr=args.lr,
            num_workers=args.workers,
        )

        return cls(dataset=ds_config, training=tr_config, seed=args.seed, num_vis_points=args.vis_points)
