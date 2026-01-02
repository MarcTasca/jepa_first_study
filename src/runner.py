import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.dataset import DatasetFactory
from src.models import Decoder, Encoder, Predictor, VisionDecoder, VisionEncoder
from src.trainer import JEPATrainer
from src.utils import setup_logger
from src.visualization import visualize_forecast, visualize_latent_reconstruction


class Runner:
    def __init__(self, config: ExperimentConfig):
        self.cfg = config

        # Create unique run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.cfg.dataset.name}_{timestamp}"
        self.run_dir = os.path.join(self.cfg.results_dir, self.run_id)

        os.makedirs(self.run_dir, exist_ok=True)

        # Setup logger
        # We configure the root logger ("") so that JEPATrainer logging also works
        setup_logger(name="", log_file=f"{self.run_dir}/run.log")
        self.logger = logging.getLogger("Runner")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if self.cfg.training.device != "auto":
            self.device = torch.device(self.cfg.training.device)

        self.logger.info(f"Initialized Runner on device: {self.device}")
        self.logger.info(f"Run Directory: {self.run_dir}")

    def setup_seeds(self):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.logger.info(f"Random seed set to {self.cfg.seed}")

    def prepare_data(self):
        self.logger.info(f"Loading dataset: {self.cfg.dataset.name}")
        self.dataset = DatasetFactory.get_dataset(self.cfg.dataset)
        # MPS doesn't support pin_memory effectively, so we disable it to avoid warnings
        # and potential overhead.
        pin_memory = self.cfg.training.pin_memory
        if self.device.type == "mps":
            pin_memory = False
            self.logger.info("Disabling pin_memory for MPS device")

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=self.cfg.training.prefetch_factor if self.cfg.training.num_workers > 0 else None,
            persistent_workers=True if self.cfg.training.num_workers > 0 else False,
        )
        self.logger.info(f"Dataset loaded with {len(self.dataset)} samples")

    def initialize_models(self):
        # Determine input/output dims from dataset properties
        is_visual = self.cfg.dataset.name == "pendulum_image"

        if is_visual:
            channels = 3  # RGB
            # For vision, input is (B, History*C, H, W)
            input_channels = channels * self.cfg.dataset.history_length

            self.logger.info(f"Initializing Vision Models with In Channels: {input_channels}")

            self.encoder = VisionEncoder(input_channels=input_channels, embedding_dim=self.cfg.model.embedding_dim)
            # Predictor is always MLP on Latents
            self.predictor = Predictor(embedding_dim=self.cfg.model.embedding_dim, hidden_dim=self.cfg.model.hidden_dim)
            self.decoder = VisionDecoder(embedding_dim=self.cfg.model.embedding_dim, output_channels=input_channels)
        else:
            # Vector based datasets
            if self.cfg.dataset.name == "pendulum":
                channels = 4
            else:
                channels = 2

            input_dim = channels * self.cfg.dataset.history_length
            output_dim = channels * self.cfg.dataset.history_length

            self.logger.info(f"Initializing models with Input Dim: {input_dim}, Output Dim: {output_dim}")

            self.encoder = Encoder(
                input_dim=input_dim, hidden_dim=self.cfg.model.hidden_dim, embedding_dim=self.cfg.model.embedding_dim
            )
            self.predictor = Predictor(embedding_dim=self.cfg.model.embedding_dim, hidden_dim=self.cfg.model.hidden_dim)
            self.decoder = Decoder(
                embedding_dim=self.cfg.model.embedding_dim, output_dim=output_dim, hidden_dim=self.cfg.model.hidden_dim
            )

    def train(self):
        self.trainer = JEPATrainer(
            self.encoder,
            self.predictor,
            self.decoder,
            self.dataloader,
            lr=self.cfg.training.lr,
            ema_start=self.cfg.training.ema_start,
            device=self.device,
        )

        self.logger.info("Starting JEPA Training (Self-Supervised)")
        self.trainer.train_jepa(epochs=self.cfg.training.jepa_epochs)

        self.logger.info("Starting Decoder Training (Verification)")
        self.trainer.train_decoder(epochs=self.cfg.training.decoder_epochs)

        # Save models to run_dir/models
        models_dir = os.path.join(self.run_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(models_dir, "encoder.pth"))
        torch.save(self.predictor.state_dict(), os.path.join(models_dir, "predictor.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(models_dir, "decoder.pth"))
        self.logger.info(f"Models saved to {models_dir}")

    def visualize(self):
        self.logger.info("Generating Visualizations...")

        recon_path = os.path.join(self.run_dir, "reconstruction.png")
        visualize_latent_reconstruction(
            self.encoder,
            self.decoder,
            save_path=recon_path,
            mode=self.cfg.dataset.name,
            num_points=self.cfg.num_vis_points,
            history_length=self.cfg.dataset.history_length,
        )

        if self.cfg.dataset.name == "pendulum":
            forecast_path = os.path.join(self.run_dir, "forecast.mp4")
            visualize_forecast(
                self.encoder,
                self.predictor,
                self.decoder,
                save_path=forecast_path,
                num_points=self.cfg.num_vis_points,
                history_length=self.cfg.dataset.history_length,
            )

    def run(self):
        self.setup_seeds()
        self.prepare_data()
        self.initialize_models()
        self.train()
        self.visualize()
        self.logger.info(f"Experiment Completed. Results saved to {self.run_dir}")
