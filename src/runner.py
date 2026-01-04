import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.dataset import DatasetFactory
from src.models import Decoder, Encoder, Predictor, VisionDecoder, VisionEncoder
from src.models_vjepa import VJepaDecoder, VJepaPredictor, VJepaViT
from src.trainer import JEPATrainer
from src.trainer_vjepa import VJepaTrainer
from src.utils import setup_logger
from src.visualization import (
    visualize_forecast,
    visualize_image_forecast,
    visualize_image_reconstruction,
    visualize_latent_reconstruction,
    visualize_vjepa_forecast,
    visualize_vjepa_reconstruction,
)


class Runner:
    def __init__(self, config: ExperimentConfig, output_dir: str = None):
        self.cfg = config

        if output_dir:
            self.run_dir = output_dir
            # Assume logging is already configured by setup_experiment
        else:
            # Create unique run directory (Legacy mode)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{self.cfg.dataset.name}_{timestamp}"
            self.run_dir = os.path.join(self.cfg.results_dir, self.run_id)
            os.makedirs(self.run_dir, exist_ok=True)
            # Setup logger
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
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            self.logger.info(f"Sample Dimensions: {sample.shape}")

    def initialize_models(self):
        # Determine input/output dims from dataset properties
        is_visual = self.cfg.dataset.name == "pendulum_image"

        # Check Model Type
        model_type = getattr(self.cfg.model, "type", "jepa")

        if model_type == "vjepa":
            self.logger.info("Initializing V-JEPA Models (Transformer)")
            # Assume 30 frames from dataset config
            frames = self.cfg.dataset.sequence_length
            if self.cfg.dataset.grayscale:
                in_chans = 1
            else:
                in_chans = 3

            self.encoder = VJepaViT(
                img_size=self.cfg.dataset.image_size,
                patch_size=self.cfg.model.patch_size,
                frames=frames,
                in_chans=in_chans,
                embed_dim=self.cfg.model.embedding_dim,
                depth=self.cfg.model.depth,
                num_heads=self.cfg.model.num_heads,
                mlp_ratio=self.cfg.model.mlp_ratio,
            )

            # Predictor also transformer
            self.predictor = VJepaPredictor(
                embed_dim=self.cfg.model.embedding_dim,
                depth=4,  # Fixed depth for predictor for now
                num_heads=self.cfg.model.num_heads,
            )

            # Autoregressive Fine-tuning Predictor (Simple MLP or small TF)
            # Reuse Predictor class? VJepaPredictor is designed for masking.
            # We need a standard AR predictor for fine-tuning.
            # Let's use the standard JEPA Predictor (Residual MLP) for phase 2 fine-tuning!
            # It takes vectors.

            # Flattened Dim per frame: N_spatial * Embed
            num_spatial = (self.cfg.dataset.image_size // self.cfg.model.patch_size) ** 2
            flat_dim = num_spatial * self.cfg.model.embedding_dim

            self.ar_predictor = Predictor(embedding_dim=flat_dim, hidden_dim=self.cfg.model.hidden_dim)

            self.decoder = VJepaDecoder(
                embed_dim=self.cfg.model.embedding_dim,
                patch_size=self.cfg.model.patch_size,
                img_size=self.cfg.dataset.image_size,
                frames=frames,
                in_chans=in_chans,
            )
            return

        if is_visual:
            if self.cfg.dataset.grayscale:
                channels = 1
            else:
                channels = 3  # RGB
            # For vision, input is (B, History*C, H, W)
            input_channels = channels * self.cfg.dataset.history_length

            self.logger.info(f"Initializing Vision Models with In Channels: {input_channels}")

            self.encoder = VisionEncoder(
                input_channels=input_channels,
                embedding_dim=self.cfg.model.embedding_dim,
                image_size=self.cfg.dataset.image_size,
            )
            # Predictor is always MLP on Latents
            self.predictor = Predictor(embedding_dim=self.cfg.model.embedding_dim, hidden_dim=self.cfg.model.hidden_dim)
            self.decoder = VisionDecoder(
                embedding_dim=self.cfg.model.embedding_dim,
                output_channels=input_channels,
                image_size=self.cfg.dataset.image_size,
            )
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
        model_type = getattr(self.cfg.model, "type", "jepa")

        if model_type == "vjepa":
            self.trainer = VJepaTrainer(
                encoder=self.encoder,
                predictor=self.predictor,
                dataloader=self.dataloader,
                device=self.device,
                lr=self.cfg.training.lr,
            )

            # Phase 1: Pre-training
            self.logger.info("Starting Phase 1: Masked Pre-training")
            self.trainer.train_pretraining(
                epochs=self.cfg.training.pretrain_epochs, mask_ratio=self.cfg.training.mask_ratio
            )

            # Phase 2: Fine-tuning
            self.logger.info("Starting Phase 2: Autoregressive Fine-tuning")
            # We pass the AR Predictor we initialized
            self.ar_predictor = self.ar_predictor.to(self.device)
            self.trainer.train_finetuning(
                autoregressive_predictor=self.ar_predictor, epochs=self.cfg.training.finetune_epochs
            )

            # Save models
            models_dir = os.path.join(self.run_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            torch.save(self.encoder.state_dict(), os.path.join(models_dir, "encoder.pth"))
            torch.save(self.predictor.state_dict(), os.path.join(models_dir, "predictor_masked.pth"))
            torch.save(self.ar_predictor.state_dict(), os.path.join(models_dir, "predictor_ar.pth"))
            return

        self.trainer = JEPATrainer(
            encoder=self.encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            dataloader=self.dataloader,
            config=self.cfg,
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
        model_type = getattr(self.cfg.model, "type", "jepa")

        if model_type == "vjepa":
            visualize_vjepa_reconstruction(
                self.encoder,
                self.decoder,
                self.dataset,
                save_path=recon_path,
                num_samples=5,
                grayscale=self.cfg.dataset.grayscale,
            )

            forecast_path = os.path.join(self.run_dir, "forecast.mp4")
            visualize_vjepa_forecast(
                self.encoder,
                self.ar_predictor,  # Use the AR predictor
                self.decoder,
                self.dataset,
                save_path=forecast_path,
                num_frames=self.cfg.num_vis_points,
                grayscale=self.cfg.dataset.grayscale,
            )
            return

        if self.cfg.dataset.name == "pendulum_image":
            visualize_image_reconstruction(
                self.encoder,
                self.decoder,
                self.dataset,
                save_path=recon_path,
                num_samples=5,
                history_length=self.cfg.dataset.history_length,
                grayscale=self.cfg.dataset.grayscale,
            )
        else:
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
        elif self.cfg.dataset.name == "pendulum_image":
            forecast_path = os.path.join(self.run_dir, "forecast.mp4")
            visualize_image_forecast(
                self.encoder,
                self.predictor,
                self.decoder,
                self.dataset,
                save_path=forecast_path,
                num_frames=self.cfg.num_vis_points,
                history_length=self.cfg.dataset.history_length,
                image_size=self.cfg.dataset.image_size,
                grayscale=self.cfg.dataset.grayscale,
            )

    def run(self):
        self.setup_seeds()
        self.prepare_data()
        self.initialize_models()
        self.train()
        self.visualize()
        self.logger.info(f"Experiment Completed. Results saved to {self.run_dir}")
