import copy
import logging
import time
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.config import ExperimentConfig
from src.models import Decoder, Encoder, Predictor, VisionDecoder, VisionEncoder


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class JEPATrainer:
    """
    Handles the training of JEPA components (Encoder, Predictor)
    and the separate Decoder.
    """

    def __init__(
        self,
        encoder: Union[Encoder, VisionEncoder],
        predictor: Predictor,
        decoder: Union[Decoder, VisionDecoder],
        dataloader: torch.utils.data.DataLoader,
        config: ExperimentConfig,
        lr: float = 1e-3,
        ema_start: float = 0.99,
        device: torch.device = torch.device("cpu"),
    ):
        self.cfg = config
        self.encoder = encoder.to(device)
        self.predictor = predictor.to(device)
        self.decoder = decoder.to(device)
        self.dataloader = dataloader
        self.device = device
        self.lr = lr
        self.ema_start = ema_start
        self.logger = logging.getLogger("JEPATrainer")

        # Target Encoder is a copy of Encoder, updated via EMA, NO gradients
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def train_jepa(self, epochs: int = 50):
        """
        Train the Joint Embedding Predictive Architecture.
        Obsertvation(t) -> Encoder -> z(t) -> Predictor -> z(t+1)
        Observation(t+1) -> TargetEncoder -> target_z(t+1)
        Loss = MSE(z(t+1), target_z(t+1))
        """
        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.predictor.parameters()), lr=self.lr)
        # criterion = nn.MSELoss() -- Unused
        ema_decay = self.ema_start

        self.logger.info(f"Starting JEPA Training for {epochs} epochs...")
        self.encoder.train()
        self.predictor.train()

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # Base EMA decay when LR is at initial value
        base_ema = self.ema_start
        initial_lr = self.lr

        for epoch in range(epochs):
            # Dynamic EMA-LR Coupling
            # EMA_decay = 1.0 - (1.0 - base_ema) * (current_lr / initial_lr)
            # This ensures EMA decay approaches 1.0 as LR approaches 0.
            # Example:
            # If LR = initial_lr, EMA = 0.99
            # If LR = initial_lr * 0.1, EMA = 0.999
            current_lr = optimizer.param_groups[0]["lr"]
            ema_decay = 1.0 - (1.0 - base_ema) * (current_lr / initial_lr)

            # Clip for safety
            ema_decay = max(0.9, min(1.0, ema_decay))

            start_time = time.time()
            epoch_loss = 0

            for trajectory in self.dataloader:
                trajectory = trajectory.to(self.device)  # (B, 30, 4)

                # Dynamic Slicing
                # History inferred from Encoder structure
                input_layer = self.encoder.net[0]
                is_vision = isinstance(input_layer, nn.Conv2d)

                if is_vision:
                    # Conv2d(in_channels, ...)
                    # in_channels = History * Channels_per_frame
                    # We need to know Channels per frame.
                    # For Pendulum Image: 3.
                    # We can infer H if we assume C=3.
                    # Or we just pass H as config? Trainer doesn't have config.
                    # Let's infer H assuming RGB (3 channels)
                    in_channels = input_layer.in_channels
                    H = in_channels // 3
                elif isinstance(input_layer, nn.Linear):
                    H = input_layer.in_features // 4
                else:
                    # Fallback or error
                    # Maybe it's a Sequential inside?
                    raise ValueError("Unknown Encoder structure")

                seq_len = trajectory.shape[1]
                max_gap = 5

                # Pick valid start index t
                max_start = seq_len - H - max_gap
                if max_start <= 0:
                    max_start = 1

                t = np.random.randint(0, max_start)
                gap = np.random.randint(1, max_gap + 1)

                context_frames = trajectory[:, t : t + H]
                target_frames = trajectory[:, t + gap : t + gap + H]

                if is_vision:
                    # (B, H, C, W, H) -> (B, H*C, W, H)
                    # Start dim 1 (H), End dim 2 (C) -> merged
                    B, _, C, Hei, Wid = context_frames.shape
                    context = context_frames.reshape(B, -1, Hei, Wid)
                    target = target_frames.reshape(B, -1, Hei, Wid)
                else:
                    context = context_frames.flatten(start_dim=1)
                    target = target_frames.flatten(start_dim=1)

                # Forward Pass
                s_curr = self.encoder(context)

                # Recursive Prediction
                for _ in range(gap):
                    s_curr = self.predictor(s_curr)

                # s_pred = s_curr -- Unused

                with torch.no_grad():
                    s_target = self.target_encoder(target)

                # Loss & Backprop
                # mse_loss = criterion(s_pred, s_target) -- Removed unused

                # Multistep VICReg Training
                input_layer = self.encoder.net[0]
                if isinstance(input_layer, nn.Conv2d):
                    H = input_layer.in_channels // 3
                    _, _, C, Hei, Wid = trajectory.shape  # (B, T, C, H, W)? No, (B, T, 3, 64, 64)
                    # wait, trajectory is (B, 30, 3, 64, 64)
                else:
                    H = 2

                # Correctly determine max_start
                T_seq = trajectory.shape[1]
                horizon = self.cfg.training.prediction_horizon
                max_start = T_seq - H - horizon - 1
                if max_start <= 0:
                    max_start = 1
                t = np.random.randint(0, max_start)

                # Prepare Context
                context_frames = trajectory[:, t : t + H]
                if isinstance(input_layer, nn.Conv2d):
                    B, _, C, Hei, Wid = context_frames.shape
                    context = context_frames.reshape(B, -1, Hei, Wid)
                else:
                    context = context_frames.flatten(start_dim=1)

                s_curr = self.encoder(context)

                # Zero out separate loss accumulators for this batch
                batch_loss = 0

                for k in range(1, horizon + 1):
                    # 1. Predict Next Step
                    s_curr = self.predictor(s_curr)

                    # 2. Get Target for this Step
                    target_frames = trajectory[:, t + k : t + k + H]
                    if isinstance(input_layer, nn.Conv2d):
                        target_in = target_frames.reshape(B, -1, Hei, Wid)
                    else:
                        target_in = target_frames.flatten(start_dim=1)

                    with torch.no_grad():
                        s_target = self.target_encoder(target_in)

                    # 3. VICReg Loss Components
                    # Invariance
                    sim_loss = F.mse_loss(s_curr, s_target)

                    # Variance
                    std_x = torch.sqrt(s_curr.var(dim=0) + 0.0001)
                    std_y = torch.sqrt(s_target.var(dim=0) + 0.0001)
                    std_loss = torch.mean(F.relu(1.0 - std_x)) + torch.mean(F.relu(1.0 - std_y))

                    # Covariance
                    x_centered = s_curr - s_curr.mean(dim=0)
                    y_centered = s_target - s_target.mean(dim=0)
                    cov_x = (x_centered.T @ x_centered) / (B - 1)
                    cov_y = (y_centered.T @ y_centered) / (B - 1)

                    # off-diagonal elements
                    cov_loss = (off_diagonal(cov_x).pow(2).sum() / 64) + (off_diagonal(cov_y).pow(2).sum() / 64)

                    step_loss = (
                        self.cfg.model.sim_coeff * sim_loss
                        + self.cfg.model.std_coeff * std_loss
                        + self.cfg.model.cov_coeff * cov_loss
                    )

                    batch_loss += step_loss

                # Average over horizon
                loss = batch_loss / horizon

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # EMA Update
                with torch.no_grad():
                    for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                        param_k.data = param_k.data * ema_decay + param_q.data * (1.0 - ema_decay)

                epoch_loss += loss.item()

            scheduler.step()

            end_time = time.time()
            epoch_duration = end_time - start_time
            avg_loss = epoch_loss / len(self.dataloader)

            self.logger.info(
                f"JEPA Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f} | "
                f"Time: {epoch_duration:.2f}s | LR: {current_lr:.2e}"
            )

    def train_decoder(self, epochs=50):
        """
        Train the Decoder to map Latent -> Observation.
        Encoder is FROZEN during this phase.
        """
        optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        self.logger.info(f"\nStarting Decoder Training for {epochs} epochs...")
        self.decoder.train()
        self.encoder.eval()  # Freeze encoder

        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0

            for trajectory in self.dataloader:
                trajectory = trajectory.to(self.device)

                # Decoder always learns to reconstruct the context itself?
                # "Train the Decoder to map Latent -> Observation."
                # Usually it reconstructs the input (Autoencoder style).

                # Decoder always learns to reconstruct the context itself?
                # "Train the Decoder to map Latent -> Observation."
                # Usually it reconstructs the input (Autoencoder style).

                input_layer = self.encoder.net[0]
                is_vision = isinstance(input_layer, nn.Conv2d)

                if is_vision:
                    in_channels = input_layer.in_channels
                    H = in_channels // 3
                elif isinstance(input_layer, nn.Linear):
                    H = input_layer.in_features // 4
                else:
                    raise ValueError("Encoder structure changed")

                # Just take the first H frames as context
                context_frames = trajectory[:, 0:H]

                if is_vision:
                    B, _, C, Hei, Wid = context_frames.shape
                    context = context_frames.reshape(B, -1, Hei, Wid)
                else:
                    context = context_frames.flatten(start_dim=1)

                with torch.no_grad():
                    embedding = self.encoder(context)

                reconstruction = self.decoder(embedding)
                loss = criterion(reconstruction, context)

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping removed
                optimizer.step()

                epoch_loss += loss.item()

            end_time = time.time()
            epoch_duration = end_time - start_time
            avg_loss = epoch_loss / len(self.dataloader)
            current_lr = optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Decoder Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.6f} | "
                f"Time: {epoch_duration:.2f}s | LR: {current_lr:.2e}"
            )
            scheduler.step()
