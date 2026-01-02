import copy
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models import Decoder, Encoder, Predictor


class JEPATrainer:
    """
    Handles the training of JEPA components (Encoder, Predictor)
    and the separate Decoder.
    """

    def __init__(
        self,
        encoder: Encoder,
        predictor: Predictor,
        decoder: Decoder,
        dataloader: torch.utils.data.DataLoader,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ):
        self.encoder = encoder.to(device)
        self.predictor = predictor.to(device)
        self.decoder = decoder.to(device)
        self.dataloader = dataloader
        self.device = device
        self.lr = lr
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
        criterion = nn.MSELoss()
        ema_decay = 0.99

        self.logger.info(f"Starting JEPA Training for {epochs} epochs...")
        self.encoder.train()
        self.predictor.train()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        for epoch in range(epochs):
            # Dynamic EMA-LR Coupling
            # EMA = 1.0 - k * LR
            # If LR=1e-3, EMA=0.95. If LR drops, EMA increases towards 1.0.
            current_lr = optimizer.param_groups[0]["lr"]
            ema_decay = 1.0 - (10.0 * current_lr)
            # Clip for safety
            ema_decay = max(0.9, min(1.0, ema_decay))

            start_time = time.time()
            epoch_loss = 0

            pbar = tqdm(self.dataloader, desc=f"JEPA Epoch {epoch + 1}/{epochs}")
            for trajectory in pbar:
                trajectory = trajectory.to(self.device)  # (B, 30, 4)

                # Dynamic Slicing
                # History inferred from Encoder input_dim
                # Linear layer is at index 0 of Sequential
                first_layer = self.encoder.net[0]
                if isinstance(first_layer, nn.Linear):
                    H = first_layer.in_features // 4
                else:
                    raise ValueError("Encoder structure changed, cannot infer history length")

                seq_len = trajectory.shape[1]
                max_gap = 5

                # Pick valid start index t
                # We need H frames for context starting at t
                # We need H frames for target starting at t+gap
                # Max index needed is t + max_gap + H

                max_start = seq_len - H - max_gap
                if max_start <= 0:
                    max_start = 1

                t = np.random.randint(0, max_start)
                gap = np.random.randint(1, max_gap + 1)

                context_frames = trajectory[:, t : t + H]
                target_frames = trajectory[:, t + gap : t + gap + H]

                context = context_frames.flatten(start_dim=1)
                target = target_frames.flatten(start_dim=1)

                # Forward Pass
                s_curr = self.encoder(context)

                # Recursive Prediction
                # We apply Predictor 'gap' times: z_t -> z_{t+1} -> ... -> z_{t+gap}
                for _ in range(gap):
                    s_curr = self.predictor(s_curr)

                s_pred = s_curr

                with torch.no_grad():
                    s_target = self.target_encoder(target)

                # Loss & Backprop
                loss = criterion(s_pred, s_target)

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping removed as requested
                optimizer.step()

                # EMA Update for Target Encoder
                with torch.no_grad():
                    for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                        param_k.data = param_k.data * ema_decay + param_q.data * (1.0 - ema_decay)

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            end_time = time.time()
            epoch_duration = end_time - start_time
            avg_loss = epoch_loss / len(self.dataloader)
            self.logger.info(
                f"JEPA Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.6f} | "
                f"Time: {epoch_duration:.2f}s | LR: {current_lr:.2e} | EMA: {ema_decay:.5f}"
            )
            scheduler.step(avg_loss)

    def train_decoder(self, epochs=50):
        """
        Train the Decoder to map Latent -> Observation.
        Encoder is FROZEN during this phase.
        """
        optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        self.logger.info(f"\nStarting Decoder Training for {epochs} epochs...")
        self.decoder.train()
        self.encoder.eval()  # Freeze encoder

        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0

            pbar = tqdm(self.dataloader, desc=f"Decoder Epoch {epoch + 1}/{epochs}")
            for trajectory in pbar:
                trajectory = trajectory.to(self.device)

                # Decoder always learns to reconstruct the context itself?
                # "Train the Decoder to map Latent -> Observation."
                # Usually it reconstructs the input (Autoencoder style).

                first_layer = self.encoder.net[0]
                if isinstance(first_layer, nn.Linear):
                    H = first_layer.in_features // 4
                else:
                    raise ValueError("Encoder structure changed")

                # Just take the first H frames as context
                context_frames = trajectory[:, 0:H]
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
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            end_time = time.time()
            epoch_duration = end_time - start_time
            avg_loss = epoch_loss / len(self.dataloader)
            current_lr = optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Decoder Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.6f} | "
                f"Time: {epoch_duration:.2f}s | LR: {current_lr:.2e}"
            )
            scheduler.step(avg_loss)
