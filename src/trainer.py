import copy
import logging
import time
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models import Decoder, Encoder, Predictor, VisionDecoder, VisionEncoder


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
        lr: float = 1e-3,
        ema_start: float = 0.99,
        device: torch.device = torch.device("cpu"),
        use_temporal_masking: bool = False,
        mask_ratio: float = 0.4,
        min_context_frames: int = 3,
    ):
        self.encoder = encoder.to(device)
        self.predictor = predictor.to(device)
        self.decoder = decoder.to(device)
        self.dataloader = dataloader
        self.device = device
        self.lr = lr
        self.ema_start = ema_start
        self.use_temporal_masking = use_temporal_masking
        self.mask_ratio = mask_ratio
        self.min_context_frames = min_context_frames
        self.logger = logging.getLogger("JEPATrainer")

        # Target Encoder is a copy of Encoder, updated via EMA, NO gradients
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def generate_temporal_mask(self, seq_len, mask_ratio=0.4, min_context=3):
        """
        Generate random temporal mask for sequence following I-JEPA philosophy.

        Args:
            seq_len: Total sequence length
            mask_ratio: Fraction of frames to mask (target)
            min_context: Minimum visible frames for context

        Returns:
            context_indices: Sorted list of visible frame indices
            target_indices: Sorted list of masked frame indices
        """
        n_mask = int(seq_len * mask_ratio)
        n_context = seq_len - n_mask

        # Ensure minimum context
        if n_context < min_context:
            n_context = min_context
            n_mask = seq_len - min_context

        # Randomly select which frames to use as context
        all_indices = np.arange(seq_len)
        np.random.shuffle(all_indices)

        context_indices = sorted(all_indices[:n_context].tolist())
        target_indices = sorted(all_indices[n_context:].tolist())

        return context_indices, target_indices

    def train_jepa(self, epochs: int = 50):
        """
        Train the Joint Embedding Predictive Architecture.
        Obsertvation(t) -> Encoder -> z(t) -> Predictor -> z(t+1)
        Observation(t+1) -> TargetEncoder -> target_z(t+1)
        Loss = MSE(z(t+1), target_z(t+1))
        """
        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.predictor.parameters()), lr=self.lr)
        criterion = nn.MSELoss()
        ema_decay = self.ema_start

        self.logger.info(f"Starting JEPA Training for {epochs} epochs...")
        self.encoder.train()
        self.predictor.train()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

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

                if self.use_temporal_masking:
                    # Temporal Masking (I-JEPA style)
                    ctx_idx, tgt_idx = self.generate_temporal_mask(
                        seq_len=seq_len, mask_ratio=self.mask_ratio, min_context=self.min_context_frames
                    )

                    # Select frames
                    context_frames = trajectory[:, ctx_idx]
                    target_frames = trajectory[:, tgt_idx]

                    # Pad/interpolate to match encoder's expected history length
                    # Simple padding: pad context to H frames with zeros if needed
                    if len(ctx_idx) < H:
                        # Pad with zeros
                        pad_len = H - len(ctx_idx)
                        if is_vision:
                            B, _, C, Hei, Wid = context_frames.shape
                            padding = torch.zeros(B, pad_len, C, Hei, Wid, device=context_frames.device)
                        else:
                            B, _, F = context_frames.shape
                            padding = torch.zeros(B, pad_len, F, device=context_frames.device)
                        context_frames = torch.cat([context_frames, padding], dim=1)
                    elif len(ctx_idx) > H:
                        # Truncate to H frames (keep most recent)
                        context_frames = context_frames[:, -H:]

                    # For target, also standardize size
                    if len(tgt_idx) < H:
                        pad_len = H - len(tgt_idx)
                        if is_vision:
                            B, _, C, Hei, Wid = target_frames.shape
                            padding = torch.zeros(B, pad_len, C, Hei, Wid, device=target_frames.device)
                        else:
                            B, _, F = target_frames.shape
                            padding = torch.zeros(B, pad_len, F, device=target_frames.device)
                        target_frames = torch.cat([target_frames, padding], dim=1)
                    elif len(tgt_idx) > H:
                        target_frames = target_frames[:, :H]

                    # Gap for predictor is the distance between last context and first target
                    gap = tgt_idx[0] - ctx_idx[-1] if len(tgt_idx) > 0 and len(ctx_idx) > 0 else 1
                    gap = max(1, gap)  # At least 1 step

                else:
                    # Original fixed-gap approach
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

                s_pred = s_curr

                with torch.no_grad():
                    s_target = self.target_encoder(target)

                # Loss & Backprop
                loss = criterion(s_pred, s_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # EMA Update for Target Encoder
                with torch.no_grad():
                    for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                        param_k.data = param_k.data * ema_decay + param_q.data * (1.0 - ema_decay)

                epoch_loss += loss.item()

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
            scheduler.step(avg_loss)
