import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.masking import RandomMaskingGenerator


class VJepaTrainer:
    def __init__(self, encoder, predictor, dataloader, device, run_dir, lr=1e-3):
        self.encoder = encoder.to(device)
        self.predictor = predictor.to(device)
        self.dataloader = dataloader
        self.device = device
        self.run_dir = run_dir
        self.lr = lr
        self.logger = logging.getLogger("VJepaTrainer")

        # Create checkpoints dir
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train_pretraining(self, epochs=20, mask_ratio=0.6):
        """
        Phase 1: Masked Feature Prediction
        """
        optimizer = optim.AdamW(list(self.encoder.parameters()) + list(self.predictor.parameters()), lr=self.lr)
        criterion = nn.SmoothL1Loss()  # Huber loss often better for features

        # Mask generator
        # We need size from encoder
        # inputs are (B, T, C, H, W).
        # We assume dataset returns (B, T, C, H, W)

        self.logger.info(f"Starting V-JEPA Pre-training for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_loss = 0
            start_time = time.time()

            for i, batch in enumerate(self.dataloader):
                # batch: (B, T, C, H, W)
                batch = batch.to(self.device)

                # Check shapes
                if len(batch.shape) == 4:  # (B, C, H, W) -> Single frame? Or (B, T, C, H*W)?
                    # If single frame, add T=1
                    batch = batch.unsqueeze(1)

                B, T, C, H, W = batch.shape

                # Create mask
                # We need input size for generator (Frames, H, W)
                # But our patch embed handles H,W. Generator needs H, W in pixels?
                # Using RandomMaskingGenerator which works on patch counts.
                # Patch Embed knows num_patches.

                # We assume 64x64 img, 8x8 patch -> 8x8=64 patches per frame.
                # 30 frames -> 1920 patches.
                # RandomMaskingGenerator takes (H_patches, W_patches)? No, it generates flat mask.

                num_patches = self.encoder.patch_embed.num_patches
                # Create flat mask (B, N)

                mask = (
                    torch.stack(
                        [
                            torch.from_numpy(
                                # Simple random masking on the flat sequence
                                # In real V-JEPA we'd use tube masking
                                RandomMaskingGenerator((1, num_patches), mask_ratio)()
                            )
                            for _ in range(B)
                        ]
                    )
                    .to(self.device)
                    .long()
                )

                # Target: Full encoding of the unmasked image (Teacher)
                # In standard I-JEPA/V-JEPA, target is computed by a moving average Teacher Encoder.
                # For simplicity in this demo, we can use the SAME encoder (Student) as target
                # but detach gradients (MAE style) or just copy it.
                # MAE uses Reconstruction (Pixels) as target.
                # JEPA uses Features as target.
                # To prevent collapse, we need a Target Encoder (EMA) or Asymmetric Path.
                # Let's use the Encoder itself but detach. (Simpler, closer to BYOL/SimSiam if predictor is strong).
                # Ideally we should implement EMA Teacher.

                with torch.no_grad():
                    target_latents = self.encoder(batch)  # (B, N, E)

                # Context: Masked encoding
                context_latents = self.encoder(batch, mask=mask)  # (B, N_vis, E)

                # Predictor
                # Needs to predict target_latents at masked positions
                # We feed (context, pos_embed, mask) -> (B, N_total, E) or (B, N_masked, E)
                # Our implemented predictor returns N_vis + N_mask tokens concatenated.
                # Order: [Visible ..., Masked ...]

                full_prediction = self.predictor(context_latents, self.encoder.pos_embed, mask)

                # Extract predictions for masked tokens
                # The predictor implementation in models_vjepa.py appends masked tokens AT THE END.
                # So the last N_masked tokens correspond to the masked indices.

                # We need to gather the corresponding Target latents
                # Target latents are (B, N_total, E) ordered by position.
                # We need to pick the ones where mask==1.

                loss = 0
                for b in range(B):
                    # Masked indices
                    mask_idx = (mask[b] == 1).nonzero(as_tuple=True)[0]
                    num_masked = len(mask_idx)

                    # Predictions are the last num_masked tokens
                    pred = full_prediction[b, -num_masked:]

                    # Targets
                    targ = target_latents[b, mask_idx]

                    loss += criterion(pred, targ)

                loss = loss / B

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    batches_done = i + 1
                    batches_total = len(self.dataloader)

                    speed = batches_done / elapsed
                    eta = (batches_total - batches_done) / speed

                    current_lr = optimizer.param_groups[0]["lr"]

                    self.logger.info(
                        f"Epoch {epoch + 1} [{i}/{batches_total}] "
                        f"Loss: {loss.item():.4f} | "
                        f"AvgLoss: {epoch_loss / batches_done:.4f} | "
                        f"Speed: {speed:.2f} batch/s | "
                        f"ETA: {eta / 60:.1f} min | "
                        f"LR: {current_lr:.2e}"
                    )

            avg_loss = epoch_loss / len(self.dataloader)
            self.logger.info(
                f"V-JEPA Pre-training Epoch {epoch + 1}/{epochs} Completed: "
                f"Loss={avg_loss:.4f} | Time={time.time() - start_time:.1f}s"
            )

            # Save Checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "encoder": self.encoder.state_dict(),
                    "predictor": self.predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                os.path.join(self.ckpt_dir, f"pretrain_epoch_{epoch + 1}.pth"),
            )

            torch.save(self.encoder.state_dict(), os.path.join(self.ckpt_dir, "latest_encoder.pth"))

            torch.save(self.encoder.state_dict(), os.path.join(self.ckpt_dir, "latest_encoder.pth"))

    def load_checkpoint(self, path):
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        # Load models
        if "encoder" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder"])
        if "predictor" in checkpoint:
            self.predictor.load_state_dict(checkpoint["predictor"])
        if "ar_predictor" in checkpoint:
            # We statefully assume caller handles this if we are in fine-tuning
            pass

        # We need to handle optimizer.
        # But optimizer is created inside train_pretraining/train_finetuning.
        # This is a design flaw in current Trainer. Optimizer should be self.optimizer?
        # For now, let's just load weights. To resume training properly with optimizer state,
        # we'd need to init optimizer in __init__ or have train() accept a resume flag to load it.

        return checkpoint.get("epoch", 0)

    def train_finetuning(self, autoregressive_predictor, epochs=20):
        """
        Phase 2: Autoregressive Fine-tuning.
        Freeze Encoder, Trace Encoder.
        Train new AR Predictor to predict z_{t+1} from z_t.
        """
        # Freeze Encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        optimizer = optim.Adam(autoregressive_predictor.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        self.logger.info(f"Starting V-JEPA Fine-tuning (Autoregressive) for {epochs} epochs...")

        # Similar to standard JEPA training loop
        # But we only train the AR predictor

        for epoch in range(epochs):
            epoch_loss = 0

            for batch in self.dataloader:
                batch = batch.to(self.device)
                B, T, C, H, W = batch.shape

                # We encode the full sequence frame-by-frame
                # Batch is (B, T, C, H, W)
                # Encoder expects (B, T, C, H, W) and returns (B, T*N_s, E)
                # Wait, our Encoder mixes time/space.
                # For autoregressive task, we need to treat Time strictly.
                # If V-JEPA was trained on spatio-temporal transformers, the latent Z represents the whole video chunk.
                # That's not causal.

                # CHALLENGE: V-JEPA encodes the whole block (Context).
                # If we want to fine-tune for Autoregressive, we need to use the encoder in a per-frame way?
                # OR we just accept that Z is a video descriptor and we predict Z of the *next* video chunk?
                # User wants "autoregressive" like the original demo (frame by frame).
                # But V-JEPA ViT usually processes a block.

                # Solution: Use the V-JEPA ViT to encode SINGLE FRAMES (T=1).
                # Since it has pos embed for T=30, we just use the first T=1 pos embeds?
                # Or we resize pos embeds.

                # Let's assume we use T=1 for fine-tuning.
                # Reshape batch to (B*T, 1, C, H, W)

                batch_flat = batch.view(B * T, 1, C, H, W)

                with torch.no_grad():
                    # We need to adjust pos_embed if trained on 30 frames.
                    # This is tricky.
                    # Simpler approach: Train V-JEPA on T=1 or small blocks?
                    # Or use interpolate_pos_encoding.

                    # For this demo, let's assume we just use the first frame's pos embed
                    # effectively treating the encoder as valid for single frames.
                    # We might need to hack the encoder's frames=30 attr.

                    # Temporary hack: Force encoder frames=1
                    orig_frames = self.encoder.patch_embed.frames
                    self.encoder.patch_embed.frames = 1
                    # Slice pos embed to first frame (N_spatial)
                    num_spatial = (H // 8) * (W // 8)
                    full_pos_embed = self.encoder.pos_embed
                    self.encoder.pos_embed = nn.Parameter(full_pos_embed[:, :num_spatial, :])

                    z_seq = self.encoder(batch_flat)  # (B*T, N_s, E)

                    # Restore
                    self.encoder.patch_embed.frames = orig_frames
                    self.encoder.pos_embed = nn.Parameter(full_pos_embed)

                # z_seq is (B*T, N_spatial, E)
                # Reshape to (B, T, N_spatial*E) -> Let's flatten spatial for the simple MLP predictor
                z_seq = z_seq.flatten(1)  # (B*T, N_s*E)
                z_seq = z_seq.view(B, T, -1)  # (B, T, FlatDim)

                # Now standard AR training
                # Context z_t, Target z_{t+1}

                context = z_seq[:, :-1]
                target = z_seq[:, 1:]

                pred = autoregressive_predictor(context)  # (B, T-1, FlatDim) checks out if RNN/Transformer
                # If MLP, we flatten time

                # Assume AR predictor is simple MLP for next step
                # context (B*(T-1), Dim)
                # target (B*(T-1), Dim)

                pred = autoregressive_predictor(context.reshape(-1, context.shape[-1]))
                targ = target.reshape(-1, target.shape[-1])

                loss = criterion(pred, targ)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)
            self.logger.info(f"Fine-tuning Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}")

            # Save Checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "ar_predictor": autoregressive_predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                os.path.join(self.ckpt_dir, f"finetune_epoch_{epoch + 1}.pth"),
            )

            torch.save(autoregressive_predictor.state_dict(), os.path.join(self.ckpt_dir, "latest_ar_predictor.pth"))
