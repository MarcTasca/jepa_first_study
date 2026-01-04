import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Video to Patch Embedding"""

    def __init__(self, img_size=64, patch_size=8, frames=30, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.frames = frames
        self.num_patches = (img_size // patch_size) * (img_size // patch_size) * frames

        # We treat time as just another dimension or we process frame by frame?
        # V-JEPA usually does (T, H, W) patches.
        # Let's do 2D patches for each frame, and flatten Time.
        # Or 3D patches (t, h, w).
        # Let's assume 2D patches (flattened time is handled by transformer position logic)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # Merge B and T for 2D Conv
        x = x.view(B * T, C, H, W)
        x = self.proj(x)  # (B*T, Embed, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B*T, num_patches_spatial, Embed)

        # Reshape back to include time
        x = x.view(B, T, -1, x.shape[-1])  # (B, T, N_s, E)
        x = x.flatten(1, 2)  # (B, T*N_s, E) -> Total patches
        return x


class VJepaViT(nn.Module):
    """
    Vision Transformer for V-JEPA.
    Acts as both Context Encoder and Target Encoder.
    """

    def __init__(
        self,
        img_size=64,
        patch_size=8,
        frames=30,
        in_chans=1,  # Grayscale default
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, frames, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional Embedding (Space-Time)
        # We use learnable pos embed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # x: (B, T, C, H, W)
        x = self.patch_embed(x)
        x = x + self.pos_embed

        if mask is not None:
            # mask: (B, N) where 1 is masked (removed)
            # We select ONLY the visible tokens
            B, N, D = x.shape

            # Mask logic: 0 is keep, 1 is remove
            # We want to keep indices where mask == 0

            # Since mask might be different per batch item (or same), let's assume same for simplicity first, or per batch.
            # Usually mask is per sample.

            x_vis_list = []
            for i in range(B):
                # indices of 0
                vis_idx = (mask[i] == 0).nonzero(as_tuple=True)[0]
                x_vis_list.append(x[i, vis_idx, :])

            # Stack? They might have different lengths if random ratio is not exact?
            # Usually ratio is exact.
            x = torch.stack(x_vis_list)

        x = self.blocks(x)
        x = self.norm(x)
        return x


class VJepaPredictor(nn.Module):
    """
    Predicts target latent from context latent + mask tokens + position queries.
    """

    def __init__(self, embed_dim=192, depth=4, num_heads=4, mlp_ratio=4.0):
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        predictor_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True
        )
        self.blocks = nn.TransformerEncoder(predictor_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.pred_head = nn.Linear(embed_dim, embed_dim)  # Map to target dim (same as embed dim)

    def forward(self, x_vis, pos_embed, mask):
        # x_vis: (B, N_vis, E) - Encoded visible tokens
        # pos_embed: (1, N_total, E) - All positional embeddings
        # mask: (B, N_total) - 1 is masked (target to predict)

        B, N_vis, E = x_vis.shape

        # We need to reconstruct the full sequence with [Mask] tokens at masked positions
        # and x_vis at visible positions.
        # Actually standard MAE/JEPA Predictor input: [ContextTokens, MaskTokens]
        # with correct positional embeddings added.

        # Create output tensor
        # We only predict the MASKED tokens? Or full sequence?
        # JEPA: Predicts specific target blocks.
        # Let's verify: Context -> Predictor -> Prediction for Target.
        # Here we just want to fill in the blanks.

        # Gather positional embeddings for visible and masked

        output_list = []
        for i in range(B):
            # indices of 0
            mask_idx = (mask[i] == 1).nonzero(as_tuple=True)[0]

            # Visible tokens already have pos embed added in encoder? Yes.
            # But the Predictor needs to know WHERE they are. The encoder output 'x_vis'
            # effectively has (Feature + Pos).
            # But for the MASK tokens, we need to add Pos(Mask_Location).

            # Mask tokens
            num_masked = len(mask_idx)
            mask_tokens = self.mask_token.squeeze(0).expand(num_masked, -1)
            mask_pos = pos_embed[0, mask_idx, :]

            mask_tokens = mask_tokens + mask_pos

            # Concatenate Visible and Masked
            # V-JEPA often processes them together
            # x_vis (N_vis, E)
            # mask_tokens (N_mask, E)

            combined = torch.cat([x_vis[i], mask_tokens], dim=0)
            output_list.append(combined)

        x = torch.stack(output_list)

        # Run Predictor
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pred_head(x)

        # We assume the LAST N_mask tokens are the predictions
        # This order (Vis, Mask) must be respected in loss calculation
        return x  # Return full sequence or split?


class VJepaDecoder(nn.Module):
    """
    Reconstructs image from latent tokens.
    MAE-style: Linear projection per token back to pixels.
    """

    def __init__(self, embed_dim=192, patch_size=8, in_chans=1, img_size=64, frames=30):
        super().__init__()
        self.patch_size = patch_size
        self.frames = frames
        self.img_size = img_size
        self.num_patches_per_frame = (img_size // patch_size) ** 2

        output_dim = (patch_size**2) * in_chans
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # x: (B, N_total, E)
        # We assume N_total covers the full video or frame.
        x = self.head(x)  # (B, N, P*P*C)

        # Unpatchify
        # This is annoying without a utility.
        # Let's assume we just return patches for now or try to assemble.
        # If we just want visualization, we can reconstruct the patches and place them.

        # Reshape to (B, T, H/P, W/P, P, P, C) and permute
        B, N, D = x.shape
        P = self.patch_size
        H_p = self.img_size // P
        W_p = self.img_size // P

        # Assume N = T * H_p * W_p
        # T should be inferred
        T = N // (H_p * W_p)
        C = D // (P * P)

        x = x.view(B, T, H_p, W_p, P, P, C)
        # Permute to (B, T, C, H_p, P, W_p, P)
        x = torch.einsum("bthwqp c->btchpwq", x)
        x = x.reshape(B, T, C, H_p * P, W_p * P)

        return x  # (B, T, C, H, W)
