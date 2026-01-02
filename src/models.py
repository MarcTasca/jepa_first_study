import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Maps 2D observation to Latent Representation.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 256, embedding_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    """
    Predicts Target Embedding from Context Embedding.
    """

    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    Decodes Latent Representation back to 2D observation.
    Used for verification/visualization only.
    """

    def __init__(self, embedding_dim: int = 32, output_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisionEncoder(nn.Module):
    """
    CNN-based Encoder for image observations.
    Input: (B, C * History, H, W)
    """

    def __init__(self, input_channels: int, embedding_dim: int = 32):
        super().__init__()
        # 64x64 input
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # -> 32x32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 16x16x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> 8x8x128
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128 * 8 * 8, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be (B, C, H, W)
        feats = self.net(x)
        return self.fc(feats)


class VisionDecoder(nn.Module):
    """
    Deconv-based Decoder to reconstruct images from Latent.
    Output: (B, C, H, W)
    """

    def __init__(self, embedding_dim: int = 32, output_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 128 * 8 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # -> 64x64
            nn.Sigmoid(),  # Images are 0-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(-1, 128, 8, 8)
        return self.net(x)
