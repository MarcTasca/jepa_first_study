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

    def __init__(self, input_channels: int, embedding_dim: int = 32, image_size: int = 64):
        super().__init__()
        self.image_size = image_size

        # We assume 3 layers of stride 2
        # Final map size = image_size / 2^3 = image_size / 8
        final_size = image_size // 8
        if final_size < 1:
            raise ValueError(f"Image size {image_size} too small for 3 layers of stride 2")

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flat_dim = 128 * final_size * final_size
        self.fc = nn.Linear(self.flat_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be (B, C, H, W)
        feats = self.net(x)
        return self.fc(feats)


class VisionDecoder(nn.Module):
    """
    Deconv-based Decoder to reconstruct images from Latent.
    Output: (B, C, H, W)
    """

    def __init__(self, embedding_dim: int = 32, output_channels: int = 3, image_size: int = 64):
        super().__init__()
        self.image_size = image_size
        self.final_size = image_size // 8

        self.flat_dim = 128 * self.final_size * self.final_size
        self.fc = nn.Linear(embedding_dim, self.flat_dim)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Images are 0-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(-1, 128, self.final_size, self.final_size)
        return self.net(x)
