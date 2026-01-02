import hashlib
import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.config import DatasetConfig


class TrajectoryDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset simulating a point moving on a circle.
    context: (x, y) at time t
    target: (x, y) at time t+1
    """

    def __init__(self, size: int = 2000, sequence_length: int = 10, data: Optional[torch.Tensor] = None):
        self.size = size

        if data is not None:
            self.data = data
            return

        self.data = []
        # Random start angles for variety
        start_angles = np.random.rand(size) * 2 * np.pi

        for i in range(size):
            # Generate a sequence of angles for consistent trajectory
            angles = start_angles[i] + np.linspace(0, 2 * np.pi, sequence_length)
            x = np.cos(angles)
            y = np.sin(angles)
            trajectory = np.stack([x, y], axis=1)
            self.data.append(trajectory)

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        traj = self.data[index]
        context = traj[0]  # (2,)
        target = traj[1]  # (2,)
        return context, target


class SpiralTrajectoryDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset simulating a point moving on a spiral.
    Non-linear manifold: r grows with theta.
    """

    def __init__(
        self, size: int = 2000, sequence_length: int = 10, loops: int = 3, data: Optional[torch.Tensor] = None
    ):
        self.size = size

        if data is not None:
            self.data = data
            return

        self.data = []
        # Random start angles
        # We want the spiral to have multiple loops, e.g., 0 to 3*2pi
        max_angle = loops * 2 * np.pi
        # Use sqrt sampling for uniform arc-length distribution: theta ~ sqrt(U)
        start_angles = np.sqrt(np.random.rand(size)) * (max_angle - np.pi / 2)

        for i in range(size):
            angles = start_angles[i] + np.linspace(0, np.pi / 2, sequence_length)  # Short segments

            # r = theta / max_angle (normalized radius 0 to 1)
            r = angles / max_angle

            x = r * np.cos(angles)
            y = r * np.sin(angles)
            trajectory = np.stack([x, y], axis=1)
            self.data.append(trajectory)

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        traj = self.data[index]
        context = traj[0]
        target = traj[1]
        return context, target


class LissajousTrajectoryDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset simulating an Expanding Lissajous curve.
    x = (theta/max) * cos(a*theta)
    y = (theta/max) * sin(b*theta)
    """

    def __init__(
        self,
        size: int = 100000,
        loops: int = 3,
        a: int = 3,
        b: int = 5,
        sequence_length: int = 10,
        data: Optional[torch.Tensor] = None,
    ):
        self.size = size

        if data is not None:
            self.data = data
            return

        self.data = []
        max_angle = loops * 2 * np.pi

        # Sqrt sampling for roughly uniform density along the expanding curve
        start_angles = np.sqrt(np.random.rand(size)) * (max_angle - np.pi / 2)

        for i in range(size):
            angles = start_angles[i] + np.linspace(0, np.pi / 10, sequence_length)  # Smaller step for higher freq

            r = angles / max_angle
            x = r * np.cos(a * angles)
            y = r * np.sin(b * angles)

            trajectory = np.stack([x, y], axis=1)
            self.data.append(trajectory)

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        traj = self.data[index]
        context = traj[0]
        target = traj[1]
        return context, target


class DoublePendulumDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset simulating a Double Pendulum (Chaotic System).
    Uses Runge-Kutta 4 integration of the Equations of Motion.
    Returns the (x, y) coordinates of the tip of the second pendulum.
    """

    def __init__(
        self,
        size: int = 100000,
        dt: float = 0.05,
        history_length: int = 3,
        sequence_length: int = 30,
        data: Optional[torch.Tensor] = None,
    ):
        self.size = size
        self.dt = dt
        self.history_length = history_length
        self.sequence_length = sequence_length
        self.L1, self.L2 = 1.0, 1.0  # Lengths
        self.m1, self.m2 = 1.0, 1.0  # Masses
        self.g = 9.81

        if data is not None:
            self.data = data
            return

        self.data = []

        # Pre-generate random initial states [theta1, theta2, omega1, omega2]
        # High energy states to ensure chaos
        init_states = np.random.rand(size, 4) * 2 * np.pi
        init_states[:, 2:] *= 1.0  # Lower initial velocity for swinging chaos

        # Required simulation steps
        sim_steps = sequence_length

        for i in range(size):
            state = init_states[i]
            trajectory = []

            # Simulate
            for _ in range(sim_steps):
                # Convert polar to cartesian (Tip of 2nd pendulum)
                theta1, theta2 = state[0], state[1]
                x1 = self.L1 * np.sin(theta1)
                y1 = -self.L1 * np.cos(theta1)
                x2 = x1 + self.L2 * np.sin(theta2)
                y2 = y1 - self.L2 * np.cos(theta2)

                trajectory.append([x1, y1, x2, y2])

                # Integrate next step
                state = self.rk4_step(state, self.dt)

            traj = np.array(trajectory)  # Shape (sequence_length, 4)
            self.data.append(traj)

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)

    def derivs(self, state: np.ndarray) -> np.ndarray:
        t1, t2, w1, w2 = state

        delta = t2 - t1
        den1 = (self.m1 + self.m2) * self.L1 - self.m2 * self.L1 * np.cos(delta) * np.cos(delta)
        den2 = (self.L2 / self.L1) * den1

        d_t1 = w1
        d_t2 = w2

        d_w1 = (
            self.m2 * self.L1 * w1 * w1 * np.sin(delta) * np.cos(delta)
            + self.m2 * self.g * np.sin(t2) * np.cos(delta)
            + self.m2 * self.L2 * w2 * w2 * np.sin(delta)
            - (self.m1 + self.m2) * self.g * np.sin(t1)
        ) / den1

        d_w2 = (
            -self.m2 * self.L2 * w2 * w2 * np.sin(delta) * np.cos(delta)
            + (self.m1 + self.m2) * self.g * np.sin(t1) * np.cos(delta)
            - (self.m1 + self.m2) * self.L1 * w1 * w1 * np.sin(delta)
            - (self.m1 + self.m2) * self.g * np.sin(t2)
        ) / den2

        return np.array([d_t1, d_t2, d_w1, d_w2])

    def rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.derivs(state)
        k2 = self.derivs(state + dt * k1 / 2)
        k3 = self.derivs(state + dt * k2 / 2)
        k4 = self.derivs(state + dt * k3)
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        # Return entire sequence
        return self.data[index]


class PixelPendulumDataset(DoublePendulumDataset):
    """
    Renders Double Pendulum trajectories as pixel frames.
    Input: Coordinates from DoublePendulumDataset
    Output: RGB Image Sequence (Sequence, 3, H, W)
    """

    def __init__(
        self,
        size: int = 100000,
        dt: float = 0.05,
        history_length: int = 3,
        sequence_length: int = 30,
        image_size: int = 64,
        data: Optional[torch.Tensor] = None,
        precompute: bool = True,
    ):
        # We reuse the physics simulation from the parent class
        # This populates self.data with coordinates
        super().__init__(size, dt, history_length, sequence_length, data)
        self.image_size = image_size
        self.precompute = precompute

        if self.precompute:
            # Check if data provided is already rendered (from cache)
            if self.data is not None and self.data.ndim == 5:
                # Already (N, Seq, 3, H, W)
                print("[PixelPendulumDataset] Loaded pre-computed images from cache. Skipping render.")
                # self.data is already set by super()
            else:
                print(
                    f"[PixelPendulumDataset] Pre-computing {size} sequences of "
                    f"{image_size}x{image_size} images into RAM..."
                )
                # We must verify we have enough RAM
                # Estimate: size * sequence * 3 * H * W * 4 bytes
                # 10k * 30 * 3 * 32 * 32 * 4 = ~3.6 GB for 32x32?
                # Wait: 10,000 * 30 * 3 * 32 * 32 * 4 / 1024^3 = 3.5 GB
                # 100k samples -> 35 GB.
                # We must warn or limit.

                rendered_data = []
                for i in tqdm(range(len(self.data)), desc="Rendering", unit="seq"):
                    coords = self.data[i]
                    frames = []
                    for t in range(len(coords)):
                        x1, y1, x2, y2 = coords[t]
                        frame = self.render_frame(x1.item(), y1.item(), x2.item(), y2.item())
                        frames.append(frame)
                    rendered_data.append(torch.stack(frames))

                print("Rendering complete.")
                self.data = torch.stack(rendered_data)
                # Now self.data is (N, Seq, 3, H, W)
                # We no longer need the coordinate data, it is overwritten.

    def render_frame(self, x1, y1, x2, y2):
        """
        Renders a single frame of the double pendulum.
        Coordinate system: center (0,0) is at image center.
        Scale: Total length (L1+L2=2) corresponds to 90% of half-width.
        """
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        cx, cy = self.image_size // 2, self.image_size // 2
        # Scale: max extent is 2.0 (L1+L2). We map 2.2 to half_width to allow margins
        scale = (self.image_size / 2) / 2.2

        # Transform coords to pixels
        # y is typically up in physics but down in images. We'll invert y.
        px0, py0 = cx, cy
        px1, py1 = cx + x1 * scale, cy - y1 * scale
        px2, py2 = cx + x2 * scale, cy - y2 * scale

        # Draw rods
        # Rod 1: Origin to P1 (Cyan)
        draw.line([(px0, py0), (px1, py1)], fill=(0, 255, 255), width=2)
        # Rod 2: P1 to P2 (Magenta)
        draw.line([(px1, py1), (px2, py2)], fill=(255, 0, 255), width=2)

        # Draw masses (circles)
        r = 2  # radius of mass
        draw.ellipse([px1 - r, py1 - r, px1 + r, py1 + r], fill=(255, 255, 0))  # Yellow
        draw.ellipse([px2 - r, py2 - r, px2 + r, py2 + r], fill=(255, 0, 0))  # Red

        # Convert to tensor (C, H, W) normalized to [0,1]
        img_np = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)

        return tensor

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.precompute:
            return self.data[index]

        # Get coordinate trajectory from parent: (Seq, 4)
        coords = self.data[index]  # Tensor

        frames = []
        for t in range(len(coords)):
            x1, y1, x2, y2 = coords[t]
            frame = self.render_frame(x1.item(), y1.item(), x2.item(), y2.item())
            frames.append(frame)

        # Stack to (Seq, C, H, W)
        return torch.stack(frames)


class DatasetFactory:
    """
    Factory to instantiate datasets based on configuration.
    Handles caching to avoid re-generating expensive datasets.
    """

    @staticmethod
    def get_cache_path(config: DatasetConfig) -> Tuple[str, dict]:
        """Generates a unique cache filename based on config parameters."""
        # Create a unique hash based on relevant config parameters
        # We assume the Global Seed is handled externally or consistent
        config_dict = {
            "name": config.name,
            "size": config.size,
            "history_length": config.history_length,
            "dt": config.dt,
            "sequence_length": config.sequence_length,
            "spiral_loops": config.spiral_loops,
            "lissajous_a": config.lissajous_a,
            "lissajous_b": config.lissajous_b,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

        # Structure: data/{name}/{hash}/
        # Files: dataset.pt, metadata.json
        cache_dir = os.path.join(config.cache_dir, config.name, config_hash)
        os.makedirs(cache_dir, exist_ok=True)

        pt_path = os.path.join(cache_dir, "dataset.pt")
        return pt_path, config_dict

    @staticmethod
    def get_dataset(config: DatasetConfig):
        # Ensure cache directory exists
        if config.use_cache:
            # Main data dir created by get_cache_path logic or explicit makedirs
            # We rely on get_cache_path creating the subdir now?
            # Ideally get_cache_path shouldn't have side effects, but for simplicity let's do it there or just before.
            # Let's keep get_cache_path pure and do makedirs here.
            pass

        if config.use_cache:
            cache_path, config_dict = DatasetFactory.get_cache_path(config)

            if not config.regenerate and os.path.exists(cache_path):
                print(f"[DatasetFactory] Loading cached dataset from {cache_path}")
                try:
                    data = torch.load(cache_path)
                    # We pass the loaded data to the constructor to skip generation
                    if config.name == "circle":
                        return TrajectoryDataset(size=config.size, sequence_length=config.sequence_length, data=data)
                    elif config.name == "spiral":
                        return SpiralTrajectoryDataset(
                            size=config.size,
                            sequence_length=config.sequence_length,
                            loops=config.spiral_loops,
                            data=data,
                        )
                    elif config.name == "lissajous":
                        return LissajousTrajectoryDataset(
                            size=config.size,
                            sequence_length=config.sequence_length,
                            a=config.lissajous_a,
                            b=config.lissajous_b,
                            data=data,
                        )
                    elif config.name == "pendulum":
                        return DoublePendulumDataset(
                            size=config.size,
                            dt=config.dt,
                            history_length=config.history_length,
                            sequence_length=config.sequence_length,
                            data=data,
                        )
                    elif config.name == "pendulum_image":
                        # For image dataset, the CACHED data is still coordinates!
                        # We just load it into PIxelPendulumDataset which renders on-the-fly.
                        return PixelPendulumDataset(
                            size=config.size,
                            dt=config.dt,
                            history_length=config.history_length,
                            sequence_length=config.sequence_length,
                            image_size=config.image_size,
                            data=data,
                        )
                except Exception as e:
                    print(f"[DatasetFactory] Failed to load cache: {e}. Regenerating...")

        # If not cached or regenerate is True, create normally
        print(f"[DatasetFactory] Generating new dataset: {config.name} (Size: {config.size})")
        dataset = None
        if config.name == "circle":
            dataset = TrajectoryDataset(size=config.size, sequence_length=config.sequence_length)
        elif config.name == "spiral":
            dataset = SpiralTrajectoryDataset(
                size=config.size, sequence_length=config.sequence_length, loops=config.spiral_loops
            )
        elif config.name == "lissajous":
            dataset = LissajousTrajectoryDataset(
                size=config.size, sequence_length=config.sequence_length, a=config.lissajous_a, b=config.lissajous_b
            )
        elif config.name == "pendulum":
            dataset = DoublePendulumDataset(
                size=config.size,
                dt=config.dt,
                history_length=config.history_length,
                sequence_length=config.sequence_length,
            )
        elif config.name == "pendulum_image":
            dataset = PixelPendulumDataset(
                size=config.size,
                dt=config.dt,
                history_length=config.history_length,
                sequence_length=config.sequence_length,
                image_size=config.image_size,
            )
        else:
            raise ValueError(f"Unknown dataset name: {config.name}")

        # Save to cache if enabled
        if config.use_cache:
            try:
                cache_path, config_dict = DatasetFactory.get_cache_path(config)
                print(f"[DatasetFactory] Saving dataset to cache: {cache_path}")
                torch.save(dataset.data, cache_path)

                # Save metadata as JSON for inspection
                json_path = cache_path.replace("dataset.pt", "metadata.json")
                with open(json_path, "w") as f:
                    json.dump(config_dict, f, indent=4)
                print(f"[DatasetFactory] Saved metadata to: {json_path}")

            except Exception as e:
                print(f"[DatasetFactory] Failed to save cache: {e}")

        return dataset
