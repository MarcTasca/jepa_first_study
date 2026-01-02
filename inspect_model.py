import torch


def inspect():
    # Load latest run
    import glob
    import os

    # specific run or latest
    # Assuming we are in root
    runs = sorted(glob.glob("results/pendulum_image_*"), reverse=True)
    latest_run = None

    for r in runs:
        if os.path.exists(f"{r}/models/encoder.pth"):
            latest_run = r
            break

    if not latest_run:
        print("No completed runs found with saved models!")
        return

    print(f"Inspecting run: {latest_run}")

    # Load config logic (simplified)
    # We'll just instantiate Runner which loads models
    # but we need to trick argparse or manually setup
    # actually Runner needs args.
    # Let's just load weights manually to be safe/fast

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Device: {device}")

    # Load Encoder
    from src.models import VisionEncoder

    encoder = VisionEncoder(input_channels=6, embedding_dim=32, image_size=64).to(device)
    encoder.load_state_dict(torch.load(f"{latest_run}/models/encoder.pth", map_location=device))

    # Load one batch of data
    from src.dataset import DatasetConfig, DatasetFactory

    cfg = DatasetConfig(name="pendulum_image", size=100, image_size=64)
    # This might trigger render if not cached, but should be cached
    dataset = DatasetFactory.get_dataset(cfg)

    # Get a batch
    batch = dataset.data[:32].to(device)  # (32, Seq, 3, 64, 64)

    # Extract context (history=2)
    # (B, H, C, W, H) -> (B, H*C, W, H)
    context_frames = batch[:, :2]
    B, _, C, H, W = context_frames.shape
    context = context_frames.reshape(B, -1, H, W)

    # Forward
    encoder.eval()
    with torch.no_grad():
        z = encoder(context)  # (B, 32)

    # Calculate stats
    var = torch.var(z, dim=0)
    std = torch.sqrt(var)
    mean_std = std.mean().item()

    print(f"\nEmbedding Stats (Batch size {B}):")
    print(f"Mean Std Dev: {mean_std:.6f}")
    print(f"Min Std Dev:  {std.min().item():.6f}")
    print(f"Max Std Dev:  {std.max().item():.6f}")

    if mean_std < 0.01:
        print("\n❌ COLLAPSED")
    else:
        print("\n✅ HEALTHY (Variance exists!)")

    print("\nSample embeddings (first 3 dims of first 3 samples):")
    print(z[:3, :3])


if __name__ == "__main__":
    inspect()
