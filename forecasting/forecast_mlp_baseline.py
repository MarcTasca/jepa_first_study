import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mlp_baseline import MLPBaseline, generate_baseline_video  # noqa: E402


def main():
    # Setup
    history_length = 2
    dt = 0.05
    save_path = "results/forecast_mlp_loaded.mp4"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Loading MLP Baseline Model...")

    # Initialize Model
    model = MLPBaseline(input_dim=history_length * 4).to(device)

    # Load Weights
    try:
        model.load_state_dict(torch.load("models/mlp_baseline.pth", map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file 'models/mlp_baseline.pth' not found. Run mlp_baseline.py first.")
        return

    # Generate Forecast
    print("Generating MLP Forecast Animation...")
    # Using the function from mlp_baseline.py which handles the simulation and video generation
    generate_baseline_video(model, history_length, dt, device, save_path=save_path)
    print(f"Done! Saved to {save_path}")


if __name__ == "__main__":
    main()
