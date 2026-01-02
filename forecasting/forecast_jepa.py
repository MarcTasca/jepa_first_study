import os
import sys

import torch

# Add parent directory to path to allow imports from src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models import Decoder, Encoder, Predictor  # noqa: E402
from src.visualization import visualize_forecast  # noqa: E402


def main():
    # Setup
    history_length = 2

    num_points = 1000
    save_path = "results/forecast_loaded.mp4"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Loading JEPA Models...")

    # Initialize Models (must match training architecture!)
    encoder = Encoder(input_dim=4 * history_length).to(device)
    predictor = Predictor().to(device)
    decoder = Decoder(output_dim=4 * history_length).to(device)

    # Load Weights
    try:
        encoder.load_state_dict(torch.load("models/encoder.pth", map_location=device))
        predictor.load_state_dict(torch.load("models/predictor.pth", map_location=device))
        decoder.load_state_dict(torch.load("models/decoder.pth", map_location=device))
        print("Models loaded successfully.")
    except FileNotFoundError:
        print("Error: Model files not found in 'models/'. Run main.py first.")
        return

    # Generate Forecast
    print(f"Generating Forecast Animation ({num_points} frames)...")
    visualize_forecast(
        encoder, predictor, decoder, save_path=save_path, num_points=num_points, history_length=history_length
    )
    print(f"Done! Saved to {save_path}")


if __name__ == "__main__":
    main()
