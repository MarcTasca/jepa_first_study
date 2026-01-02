import os
import sys

import imageio
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mlp_baseline import MLPBaseline  # noqa: E402
from src.dataset import DoublePendulumDataset  # noqa: E402
from src.models import Decoder, Encoder, Predictor  # noqa: E402


def main():
    # Setup
    history_length = 2
    dt = 0.05
    num_points = 500
    save_path = "results/comparison_forecast.mp4"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Loading Models...")

    # 1. Load JEPA
    encoder = Encoder(input_dim=4 * history_length).to(device)
    predictor = Predictor().to(device)
    decoder = Decoder(output_dim=4 * history_length).to(device)

    try:
        encoder.load_state_dict(torch.load("models/encoder.pth", map_location=device))
        predictor.load_state_dict(torch.load("models/predictor.pth", map_location=device))
        decoder.load_state_dict(torch.load("models/decoder.pth", map_location=device))
    except FileNotFoundError:
        print("Error: JEPA models not found. Run main.py first.")
        return

    # 2. Load MLP Baseline
    mlp = MLPBaseline(input_dim=history_length * 4).to(device)
    try:
        mlp.load_state_dict(torch.load("models/mlp_baseline.pth", map_location=device))
    except FileNotFoundError:
        print("Error: MLP Baseline model not found. Run mlp_baseline.py first.")
        return

    encoder.eval()
    predictor.eval()
    decoder.eval()
    mlp.eval()

    print("Generating Joint Forecast...")

    # --- Simulation Setup ---
    ds = DoublePendulumDataset(size=1, history_length=history_length, dt=dt)

    # Random Initial State
    state = np.random.rand(4) * 2 * np.pi
    state[2:] *= 1.0  # Scale velocity
    # state = np.array([np.pi/2, np.pi/2, 0, 0]) # High energy standard star

    ground_truth = []

    # Warmup (Get initial frames)
    for _ in range(history_length):
        t1, t2 = state[0], state[1]
        x1 = np.sin(t1)
        y1 = -np.cos(t1)
        x2 = x1 + np.sin(t2)
        y2 = y1 - np.cos(t2)
        ground_truth.append([x1, y1, x2, y2])
        state = ds.rk4_step(state, dt)

    # Initialize Forecasts with Ground Truth Warmup
    jepa_predictions = list(ground_truth)
    mlp_predictions = list(ground_truth)

    # We need latent state for JEPA
    # Extract last H frames from warmup
    initial_context_frames = np.array(ground_truth[-history_length:])  # (H, 4)
    initial_context_tensor = torch.tensor(initial_context_frames.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

    # Encode Initial State for JEPA
    with torch.no_grad():
        s_curr = encoder(initial_context_tensor)

    # --- Autoregressive Loops ---
    # We run them step-by-step together for convenience (or separately, doesn't matter)

    print(f"Simulating {num_points} steps...")

    with torch.no_grad():
        for _ in range(num_points):
            # 1. Ground Truth Step
            t1, t2 = state[0], state[1]
            x1 = np.sin(t1)
            y1 = -np.cos(t1)
            x2 = x1 + np.sin(t2)
            y2 = y1 - np.cos(t2)
            ground_truth.append([x1, y1, x2, y2])
            state = ds.rk4_step(state, dt)

            # 2. JEPA Step
            # Predict next latent state
            s_curr = predictor(s_curr)
            # Decode to get observation (for visualization/next set, BUT JEPA works in latent!)
            # Wait, purely latent JEPA autoregression means we rely on s_curr.
            # But we need to visualize it!
            recon = decoder(s_curr)  # (1, H*4)
            # The decoder outputs a sequence of H frames. We care about the last one.
            # Or does the decoder reconstruct the *current* window?
            # Decoder(z_{t+1}) -> recreates window [t+1 ... t+1+H] ??
            # No, based on training: Context [t .. t+H] -> Enc -> Pred -> z' -> Dec -> Target [t+1 .. t+1+H]
            # So Decoder(z) outputs the WINDOW at that time.
            # The "next step" is the LAST frame of that window.
            # actually, let's just take the last frame of the reconstructed window as the "new" frame.

            recon_frames = recon.cpu().numpy().reshape(history_length, 4)
            # We append the NEWEST frame (the one at the end of the window)
            # Careful: The window shifts by 1. The last frame is the new one.
            jepa_predictions.append(recon_frames[-1])

            # 3. MLP Step
            # MLP inputs last H frames of *its own* past predictions
            mlp_context_frames = np.array(mlp_predictions[-history_length:])  # (H, 4)
            mlp_context_tensor = torch.tensor(mlp_context_frames.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
            mlp_next_step = mlp(mlp_context_tensor)  # (1, 4)
            mlp_predictions.append(mlp_next_step.cpu().numpy()[0])

    # --- Rendering ---
    print("Rendering Video...")
    frames = []
    fig = Figure(figsize=(8, 8))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)

    gt_data = np.array(ground_truth)
    jepa_data = np.array(jepa_predictions)
    mlp_data = np.array(mlp_predictions)

    # Trace buffers
    trace_len = 50
    h_gt_x, h_gt_y = [], []
    h_jepa_x, h_jepa_y = [], []
    h_mlp_x, h_mlp_y = [], []

    for i in range(len(ground_truth) - num_points, len(ground_truth)):
        ax.clear()
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Forecast Comparison (Step {i})")

        # Ground Truth (Black)
        x1, y1, x2, y2 = gt_data[i]
        h_gt_x.append(x2)
        h_gt_y.append(y2)
        if len(h_gt_x) > trace_len:
            h_gt_x.pop(0)
            h_gt_y.pop(0)

        ax.plot(h_gt_x, h_gt_y, "k-", lw=1, alpha=0.2)  # Trace
        ax.plot([0, x1], [0, y1], "k-", lw=3, alpha=0.3, label="Ground Truth")  # Ghostly
        ax.plot([x1, x2], [y1, y2], "k-", lw=3, alpha=0.3)
        ax.plot(x2, y2, "ko", ms=8, alpha=0.3)

        # JEPA (Blue)
        jx1, jy1, jx2, jy2 = jepa_data[i]
        h_jepa_x.append(jx2)
        h_jepa_y.append(jy2)
        if len(h_jepa_x) > trace_len:
            h_jepa_x.pop(0)
            h_jepa_y.pop(0)

        ax.plot(h_jepa_x, h_jepa_y, "b-", lw=1, alpha=0.4)  # Trace
        ax.plot([0, jx1], [0, jy1], "b-", lw=2, label="JEPA (Latent)")
        ax.plot([jx1, jx2], [jy1, jy2], "b-", lw=2)
        ax.plot(jx2, jy2, "bo", ms=6)

        # MLP (Red)
        mx1, my1, mx2, my2 = mlp_data[i]
        h_mlp_x.append(mx2)
        h_mlp_y.append(my2)
        if len(h_mlp_x) > trace_len:
            h_mlp_x.pop(0)
            h_mlp_y.pop(0)

        ax.plot(h_mlp_x, h_mlp_y, "r-", lw=1, alpha=0.4)  # Trace
        ax.plot([0, mx1], [0, my1], "r-", lw=2, label="MLP (Baseline)")
        ax.plot([mx1, mx2], [my1, my2], "r-", lw=2)
        ax.plot(mx2, my2, "ro", ms=6)

        ax.legend(loc="upper right")

        canvas.draw()
        img = np.asarray(canvas.buffer_rgba()).copy()
        frames.append(img[:, :, :3])

    imageio.mimsave(save_path, frames, fps=30)
    print(f"Comparison video saved to {save_path}")


if __name__ == "__main__":
    main()
