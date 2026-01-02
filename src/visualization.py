import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_latent_reconstruction(
    encoder,
    decoder,
    save_path="reconstruction.png",
    mode="circle",
    spiral_loops=3,
    extra_loops=2,
    num_points=1000,
    history_length=3,
):
    """
    Generates a plot comparing ground truth points
    with their reconstructed counterparts.
    """
    encoder.eval()
    decoder.eval()

    # Generate ground truth points
    if mode == "spiral":
        max_angle = spiral_loops * 2 * np.pi
        cutoff_angle = max_angle - np.pi / 2
        visual_limit = cutoff_angle + extra_loops * 2 * np.pi

        test_angles = np.sqrt(np.linspace(0, 1, num_points)) * visual_limit
        r = test_angles / max_angle
        test_x = r * np.cos(test_angles)
        test_y = r * np.sin(test_angles)
        test_data = torch.tensor(np.stack([test_x, test_y], axis=1), dtype=torch.float32)

        mask_in = test_angles <= cutoff_angle
        mask_out = test_angles > cutoff_angle
        label_gt = "Ground Truth (Spiral)"

    elif mode == "lissajous":
        a, b = 3, 5
        max_angle = spiral_loops * 2 * np.pi
        cutoff_angle = max_angle - np.pi / 2
        visual_limit = cutoff_angle + extra_loops * 2 * np.pi

        test_angles = np.sqrt(np.linspace(0, 1, num_points)) * visual_limit
        r = test_angles / max_angle
        test_x = r * np.cos(a * test_angles)
        test_y = r * np.sin(b * test_angles)
        test_data = torch.tensor(np.stack([test_x, test_y], axis=1), dtype=torch.float32)

        mask_in = test_angles <= cutoff_angle
        mask_out = test_angles > cutoff_angle
        label_gt = "Ground Truth (Lissajous)"

    elif mode == "pendulum":
        from src.dataset import DoublePendulumDataset

        ds = DoublePendulumDataset(size=1, history_length=history_length)
        history = ds.history_length

        state = np.random.rand(4) * 2 * np.pi
        state[2:] *= 1.0

        traj = []
        for _ in range(num_points + history):
            theta1, theta2 = state[0], state[1]
            x1 = ds.L1 * np.sin(theta1)
            y1 = -ds.L1 * np.cos(theta1)
            x2 = x1 + ds.L2 * np.sin(theta2)
            y2 = y1 - ds.L2 * np.cos(theta2)
            # Store full state [x1, y1, x2, y2]
            traj.append([x1, y1, x2, y2])
            state = ds.rk4_step(state, ds.dt)

        traj = np.array(traj)  # Shape (N+k, 4)

        contexts = []
        for i in range(num_points):
            window = traj[i : i + history].flatten()
            contexts.append(window)
        contexts = np.array(contexts)

        # Input Tensor (N, 12)
        test_data = torch.tensor(contexts, dtype=torch.float32)

        # Plotting the tip (x2, y2)
        gt_plot_indices = np.arange(history - 1, history - 1 + num_points)
        gt_plot = traj[gt_plot_indices]
        test_x, test_y = gt_plot[:, 2], gt_plot[:, 3]  # x2, y2

        cutoff_index = len(test_x) // 2
        mask_in = np.arange(len(test_x)) <= cutoff_index
        mask_out = np.arange(len(test_x)) > cutoff_index
        label_gt = "Ground Truth (Chaos)"

    else:  # circle
        test_angles = np.linspace(0, 2 * np.pi, 200)
        test_x = np.cos(test_angles)
        test_y = np.sin(test_angles)
        test_data = torch.tensor(np.stack([test_x, test_y], axis=1), dtype=torch.float32)
        mask_in = np.ones_like(test_x, dtype=bool)
        mask_out = np.zeros_like(test_x, dtype=bool)
        label_gt = "Ground Truth (Circle)"

    # helper to check device
    device = next(encoder.parameters()).device
    test_data = test_data.to(device)

    with torch.no_grad():
        embeddings = encoder(test_data)
        reconstructed = decoder(embeddings).cpu().numpy()

    # If pendulum, reconstruction is (N, 12). Extract last frame (x2, y2) for plotting
    if mode == "pendulum":
        # [x1, y1, x2, y2] are last 4 elements. We want x2, y2 which are last 2.
        reconstructed = reconstructed[:, -2:]

    plt.figure(figsize=(6, 6))

    if mode == "pendulum":
        plt.plot(test_x, test_y, c="blue", alpha=0.3, linewidth=1, label=label_gt)
    else:
        plt.scatter(test_x, test_y, c="blue", alpha=0.3, s=1, label=label_gt)

    plt.scatter(reconstructed[mask_in, 0], reconstructed[mask_in, 1], c="green", s=2, label="Reconstruction (In-Dist)")
    if np.any(mask_out):
        plt.scatter(
            reconstructed[mask_out, 0],
            reconstructed[mask_out, 1],
            c="orange",
            s=2,
            label="Reconstruction (Extrapolation)",
        )

    plt.title(f"Reconstructing Latent Representation ({mode.title()})")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1))
    plt.grid(True)
    plt.axis("equal")

    print(f"Saving reconstruction plot to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def visualize_forecast(
    encoder, predictor, decoder, save_path="forecast.mp4", num_points=300, dt=0.05, history_length=3
):
    """
    Visualizes AUTOREGRESSIVE FORECASTING via a Side-by-Side Animation.
    Left: Ground Truth Physics
    Right: JEPA Latent Simulation
    """
    encoder.eval()
    predictor.eval()
    decoder.eval()

    # 1. Generate Ground Truth
    from src.dataset import DoublePendulumDataset

    ds = DoublePendulumDataset(size=1, history_length=history_length)
    history = ds.history_length

    # Start state
    state = np.random.rand(4) * 2 * np.pi
    state[2:] *= 1.0

    gt_traj = []
    # Simulate GT
    for _ in range(num_points + history):
        theta1, theta2 = state[0], state[1]
        x1 = ds.L1 * np.sin(theta1)
        y1 = -ds.L1 * np.cos(theta1)
        x2 = x1 + ds.L2 * np.sin(theta2)
        y2 = y1 - ds.L2 * np.cos(theta2)
        gt_traj.append([x1, y1, x2, y2])
        state = ds.rk4_step(state, ds.dt)

    gt_traj = np.array(gt_traj)  # (T, 4)

    # 2. Run Latent Forecast
    # Initial Window: frames 0, 1, 2
    x0_window = gt_traj[:history].flatten()
    x0 = torch.tensor(x0_window, dtype=torch.float32).unsqueeze(0)  # (1, 12)

    device = next(encoder.parameters()).device
    x0 = x0.to(device)

    pred_traj = []  # Stores (x1, y1, x2, y2)

    with torch.no_grad():
        z = encoder(x0)

        for _ in range(num_points):
            # Decode current state -> Returns window [t, t+1, t+2]
            # Last 4 elements are the frame at t+2
            x_hat_window = decoder(z).cpu().numpy().flatten()

            # Extract last frame (x1, y1, x2, y2)
            last_frame = x_hat_window[-4:]
            pred_traj.append(last_frame)

            # Next step in latent space
            z = predictor(z)

    pred_traj = np.array(pred_traj)  # (num_points, 4)

    # GT Trajectory aligned with prediction
    # Prediction starts after 'history' frames.
    # gt_traj[history] corresponds to pred_traj[0] (roughly)
    # Actually, if z0 encodes [0,1,2], z1 = P(z0) encodes [1,2,3].
    # Decoding z1 gives [1,2,3]. Last frame is 3.
    # So pred[0] is frame 3.
    # gt_traj index 3 is frame 3.
    # So we compare pred_traj[i] vs gt_traj[history + i]

    gt_comparison = gt_traj[history : history + num_points]

    # 3. Animate
    # Use explicit Figure and Canvas backend to avoid issues with global pyplot state
    # and ensure frames are captured correctly even if backend is not interactive
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 6))
    canvas = FigureCanvasAgg(fig)

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for ax in [ax1, ax2]:
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.grid(True)

    ax1.set_title("Ground Truth (Physics)")
    ax2.set_title("JEPA Forecast (Latent Dream)")

    # Line objects
    (line_gt,) = ax1.plot([], [], "o-", lw=2, color="blue")
    (trace_gt,) = ax1.plot([], [], "-", lw=1, alpha=0.3, color="blue")

    (line_pred,) = ax2.plot([], [], "o-", lw=2, color="red")
    (trace_pred,) = ax2.plot([], [], "-", lw=1, alpha=0.3, color="red")

    hist_gt_x, hist_gt_y = [], []
    hist_pred_x, hist_pred_y = [], []

    def init():
        line_gt.set_data([], [])
        trace_gt.set_data([], [])
        line_pred.set_data([], [])
        trace_pred.set_data([], [])
        return line_gt, trace_gt, line_pred, trace_pred

    def update(frame):
        # GT
        g = gt_comparison[frame]  # x1, y1, x2, y2
        line_gt.set_data([0, g[0], g[2]], [0, g[1], g[3]])

        hist_gt_x.append(g[2])  # Tip x2
        hist_gt_y.append(g[3])  # Tip y2
        if len(hist_gt_x) > 50:
            hist_gt_x.pop(0)
            hist_gt_y.pop(0)
        trace_gt.set_data(hist_gt_x, hist_gt_y)

        # Pred
        p = pred_traj[frame]
        line_pred.set_data([0, p[0], p[2]], [0, p[1], p[3]])

        hist_pred_x.append(p[2])
        hist_pred_y.append(p[3])
        if len(hist_pred_x) > 50:
            hist_pred_x.pop(0)
            hist_pred_y.pop(0)
        trace_pred.set_data(hist_pred_x, hist_pred_y)

        return line_gt, trace_gt, line_pred, trace_pred

    print(f"Generating animation ({num_points} frames)...")

    # Use imageio to write video
    import imageio

    # Create a temporary list to store frames
    frames = []

    # We need to manually iterate and capture
    # Re-initialize
    init()

    for i in range(num_points):
        update(i)

        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())
        # Must copy to avoid holding a reference to the buffer which gets overwritten
        image = image.copy()
        image = image[:, :, :3]

        frames.append(image)

    # Ensure extension is .mp4
    if not save_path.endswith(".mp4"):
        save_path = save_path.replace(".gif", ".mp4")

    print(f"Saving forecast animation to {save_path}")

    dirname = os.path.dirname(save_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    imageio.mimsave(save_path, frames, fps=30)


def visualize_image_reconstruction(
    encoder,
    decoder,
    dataset,
    save_path="reconstruction.png",
    num_samples=5,
    history_length=3,
):
    """
    Visualizes original vs reconstructed images.
    """
    encoder.eval()
    decoder.eval()
    device = next(encoder.parameters()).device

    # Get some samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    # Dataset returns (30, C, H, W) usually
    # We want to reconstruct the LAST frame of a context window
    # Context: [0, 1, 2] -> Predict 3?
    # Or just reconstruct context?
    # JEPA Decoder: Latent -> Input.
    # If Encoder encodes [0,1,2], Latent represents state at 2 (or 3?).
    # Decoder should reconstruct something.
    # In Trainer: `reconstruction = self.decoder(embedding)`. `loss(reconstruction, context)`.
    # So `reconstruction` must match `context`.
    # Whatever context is (flattened or reshaped).

    # Let's take [0..H] frames.

    fig, axes = plt.subplots(num_samples, 2, figsize=(5, 2.5 * num_samples))
    plt.tight_layout()

    for i, idx in enumerate(indices):
        seq = dataset[idx]  # (Seq, C, H, W)

        # Taking random t
        t = 0
        context_frames = seq[t : t + history_length]  # (H, 3, 64, 64) OR (H, 3, 32, 32)

        # Reshape for model: (1, H*C, H, W)
        H, C, Hei, Wid = context_frames.shape
        x_in = context_frames.reshape(1, H * C, Hei, Wid).to(device)

        with torch.no_grad():
            z = encoder(x_in)
            rec = decoder(z)  # (1, H*C, H, W)

        # We visualize the LAST frame of the context
        # Img (C, H, W). Need (H, W, C) for matplotlib.

        # Original Last Frame
        orig_img = context_frames[-1].permute(1, 2, 0).numpy()  # (H, W, 3)

        # Reconstructed Last Frame
        rec_cpu = rec.cpu().reshape(H, C, Hei, Wid)
        rec_img = rec_cpu[-1].permute(1, 2, 0).numpy()  # (H, W, 3)

        ax_orig = axes[i, 0] if num_samples > 1 else axes[0]
        ax_rec = axes[i, 1] if num_samples > 1 else axes[1]

        ax_orig.imshow(orig_img)
        ax_orig.set_title("Original")
        ax_orig.axis("off")

        ax_rec.imshow(rec_img)
        ax_rec.set_title("Reconstruction")
        ax_rec.axis("off")

    print(f"Saving image reconstruction plot to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def visualize_image_forecast(
    encoder,
    predictor,
    decoder,
    dataset,
    save_path="forecast.mp4",
    num_frames=300,
    history_length=3,
    image_size=64,
):
    """
    Autoregressive forecasting for image-based JEPA model.
    Shows ground truth vs predicted frames side-by-side.
    """
    encoder.eval()
    predictor.eval()
    decoder.eval()
    device = next(encoder.parameters()).device

    # Import physics simulator for ground truth
    from src.dataset import DoublePendulumDataset

    # Create a fresh physics simulation for ground truth
    # Use same parameters as dataset
    physics_sim = DoublePendulumDataset(size=1, history_length=history_length)

    # Random initial state
    state = np.random.rand(4) * 2 * np.pi
    state[2:] *= 1.0  # Scale velocities

    # Simulate physics for initial context + num_frames
    total_frames = history_length + num_frames
    gt_coords = []

    for _ in range(total_frames):
        theta1, theta2 = state[0], state[1]
        x1 = physics_sim.L1 * np.sin(theta1)
        y1 = -physics_sim.L1 * np.cos(theta1)
        x2 = x1 + physics_sim.L2 * np.sin(theta2)
        y2 = y1 - physics_sim.L2 * np.cos(theta2)
        gt_coords.append([x1, y1, x2, y2])
        state = physics_sim.rk4_step(state, physics_sim.dt)

    gt_coords = np.array(gt_coords)

    # Render GT frames as images
    # Create a temporary renderer (reuse PixelPendulumDataset logic)
    def render_pendulum_frame(x1, y1, x2, y2, img_size):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        cx, cy = img_size // 2, img_size // 2
        scale = (img_size / 2) / 2.2

        px0, py0 = cx, cy
        px1, py1 = cx + x1 * scale, cy - y1 * scale
        px2, py2 = cx + x2 * scale, cy - y2 * scale

        draw.line([(px0, py0), (px1, py1)], fill=(0, 255, 255), width=2)
        draw.line([(px1, py1), (px2, py2)], fill=(255, 0, 255), width=2)

        r = 2
        draw.ellipse([px1 - r, py1 - r, px1 + r, py1 + r], fill=(255, 255, 0))
        draw.ellipse([px2 - r, py2 - r, px2 + r, py2 + r], fill=(255, 0, 0))

        img_np = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return tensor

    # Render all GT frames
    gt_frames_list = []
    for coord in gt_coords:
        frame = render_pendulum_frame(coord[0], coord[1], coord[2], coord[3], image_size)
        gt_frames_list.append(frame)

    gt_frames_tensor = torch.stack(gt_frames_list)  # (total_frames, C, H, W)

    # Extract initial context for JEPA
    context_frames = gt_frames_tensor[:history_length]  # (H, C, H, W)
    H, C, Hei, Wid = context_frames.shape

    # Encode initial context
    x0 = context_frames.reshape(1, H * C, Hei, Wid).to(device)

    # Run autoregressive prediction
    pred_frames = []

    with torch.no_grad():
        z = encoder(x0)

        for _ in range(num_frames):
            # Predict next latent state
            z = predictor(z)

            # Decode to image
            x_hat = decoder(z)  # (1, H*C, H, W)

            # Extract last frame
            x_hat_reshaped = x_hat.reshape(H, C, Hei, Wid)
            last_frame = x_hat_reshaped[-1]  # (C, H, W)

            pred_frames.append(last_frame.cpu())

    # Convert to numpy for visualization
    pred_frames = torch.stack(pred_frames).permute(0, 2, 3, 1).numpy()  # (T, H, W, C)

    # GT frames (skip initial context)
    gt_frames = gt_frames_tensor[history_length:].permute(0, 2, 3, 1).numpy()  # (T, H, W, C)

    # Use minimum length for safety
    min_len = min(len(gt_frames), len(pred_frames))
    gt_frames = gt_frames[:min_len]
    pred_frames = pred_frames[:min_len]

    # Create side-by-side animation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 5))
    canvas = FigureCanvasAgg(fig)

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.set_title("Ground Truth")
    ax2.set_title("JEPA Forecast")

    for ax in [ax1, ax2]:
        ax.axis("off")

    frames = []

    print(f"Generating forecast animation ({min_len} frames)...")

    for i in range(min_len):
        ax1.clear()
        ax2.clear()

        ax1.imshow(gt_frames[i])
        ax1.set_title("Ground Truth")
        ax1.axis("off")

        ax2.imshow(pred_frames[i])
        ax2.set_title("JEPA Forecast")
        ax2.axis("off")

        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())
        image = image.copy()[:, :, :3]
        frames.append(image)

    # Save video
    import imageio

    if not save_path.endswith(".mp4"):
        save_path = save_path.replace(".gif", ".mp4")

    print(f"Saving forecast animation to {save_path}")

    dirname = os.path.dirname(save_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    imageio.mimsave(save_path, frames, fps=30)
