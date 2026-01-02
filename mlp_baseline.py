import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from src.dataset import DoublePendulumDataset
from src.visualization import visualize_forecast # Reuse this if possible, or adapt

class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def train_baseline():
    # Setup
    history_length = 2
    dt = 0.05
    size = 100000
    batch_size = 4096
    epochs = 100
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    print("Initializing Baseline (Supervised MLP)...")
    dataset = DoublePendulumDataset(size=size, history_length=history_length, dt=dt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model (Input: History*4 -> Output: 4)
    # Note: Dataset returns (Context, Target). 
    # Context is (History, 4). Target is (History, 4) shifted by 1.
    # We only want to predict the *last* step of the target given the context?
    # Or autoregressively? 
    # Let's match JEPA: Predict next step given History.
    # JEPA Predictor: z(t) -> z(t+1).
    # Baseline: x(t-H...t) -> x(t+1).
    
    model = MLPBaseline(input_dim=history_length*4, output_dim=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"Starting Baseline Training for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for context, target in dataloader:
            # context shape: (B, H*4)
            # target shape: (B, H*4). Target contains [t_1, ..., t_H+1].
            # We want to predict t_H+1 given [t_1, ... t_H] (which is context).
            # The DoublePendulumDataset returns flatted vectors.
            # Context: [x(0)...x(H-1)]
            # Target:  [x(1)...x(H)]
            # So Target's LAST 4 elements are the "next step" x(H).
            
            context = context.to(device)
            target = target.to(device)
            
            # Extract only the last frame (4 values) from target as the label
            # Target is flattened (B, History*4). We want the last 4.
            next_step_label = target[:, -4:] 
            
            pred = model(context)
            loss = criterion(pred, next_step_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Baseline Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
        scheduler.step(avg_loss)

    # Verification / Forecast
    # We need a custom forecast loop for the baseline since visualize_forecast expects (enc, pred, dec).
    print("Generating Baseline Forecast...")
    generate_baseline_video(model, history_length, dt, device)

def generate_baseline_video(model, history_length, dt, device, save_path='figures/baseline_forecast.mp4'):
    import imageio
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    # Simulation for Ground Truth
    # We create a new dataset instance just to get a fresh trajectory
    ds = DoublePendulumDataset(size=1, history_length=history_length, dt=dt)
    # We actually need a LONG simulation for the video (e.g. 1000 steps)
    # But the dataset only gives short chunks.
    # We'll simulate manually or hack the dataset.
    # Let's manually simulate a test trajectory.
    
    # Initial State
    state = np.array([np.pi/2, np.pi/2, 0, 0]) # High energy
    
    ground_truth = []
    # Warmup
    for _ in range(history_length):
        t1, t2 = state[0], state[1]
        x1 = np.sin(t1); y1 = -np.cos(t1)
        x2 = x1 + np.sin(t2); y2 = y1 - np.cos(t2)
        ground_truth.append([x1, y1, x2, y2])
        state = ds.rk4_step(state, dt)
        
    predictions = list(ground_truth) # Initialize with warmup
    
    # Autoregressive Loop
    num_points = 300
    model.eval()
    with torch.no_grad():
        for _ in range(num_points):
            # Ground Truth Update
            t1, t2 = state[0], state[1]
            x1 = np.sin(t1); y1 = -np.cos(t1)
            x2 = x1 + np.sin(t2); y2 = y1 - np.cos(t2)
            ground_truth.append([x1, y1, x2, y2])
            state = ds.rk4_step(state, dt)
            
            # Model Prediction
            # Context is the last 'history_length' frames of PREDICTIONS
            context_frames = np.array(predictions[-history_length:]) # (H, 4)
            context_tensor = torch.tensor(context_frames.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
            
            next_step = model(context_tensor) # (1, 4)
            predictions.append(next_step.cpu().numpy()[0])

    # Rendering
    frames = []
    fig = Figure(figsize=(10, 5))
    canvas = FigureCanvasAgg(fig)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Static limits
    for ax in [ax1, ax2]:
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    ax1.set_title("Ground Truth (Physics)")
    ax2.set_title("Baseline MLP Forecast")

    gt_data = np.array(ground_truth)
    pred_data = np.array(predictions)

    for i in range(len(ground_truth) - num_points, len(ground_truth)): # Show last num_points
        # Reset backgrounds
        ax1.lines.clear(); ax2.lines.clear(); ax1.patches.clear(); ax2.patches.clear()
        
        # GT
        x1, y1, x2, y2 = gt_data[i]
        ax1.plot([0, x1], [0, y1], 'k-', lw=2)
        ax1.plot([x1, x2], [y1, y2], 'k-', lw=2)
        ax1.plot(x1, y1, 'ro', ms=5)
        ax1.plot(x2, y2, 'bo', ms=5)
        
        # Pred
        px1, py1, px2, py2 = pred_data[i]
        ax2.plot([0, px1], [0, py1], 'k-', lw=2)
        ax2.plot([px1, px2], [py1, py2], 'k-', lw=2)
        ax2.plot(px1, py1, 'ro', ms=5)
        ax2.plot(px2, py2, 'bo', ms=5)
        
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba()).copy()
        frames.append(img[:,:,:3])

    imageio.mimsave(save_path, frames, fps=30)
    print(f"Baseline video saved to {save_path}")

if __name__ == "__main__":
    train_baseline()
