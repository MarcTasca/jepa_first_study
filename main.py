import torch
import numpy as np
from src.dataset import TrajectoryDataset, SpiralTrajectoryDataset, LissajousTrajectoryDataset, DoublePendulumDataset
from src.models import Encoder, Predictor, Decoder
from src.trainer import JEPATrainer
from src.visualization import visualize_latent_reconstruction, visualize_forecast

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Initializing Project (Chaos Mode: Double Pendulum)...")

    # 0. Setup Environment
    history_length=2
    dt=0.05
    size=200000
    batch_size=4096
    jepa_epochs=500
    decoder_epochs=500
    lr=1e-4
    mode='pendulum'
    num_points=1000
    
    # 1. Prepare Data
    # Dataset size is slightly smaller because simulation is expensive
    dataset = DoublePendulumDataset(size=size, history_length=history_length, dt=dt) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    # 2. Initialize Models
    # Input dim = 12 (3 frames x 4 coordinates: x1,y1,x2,y2)
    encoder = Encoder(input_dim=4*history_length)
    predictor = Predictor()
    decoder = Decoder(output_dim=4*history_length) # For verification only
    
    # 3. Setup Trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    trainer = JEPATrainer(encoder, predictor, decoder, dataloader, lr=lr, device=device)
    
    # 4. Train JEPA (Self-Supervised Phase)
    trainer.train_jepa(epochs=jepa_epochs)
    
    # 5. Train Decoder (Verification Phase)
    trainer.train_decoder(epochs=decoder_epochs)
    
    # 6. Visualize
    visualize_latent_reconstruction(encoder, decoder, save_path='figures/pendulum_reconstruction.png', mode=mode, num_points=num_points, history_length=history_length)
    
    # 7. Forecast (The Real Test)
    # This tests the predictor's ability to simulate the physics autoregressively
    visualize_forecast(encoder, predictor, decoder, save_path='figures/pendulum_forecast.mp4', num_points=num_points, history_length=history_length)
    
    print("Done! Check 'figures/pendulum_reconstruction.png' and 'figures/pendulum_forecast.mp4'.")

if __name__ == "__main__":
    main()
