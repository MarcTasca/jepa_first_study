import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import numpy as np

class JEPATrainer:
    """
    Handles the training of JEPA components (Encoder, Predictor) 
    and the separate Decoder.
    """
    def __init__(self, encoder, predictor, decoder, dataloader, lr=1e-3, device='cpu'):
        self.encoder = encoder.to(device)
        self.predictor = predictor.to(device)
        self.decoder = decoder.to(device)
        self.dataloader = dataloader
        self.device = device
        self.lr = lr
        
        # Target Encoder is a copy of Encoder, updated via EMA, NO gradients
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
    def train_jepa(self, epochs=50):
        """
        Train the Joint Embedding Predictive Architecture.
        Obsertvation(t) -> Encoder -> z(t) -> Predictor -> z(t+1)
        Observation(t+1) -> TargetEncoder -> target_z(t+1)
        Loss = MSE(z(t+1), target_z(t+1))
        """
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()), 
            lr=self.lr
        )
        criterion = nn.MSELoss()
        ema_decay = 0.99
        
        print(f"Starting JEPA Training for {epochs} epochs...")
        self.encoder.train()
        self.predictor.train()
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        for epoch in range(epochs):
            # Dynamic EMA-LR Coupling
            # EMA = 1.0 - k * LR
            # If LR=1e-3, EMA=0.95. If LR drops, EMA increases towards 1.0.
            current_lr = optimizer.param_groups[0]['lr']
            ema_decay = 1.0 - (50.0 * current_lr)
            # Clip for safety
            ema_decay = max(0.9, min(1.0, ema_decay))
            
            start_time = time.time()
            epoch_loss = 0
            for context, target in self.dataloader:
                context, target = context.to(self.device), target.to(self.device)
                
                # Forward Pass
                s_context = self.encoder(context)
                with torch.no_grad():
                    s_target = self.target_encoder(target)
                s_pred = self.predictor(s_context)
                
                # Loss & Backprop
                loss = criterion(s_pred, s_target)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.predictor.parameters()), 1.0)
                optimizer.step()
                
                # EMA Update for Target Encoder
                with torch.no_grad():
                    for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                        param_k.data = param_k.data * ema_decay + param_q.data * (1. - ema_decay)
                        
                epoch_loss += loss.item()
            
            end_time = time.time()
            epoch_duration = end_time - start_time
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"JEPA Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f} | Time: {epoch_duration:.2f}s | LR: {current_lr:.2e} | EMA: {ema_decay:.5f}")
            scheduler.step(avg_loss)
                
    def train_decoder(self, epochs=50):
        """
        Train the Decoder to map Latent -> Observation.
        Encoder is FROZEN during this phase.
        """
        optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        print(f"\nStarting Decoder Training for {epochs} epochs...")
        self.decoder.train()
        self.encoder.eval() # Freeze encoder
        
        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0
            for context, _ in self.dataloader:
                context = context.to(self.device)
                
                with torch.no_grad():
                    embedding = self.encoder(context)
                
                reconstruction = self.decoder(embedding)
                loss = criterion(reconstruction, context)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            end_time = time.time()
            epoch_duration = end_time - start_time
            avg_loss = epoch_loss / len(self.dataloader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Decoder Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f} | Time: {epoch_duration:.2f}s | LR: {current_lr:.2e}")
            scheduler.step(avg_loss)

