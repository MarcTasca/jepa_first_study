import torch
import numpy as np
from typing import Tuple

class TrajectoryDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset simulating a point moving on a circle.
    context: (x, y) at time t
    target: (x, y) at time t+1
    """
    def __init__(self, size=2000, sequence_length=10):
        self.size = size
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

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        traj = self.data[index]
        context = traj[0]  # (2,)
        target = traj[1]   # (2,)
        return context, target


class SpiralTrajectoryDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset simulating a point moving on a spiral.
    Non-linear manifold: r grows with theta.
    """
    def __init__(self, size=2000, sequence_length=10, loops=3):
        self.size = size
        self.data = []
        # Random start angles 
        # We want the spiral to have multiple loops, e.g., 0 to 3*2pi
        max_angle = loops * 2 * np.pi
        # Use sqrt sampling for uniform arc-length distribution: theta ~ sqrt(U)
        start_angles = np.sqrt(np.random.rand(size)) * (max_angle - np.pi/2)
        
        for i in range(size):
            angles = start_angles[i] + np.linspace(0, np.pi/2, sequence_length) # Short segments
            
            # r = theta / max_angle (normalized radius 0 to 1)
            r = angles / max_angle
            
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            trajectory = np.stack([x, y], axis=1)
            self.data.append(trajectory)
            
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
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
    def __init__(self, size=100000, loops=3, a=3, b=5, sequence_length=10):
        self.size = size
        self.data = []
        max_angle = loops * 2 * np.pi
        
        # Sqrt sampling for roughly uniform density along the expanding curve
        start_angles = np.sqrt(np.random.rand(size)) * (max_angle - np.pi/2)
        
        for i in range(size):
            angles = start_angles[i] + np.linspace(0, np.pi/10, sequence_length) # Smaller step for higher freq
            
            r = angles / max_angle
            x = r * np.cos(a * angles)
            y = r * np.sin(b * angles)
            
            trajectory = np.stack([x, y], axis=1)
            self.data.append(trajectory)
            
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
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
    def __init__(self, size=100000, dt=0.05, history_length=3):
        self.size = size
        self.dt = dt
        self.history_length = history_length
        self.data = []
        
        # Physics Constants
        self.L1, self.L2 = 1.0, 1.0  # Lengths
        self.m1, self.m2 = 1.0, 1.0  # Masses
        self.g = 9.81
        
        # Pre-generate random initial states [theta1, theta2, omega1, omega2]
        # High energy states to ensure chaos
        init_states = np.random.rand(size, 4) * 2 * np.pi
        init_states[:, 2:] *= 1.0 # Lower initial velocity for swinging chaos
        
        # Required simulation steps: history_length + 1 (for target)
        sim_steps = history_length + 1
        
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
                
                # Normalize roughly to [-1, 1]
                # x1, y1 are naturally in [-1, 1] (L1=1)
                # x2, y2 can reach ~2 (L1+L2)
                # scale removed as requested by user
                trajectory.append([x1, y1, x2, y2])
                
                # Integrate next step
                state = self.rk4_step(state, self.dt)
                
            traj = np.array(trajectory) # Shape (sim_steps, 2)
            
            # Context: frames [0, ..., history-1]
            context = traj[:self.history_length].flatten()
            
            # Target: frames [1, ..., history] (Shifted by 1 time step)
            target = traj[1:self.history_length+1].flatten()
            
            self.data.append([context, target])
            
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)

    def derivs(self, state):
        t1, t2, w1, w2 = state
        
        delta = t2 - t1
        den1 = (self.m1 + self.m2) * self.L1 - self.m2 * self.L1 * np.cos(delta) * np.cos(delta)
        den2 = (self.L2/self.L1) * den1

        d_t1 = w1
        d_t2 = w2
        
        d_w1 = (self.m2 * self.L1 * w1 * w1 * np.sin(delta) * np.cos(delta) +
                self.m2 * self.g * np.sin(t2) * np.cos(delta) +
                self.m2 * self.L2 * w2 * w2 * np.sin(delta) -
                (self.m1 + self.m2) * self.g * np.sin(t1)) / den1
                
        d_w2 = (-self.m2 * self.L2 * w2 * w2 * np.sin(delta) * np.cos(delta) +
                (self.m1 + self.m2) * self.g * np.sin(t1) * np.cos(delta) -
                (self.m1 + self.m2) * self.L1 * w1 * w1 * np.sin(delta) -
                (self.m1 + self.m2) * self.g * np.sin(t2)) / den2
                
        return np.array([d_t1, d_t2, d_w1, d_w2])

    def rk4_step(self, state, dt):
        k1 = self.derivs(state)
        k2 = self.derivs(state + dt * k1 / 2)
        k3 = self.derivs(state + dt * k2 / 2)
        k4 = self.derivs(state + dt * k3)
        return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        traj = self.data[index]
        context = traj[0]
        target = traj[1]
        return context, target
