"""
Baseline models for comparison: SimCLR and VICReg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    """SimCLR baseline implementation"""
    def __init__(self, latent_dim=32):
        super().__init__()
        
        # Encoder (same architecture as LePhyJEPA for fair comparison)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, view1, view2, rgb=None):
        z1 = self.encoder(view1)
        z2 = self.encoder(view2)
        
        # Project and normalize
        z1_proj = F.normalize(self.projection(z1), dim=1)
        z2_proj = F.normalize(self.projection(z2), dim=1)
        
        # SimCLR loss: negative cosine similarity
        loss = 2 - 2 * (z1_proj * z2_proj).sum(dim=-1).mean()
        
        return {
            "total": loss,
            "jepa": loss,
            "physics": torch.tensor(0.0, device=view1.device),
            "energy": torch.tensor(0.0, device=view1.device),
            "z1": z1,
            "z2": z2
        }
    
    def encode(self, x):
        """Get representations without projection"""
        return self.encoder(x)


class VICReg(nn.Module):
    """VICReg baseline implementation"""
    def __init__(self, latent_dim=32, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        
        # Encoder (same architecture)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
    
    def forward(self, view1, view2, rgb=None):
        z1 = self.encoder(view1)
        z2 = self.encoder(view2)
        
        # Invariance loss (similarity)
        sim_loss = F.mse_loss(z1, z2)
        
        # Variance loss (std regularization)
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        
        # Covariance loss (decorrelation)
        def cov_loss(z):
            z = z - z.mean(dim=0)
            cov = (z.T @ z) / (z.size(0) - 1)
            return cov.fill_diagonal_(0).pow_(2).sum() / z.size(1)
        
        cov_loss_z = cov_loss(z1) + cov_loss(z2)
        
        # Total loss
        loss = (
            self.sim_coeff * sim_loss +
            self.std_coeff * std_loss +
            self.cov_coeff * cov_loss_z
        )
        
        return {
            "total": loss,
            "jepa": sim_loss,
            "physics": torch.tensor(0.0, device=view1.device),
            "energy": torch.tensor(0.0, device=view1.device),
            "z1": z1,
            "z2": z2
        }
    
    def encode(self, x):
        return self.encoder(x)


class BaselineTrainer:
    """Utility to train baselines with same interface"""
    @staticmethod
    def create_baseline(name, **kwargs):
        if name.lower() == "simclr":
            return SimCLR(**kwargs)
        elif name.lower() == "vicreg":
            return VICReg(**kwargs)
        else:
            raise ValueError(f"Unknown baseline: {name}")
