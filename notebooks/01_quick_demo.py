#!/usr/bin/env python
"""
Quick demo notebook (convertible to Jupyter notebook)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Markdown

print("🚀 LePhyJEPA Quick Demo")
print("="*60)

# 1. Import models
from models.lephyjepa_core import LePhyJEPA
from models.baselines import SimCLR, VICReg
from data.synthetic_generator import SyntheticNYUv2
from models.theorem_verifier import TheoremVerifier

# 2. Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Create synthetic data
print("\n📊 Creating synthetic dataset...")
dataset = SyntheticNYUv2(num_samples=10)
sample = dataset[0]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(sample["rgb"].permute(1, 2, 0).numpy())
axes[0].set_title("RGB Image")
axes[0].axis('off')

axes[1].imshow(sample["depth"].squeeze().numpy(), cmap='viridis')
axes[1].set_title("Depth Map")
axes[1].axis('off')

axes[2].imshow(sample["view1"].permute(1, 2, 0).numpy())
axes[2].set_title("Augmented View 1")
axes[2].axis('off')

plt.tight_layout()
plt.show()

# 4. Create models
print("\n🧠 Creating models...")
models = {
    "LePhyJEPA": LePhyJEPA(latent_dim=32).to(device),
    "SimCLR": SimCLR(latent_dim=32).to(device),
    "VICReg": VICReg(latent_dim=32).to(device)
}

# 5. Test forward pass
print("\n🧪 Testing forward pass...")
for name, model in models.items():
    with torch.no_grad():
        view1 = sample["view1"].unsqueeze(0).to(device)
        view2 = sample["view2"].unsqueeze(0).to(device)
        rgb = sample["rgb"].unsqueeze(0).to(device)
        
        outputs = model(view1, view2, rgb)
        print(f"{name:10s} Loss: {outputs['total'].item():.4f}, "
              f"Features: {outputs.get('z1', torch.zeros(1)).shape}")

# 6. Quick theorem verification
print("\n🔬 Quick theorem verification (LePhyJEPA only)...")
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
verifier = TheoremVerifier(models["LePhyJEPA"], dataloader, device)

print("\nVerifying Theorem 1 (Non-collapse):")
result1 = verifier.verify_theorem_1()

print("\nVerifying Theorem 5 (Physics compliance):")
result5 = verifier.verify_theorem_5()

# 7. Parameter count
print("\n📈 Model sizes:")
for name, model in models.items():
    params = sum(p.numel() for p in model.parameters())
    print(f"{name:10s} Parameters: {params:,}")

print("\n✅ Demo complete!")
print("\nNext steps:")
print("1. Run ablation study: python training/ablation_study.py")
print("2. Train full model: python training/train_core.py")
print("3. Compare baselines: python training/train_extended.py --config configs/baseline.yaml")
