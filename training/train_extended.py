"""
Training script for extended models (Code 4) including baselines
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import sys
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_generator import SyntheticNYUv2
from models.lephyjepa_extended import FinalLePhyJEPA, ScaledLePhyJEPA
from models.baselines import SimCLR, VICReg, BaselineTrainer
from models.theorem_verifier import TheoremVerifier


class ExtendedTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        # Create datasets
        print("📊 Creating datasets...")
        self.train_dataset = SyntheticNYUv2(
            num_samples=config['data']['max_samples'],
            img_size=config['data']['img_size'],
            split='train'
        )
        self.val_dataset = SyntheticNYUv2(
            num_samples=config['data']['max_samples'],
            img_size=config['data']['img_size'],
            split='val'
        )
        self.test_dataset = SyntheticNYUv2(
            num_samples=config['data']['max_samples'],
            img_size=config['data']['img_size'],
            split='test'
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['lr']
        )
        
        # Tracking
        self.loss_history = {
            'train': [],
            'val': [],
            'test': None
        }
        
        print(f"✅ Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        print(f"✅ Batch size: {config['training']['batch_size']}")
        print(f"✅ Device: {self.device}")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            view1 = batch["view1"].to(self.device)
            view2 = batch["view2"].to(self.device)
            rgb = batch.get("rgb", None)
            if rgb is not None:
                rgb = rgb.to(self.device)
            
            # Forward pass
            outputs = self.model(view1, view2, rgb)
            loss = outputs["total"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # EMA update if available
            if hasattr(self.model, 'update_target_encoder'):
                self.model.update_target_encoder()
            elif hasattr(self.model, 'update_target'):
                self.model.update_target()
            
            # Track
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item(),
                "jepa": outputs.get("jepa", 0).item(),
                "phy": outputs.get("physics", 0).item(),
            })
        
        avg_loss = total_loss / max(num_batches, 1)
        self.loss_history['train'].append(avg_loss)
        return avg_loss
    
    def evaluate(self, loader, name="val"):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                view1 = batch["view1"].to(self.device)
                view2 = batch["view2"].to(self.device)
                rgb = batch.get("rgb", None)
                if rgb is not None:
                    rgb = rgb.to(self.device)
                
                outputs = self.model(view1, view2, rgb)
                loss = outputs["total"]
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        if name in self.loss_history:
            self.loss_history[name].append(avg_loss)
        else:
            self.loss_history[name] = [avg_loss]
        
        return avg_loss
    
    def train(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        print(f"\n🔥 Training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            
            # Validation every 2 epochs
            if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                val_loss = self.evaluate(self.val_loader, 'val')
                print(f"📊 Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            else:
                print(f"📈 Epoch {epoch}: Train Loss={train_loss:.4f}")
        
        # Final test evaluation
        test_loss = self.evaluate(self.test_loader, 'test')
        print(f"\n🎯 Final Test Loss: {test_loss:.4f}")
        
        return self.loss_history
    
    def verify_theorems(self):
        """Run theorem verification"""
        verifier = TheoremVerifier(self.model, self.test_loader, self.device)
        return verifier.verify_all_theorems()


def create_model_from_config(config):
    """Create model based on config"""
    model_type = config['model']['type']
    
    if model_type == 'lephyjepa_core':
        return FinalLePhyJEPA(
            latent_dim=config['model']['latent_dim'],
            physics_type=config['model'].get('physics_type', 'depth_smoothness')
        )
    elif model_type == 'lephyjepa_scaled':
        return ScaledLePhyJEPA(
            latent_dim=config['model']['latent_dim'],
            backbone=config['model'].get('backbone', 'resnet18')
        )
    elif model_type in ['simclr', 'vicreg']:
        return BaselineTrainer.create_baseline(
            model_type,
            latent_dim=config['model']['latent_dim']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main(config_path="configs/baseline.yaml"):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    print(f"🧠 Creating {config['model']['type']} model...")
    model = create_model_from_config(config)
    
    # Create trainer
    trainer = ExtendedTrainer(model, config)
    
    # Train
    losses = trainer.train()
    
    # Verify theorems (if applicable)
    if config['model']['type'] in ['lephyjepa_core', 'lephyjepa_scaled']:
        theorems = trainer.verify_theorems()
    else:
        theorems = None
    
    # Save results
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'config': config,
        'losses': losses,
        'theorems': theorems,
        'model_type': config['model']['type']
    }
    
    # Save model
    model_path = f"{results_dir}/{config['model']['type']}_model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Save results as JSON
    results_path = f"{results_dir}/{config['model']['type']}_results.json"
    
    # Convert tensors to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(x) for x in obj]
        elif torch.is_tensor(obj):
            return obj.cpu().tolist() if obj.numel() > 1 else obj.cpu().item()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    with open(results_path, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    print(f"\n💾 Saved model to: {model_path}")
    print(f"💾 Saved results to: {results_path}")
    
    return model, trainer, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    args = parser.parse_args()
    
    main(args.config)
