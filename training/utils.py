"""
Training utilities for LePhyJEPA
"""

import torch
import numpy as np
import random
import os
from datetime import datetime


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(use_cuda=True):
    """Setup device (CPU or GPU)"""
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✅ Using CPU")
    return device


def save_checkpoint(model, optimizer, epoch, loss, path="checkpoints"):
    """Save training checkpoint"""
    os.makedirs(path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"{path}/checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, filename)
    print(f"💾 Saved checkpoint: {filename}")
    
    return filename


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"📂 Loaded checkpoint from epoch {epoch}, loss: {loss:.4f}")
    return model, optimizer, epoch, loss


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    return 0, 0


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class MetricTracker:
    """Track metrics during training"""
    def __init__(self):
        self.metrics = {}
    
    def update(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_average(self, name):
        if name in self.metrics and self.metrics[name]:
            return np.mean(self.metrics[name])
        return 0.0
    
    def reset(self):
        self.metrics = {}
    
    def to_dict(self):
        return {k: np.mean(v) for k, v in self.metrics.items() if v}
