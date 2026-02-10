"""
Synthetic RGB-D data generator for NYUv2-like data
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os

class SyntheticNYUv2(Dataset):
    def __init__(self, num_samples=100, img_size=(120, 160), split='train'):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.split = split
        
        # Define splits
        if split == 'train':
            self.indices = range(0, int(0.7 * num_samples))
        elif split == 'val':
            self.indices = range(int(0.7 * num_samples), int(0.85 * num_samples))
        elif split == 'test':
            self.indices = range(int(0.85 * num_samples), num_samples)
        else:
            self.indices = range(num_samples)
        
        self.indices = list(self.indices)
        
        # Transforms
        self.rgb_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.augmentation = T.Compose([
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.RandomHorizontalFlip(p=0.3),
        ])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        H, W = self.img_size
        
        # Generate synthetic RGB with structure
        rgb_array = np.random.rand(H, W, 3).astype(np.float32)
        y_coord, x_coord = np.mgrid[0:H, 0:W]
        
        # Add gradients for realism
        rgb_array[:, :, 0] += (x_coord / W * 0.2)  # Red gradient
        rgb_array[:, :, 1] += (y_coord / H * 0.1)  # Green gradient
        rgb_array = np.clip(rgb_array, 0, 1)
        
        rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1)
        
        # Generate synthetic depth with structure
        depth_array = np.random.rand(H, W).astype(np.float32)
        depth_array += (x_coord / W * 0.5)  # Horizontal gradient
        depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
        
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)
        
        # Create two augmented views
        view1 = self.augmentation(rgb_tensor)
        view2 = self.augmentation(rgb_tensor)
        
        return {
            "view1": view1,
            "view2": view2,
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "index": self.indices[idx]
        }


def save_synthetic_samples(output_dir="data/synthetic", num_samples=10):
    """Save synthetic samples as images for visualization"""
    os.makedirs(output_dir, exist_ok=True)
    dataset = SyntheticNYUv2(num_samples=num_samples)
    
    for i in range(min(num_samples, 5)):
        sample = dataset[i]
        
        # Save RGB
        rgb_img = (sample["rgb"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb_img).save(f"{output_dir}/rgb_{i}.png")
        
        # Save depth
        depth_img = (sample["depth"].squeeze().numpy() * 255).astype(np.uint8)
        Image.fromarray(depth_img).save(f"{output_dir}/depth_{i}.png")
        
    print(f"Saved {min(num_samples, 5)} synthetic samples to {output_dir}/")


if __name__ == "__main__":
    save_synthetic_samples()
