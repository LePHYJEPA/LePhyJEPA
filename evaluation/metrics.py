"""
Evaluation metrics for LePhyJEPA
"""

import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class RepresentationMetrics:
    """Metrics for evaluating learned representations"""
    
    @staticmethod
    def feature_variance(features):
        """Calculate variance of features (anti-collapse)"""
        return torch.var(features, dim=0).mean().item()
    
    @staticmethod
    def feature_covariance_rank(features):
        """Calculate rank of covariance matrix"""
        cov = torch.cov(features.T)
        rank = torch.linalg.matrix_rank(cov).item()
        return rank
    
    @staticmethod
    def energy_constraint(features, target_sigma=1.0):
        """Check energy constraint satisfaction"""
        energy = torch.mean(torch.norm(features, dim=1) ** 2).item()
        return energy, abs(energy - target_sigma ** 2)
    
    @staticmethod
    def alignment_and_uniformity(features1, features2):
        """Alignment and uniformity metrics from Wang & Isola (2020)"""
        # Alignment: features from same sample should be similar
        alignment = torch.mean(torch.norm(features1 - features2, dim=1) ** 2).item()
        
        # Uniformity: features should be uniformly distributed on unit sphere
        # Normalize features
        features1_norm = F.normalize(features1, dim=1)
        uniformity = torch.pdist(features1_norm, p=2).pow(2).mul(-2).exp().mean().log().item()
        
        return alignment, uniformity
    
    @staticmethod
    def clustering_metrics(features, labels=None):
        """Clustering metrics (if labels available)"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        features_np = features.cpu().numpy()
        
        # Silhouette score
        if labels is not None:
            labels_np = labels.cpu().numpy()
            if len(np.unique(labels_np)) > 1:
                silhouette = silhouette_score(features_np, labels_np)
            else:
                silhouette = 0.0
        else:
            # Use k-means clustering
            kmeans = KMeans(n_clusters=min(10, len(features)), random_state=42)
            cluster_labels = kmeans.fit_predict(features_np)
            silhouette = silhouette_score(features_np, cluster_labels)
        
        return silhouette


class Visualization:
    """Visualization utilities for representations"""
    
    @staticmethod
    def plot_tsne(features, labels=None, title="t-SNE Visualization", save_path=None):
        """Plot t-SNE of features"""
        features_np = features.cpu().numpy()
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features_np)
        
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            labels_np = labels.cpu().numpy()
            unique_labels = np.unique(labels_np)
            for label in unique_labels:
                mask = labels_np == label
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           label=f'Class {label}', alpha=0.6)
            plt.legend()
        else:
            plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6)
        
        plt.title(title)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_feature_norms(features, target_sigma=1.0, save_path=None):
        """Plot distribution of feature norms"""
        norms = torch.norm(features, dim=1).cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.hist(norms, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=target_sigma, color='red', linestyle='--', 
                   label=f'Target σ={target_sigma:.2f}', linewidth=2)
        plt.axvline(x=np.mean(norms), color='green', linestyle='--', 
                   label=f'Mean={np.mean(norms):.2f}', linewidth=2)
        
        plt.title("Feature Norm Distribution")
        plt.xlabel("Norm")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return norms.mean(), norms.std()


class PhysicsComplianceMetrics:
    """Metrics for physics compliance"""
    
    @staticmethod
    def depth_smoothness_score(depth_pred, rgb=None):
        """Calculate depth smoothness score"""
        # Horizontal smoothness
        depth_dx = depth_pred[:, :, :, :-1] - depth_pred[:, :, :, 1:]
        # Vertical smoothness
        depth_dy = depth_pred[:, :, :-1, :] - depth_pred[:, :, 1:, :]
        
        smoothness = torch.abs(depth_dx).mean() + torch.abs(depth_dy).mean()
        return smoothness.item()
    
    @staticmethod
    def boundary_alignment_score(depth_pred, rgb=None):
        """Calculate alignment between depth boundaries and RGB edges"""
        if rgb is None:
            return 0.0
        
        # Simple edge detection on RGB
        rgb_gray = rgb.mean(dim=1, keepdim=True)
        rgb_dx = rgb_gray[:, :, :, :-1] - rgb_gray[:, :, :, 1:]
        rgb_dy = rgb_gray[:, :, :-1, :] - rgb_gray[:, :, 1:, :]
        rgb_edges = torch.abs(rgb_dx) + torch.abs(rgb_dy)
        
        # Depth edges
        depth_dx = depth_pred[:, :, :, :-1] - depth_pred[:, :, :, 1:]
        depth_dy = depth_pred[:, :, :-1, :] - depth_pred[:, :, 1:, :]
        depth_edges = torch.abs(depth_dx) + torch.abs(depth_dy)
        
        # Correlation between edges
        if rgb_edges.numel() > 0 and depth_edges.numel() > 0:
            correlation = torch.corrcoef(
                torch.stack([rgb_edges.flatten(), depth_edges.flatten()])
            )[0, 1].item()
        else:
            correlation = 0.0
        
        return correlation


def compute_all_metrics(model, dataloader, device):
    """Compute all metrics for a model"""
    model.eval()
    all_features = []
    all_losses = []
    
    with torch.no_grad():
        for batch in dataloader:
            view1 = batch["view1"].to(device)
            view2 = batch["view2"].to(device)
            rgb = batch.get("rgb", None)
            if rgb is not None:
                rgb = rgb.to(device)
            
            outputs = model(view1, view2, rgb)
            all_losses.append(outputs["total"].item())
            
            if "z1" in outputs:
                all_features.append(outputs["z1"].cpu())
    
    if not all_features:
        return {}
    
    features = torch.cat(all_features, dim=0)
    
    metrics = {
        "avg_loss": np.mean(all_losses),
        "variance": RepresentationMetrics.feature_variance(features),
        "covariance_rank": RepresentationMetrics.feature_covariance_rank(features),
        "energy": RepresentationMetrics.energy_constraint(features)[0],
        "num_samples": len(features),
        "feature_dim": features.shape[1]
    }
    
    return metrics
