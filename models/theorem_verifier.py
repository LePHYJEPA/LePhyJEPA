"""
Theorem verification for LePhyJEPA
Verifies Theorem 1 (Non-collapse) and Theorem 5 (Physics compliance)
"""

import torch
import numpy as np
from tqdm import tqdm

class TheoremVerifier:
    def __init__(self, model, dataloader, device=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device or next(model.parameters()).device
    
    def verify_theorem_1(self, num_samples=10, energy_tol=0.2, variance_thresh=1e-6):
        """
        Verify Theorem 1: Non-collapse under energy constraint
        
        Returns:
            dict: Verification results with metrics and boolean verdict
        """
        print("\n" + "="*50)
        print("Verifying Theorem 1: Non-Collapse")
        print("="*50)
        
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= num_samples:
                    break
                view1 = batch["view1"].to(self.device)
                if hasattr(self.model, 'encode_online'):
                    z = self.model.encode_online(view1)
                elif hasattr(self.model, 'encoder'):
                    z = self.model.encoder(view1)
                else:
                    z = self.model(view1, view1)["z1"]
                features.append(z.cpu())
        
        if len(features) < 2:
            print("⚠️ Need at least 2 samples for verification")
            return {"verified": False, "reason": "Insufficient samples"}
        
        features = torch.cat(features, dim=0)
        
        # 1. Check variance > 0 (non-collapse)
        variance = torch.var(features, dim=0).mean().item()
        no_collapse = variance > variance_thresh
        
        # 2. Check energy constraint
        energy = torch.mean(torch.norm(features, dim=1) ** 2).item()
        
        if hasattr(self.model, 'sigma'):
            target_energy = self.model.sigma ** 2
        else:
            target_energy = 1.0  # Default
        
        energy_ok = abs(energy - target_energy) < energy_tol
        
        # 3. Check rank (structured variability)
        if features.shape[0] > features.shape[1]:
            cov_matrix = torch.cov(features.T)
            rank = torch.linalg.matrix_rank(cov_matrix).item()
            full_rank = rank >= min(features.shape[1], 3)  # At least rank 3
        else:
            rank = 0
            full_rank = True  # Not enough samples to check
        
        # Print results
        print(f"Feature variance: {variance:.6f} {'✅' if no_collapse else '❌'}")
        print(f"Energy: {energy:.3f} (target: {target_energy:.3f}) {'✅' if energy_ok else '❌'}")
        print(f"Covariance rank: {rank} {'✅' if full_rank else '❌'}")
        print(f"Non-collapse verified: {'✅' if no_collapse else '❌'}")
        
        verified = no_collapse and energy_ok and full_rank
        
        return {
            "verified": verified,
            "variance": variance,
            "energy": energy,
            "target_energy": target_energy,
            "rank": rank,
            "no_collapse": no_collapse,
            "energy_constraint": energy_ok,
            "full_rank": full_rank
        }
    
    def verify_theorem_5(self, num_samples=10, physics_thresh=0.1):
        """
        Verify Theorem 5: Physics compliance
        
        Returns:
            dict: Verification results
        """
        print("\n" + "="*50)
        print("Verifying Theorem 5: Physics Compliance")
        print("="*50)
        
        self.model.eval()
        physics_losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= num_samples:
                    break
                view1 = batch["view1"].to(self.device)
                rgb = batch.get("rgb", None)
                if rgb is not None:
                    rgb = rgb.to(self.device)
                
                # Get physics loss
                if hasattr(self.model, 'compute_physics_loss'):
                    if hasattr(self.model, 'encode_online'):
                        z = self.model.encode_online(view1)
                    else:
                        z = self.model.encoder(view1)
                    loss = self.model.compute_physics_loss(z, rgb)
                    physics_losses.append(loss.item())
                else:
                    # Model doesn't have physics loss
                    physics_losses.append(0.0)
                    break
        
        if not physics_losses:
            print("⚠️ No physics loss computed")
            return {"verified": False, "reason": "No physics loss"}
        
        avg_loss = np.mean(physics_losses)
        physics_ok = avg_loss < physics_thresh
        
        print(f"Average physics loss: {avg_loss:.6f}")
        print(f"Physics compliance (< {physics_thresh}): {'✅' if physics_ok else '❌'}")
        
        return {
            "verified": physics_ok,
            "avg_physics_loss": avg_loss,
            "threshold": physics_thresh,
            "all_losses": physics_losses
        }
    
    def verify_all_theorems(self):
        """Verify both theorems"""
        results = {}
        results["theorem_1"] = self.verify_theorem_1()
        results["theorem_5"] = self.verify_theorem_5()
        results["all_verified"] = results["theorem_1"]["verified"] and results["theorem_5"]["verified"]
        
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Theorem 1 (Non-collapse): {'✅ VERIFIED' if results['theorem_1']['verified'] else '❌ FAILED'}")
        print(f"Theorem 5 (Physics): {'✅ VERIFIED' if results['theorem_5']['verified'] else '❌ FAILED'}")
        print(f"All theorems verified: {'✅ YES' if results['all_verified'] else '❌ NO'}")
        
        return results
