"""
Run ablation studies for LePhyJEPA
"""

import torch
import yaml
import json
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_generator import SyntheticNYUv2
from models.lephyjepa_core import LePhyJEPA
from models.theorem_verifier import TheoremVerifier
from torch.utils.data import DataLoader


class AblationStudy:
    def __init__(self, config_path="configs/ablation.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Create dataset
        self.dataset = SyntheticNYUv2(
            num_samples=self.config['data']['max_samples'],
            img_size=self.config['data']['img_size']
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
        self.results = []
    
    def run_config(self, config_name, model_config):
        """Run training for one ablation configuration"""
        print(f"\n🔬 Testing: {config_name}")
        
        # Create model with config
        model = LePhyJEPA(
            latent_dim=model_config.get('latent_dim', 32),
            lambda_phy=model_config.get('lambda_phy', 0.1),
            sigma=model_config.get('sigma', 1.0)
        )
        model.to(self.device)
        
        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        losses = []
        
        model.train()
        for epoch in range(3):  # Short training
            epoch_loss = 0
            for i, batch in enumerate(self.dataloader):
                if i >= 10:  # Limit batches
                    break
                
                view1 = batch["view1"].to(self.device)
                view2 = batch["view2"].to(self.device)
                rgb = batch.get("rgb", None)
                if rgb is not None:
                    rgb = rgb.to(self.device)
                
                outputs = model(view1, view2, rgb)
                loss = outputs["total"]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / min(10, len(self.dataloader)))
        
        # Evaluate
        verifier = TheoremVerifier(model, self.dataloader, self.device)
        theorem1 = verifier.verify_theorem_1()
        theorem5 = verifier.verify_theorem_5() if model_config.get('lambda_phy', 0) > 0 else None
        
        # Calculate collapse risk (inverse of variance)
        collapse_risk = max(0, 1 - theorem1.get('variance', 0) * 1000)
        
        result = {
            "config": config_name,
            "final_loss": losses[-1] if losses else 0.0,
            "theorem1": theorem1.get('verified', False),
            "theorem5": theorem5.get('verified', False) if theorem5 else None,
            "collapse_risk": min(1.0, collapse_risk),
            "variance": theorem1.get('variance', 0),
            "energy": theorem1.get('energy', 0)
        }
        
        self.results.append(result)
        print(f"  Final loss: {result['final_loss']:.4f}")
        print(f"  Theorem 1: {'✅' if result['theorem1'] else '❌'}")
        print(f"  Collapse risk: {result['collapse_risk']:.2f}")
        
        return result
    
    def run_all(self):
        """Run all ablation configurations"""
        ablation_configs = [
            {
                "name": "Full LePhyJEPA",
                "lambda_phy": 0.1,
                "sigma": 1.0
            },
            {
                "name": "No Physics",
                "lambda_phy": 0.0,
                "sigma": 1.0
            },
            {
                "name": "No Energy Constraint",
                "lambda_phy": 0.1,
                "sigma": 0.0
            },
            {
                "name": "Baseline (JEPA only)",
                "lambda_phy": 0.0,
                "sigma": 0.0
            }
        ]
        
        print("="*60)
        print("RUNNING ABLATION STUDY")
        print("="*60)
        
        for config in ablation_configs:
            self.run_config(config["name"], config)
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save results to file"""
        os.makedirs("experiments/results", exist_ok=True)
        
        # Save as JSON
        with open("experiments/results/ablation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV
        import pandas as pd
        df = pd.DataFrame(self.results)
        df.to_csv("experiments/results/ablation_results.csv", index=False)
        
        print(f"\n💾 Saved results to experiments/results/")
    
    def plot_results(self):
        """Generate plots for ablation study"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            df = pd.DataFrame(self.results)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Plot 1: Final Loss
            axes[0, 0].bar(df['config'], df['final_loss'], color='skyblue')
            axes[0, 0].set_title("Final Training Loss")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Theorem Compliance
            theorem1_scores = [1.0 if r else 0.0 for r in df['theorem1']]
            axes[0, 1].bar(df['config'], theorem1_scores, color='lightgreen')
            axes[0, 1].set_title("Theorem 1 Compliance")
            axes[0, 1].set_ylabel("Verified (1=True)")
            axes[0, 1].set_ylim(0, 1.2)
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Collapse Risk
            axes[1, 0].bar(df['config'], df['collapse_risk'], color='salmon')
            axes[1, 0].set_title("Collapse Risk")
            axes[1, 0].set_ylabel("Risk (1=High)")
            axes[1, 0].set_ylim(0, 1.2)
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Variance vs Energy
            axes[1, 1].scatter(df['variance'], df['energy'], c=df['collapse_risk'], cmap='RdYlGn_r', s=100)
            for i, row in df.iterrows():
                axes[1, 1].annotate(row['config'][:10], (row['variance'], row['energy']), fontsize=8)
            axes[1, 1].set_title("Variance vs Energy")
            axes[1, 1].set_xlabel("Variance")
            axes[1, 1].set_ylabel("Energy")
            
            plt.tight_layout()
            plt.savefig("experiments/results/figures/ablation_plot.png", dpi=150)
            plt.show()
            
            print("📊 Generated ablation plots")
            
        except ImportError:
            print("⚠️ Matplotlib not available for plotting")


def main():
    study = AblationStudy()
    results = study.run_all()
    study.plot_results()
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    
    import pandas as pd
    df = pd.DataFrame(results)
    print(df[['config', 'final_loss', 'theorem1', 'collapse_risk']].to_string(index=False))
    
    best = df.loc[df['final_loss'].idxmin(), 'config']
    safest = df.loc[df['collapse_risk'].idxmin(), 'config']
    
    print(f"\n🔑 Key Findings:")
    print(f"  • Best performing: {best} (loss: {df['final_loss'].min():.4f})")
    print(f"  • Most robust: {safest} (collapse risk: {df['collapse_risk'].min():.2f})")


if __name__ == "__main__":
    main()
