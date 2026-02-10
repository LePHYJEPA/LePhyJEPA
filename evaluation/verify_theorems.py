"""
Standalone theorem verification script
"""

import torch
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_generator import SyntheticNYUv2
from models.lephyjepa_core import LePhyJEPA
from models.theorem_verifier import TheoremVerifier
from torch.utils.data import DataLoader


def verify_model(model_path, config_path="configs/core.yaml"):
    """Load and verify a trained model"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = LePhyJEPA(
        latent_dim=config['model']['latent_dim'],
        lambda_phy=config['model']['lambda_phy'],
        sigma=config['model']['sigma']
    )
    
    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"📂 Loaded model from {model_path}")
    else:
        print(f"⚠️ Model not found at {model_path}")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create dataloader
    dataset = SyntheticNYUv2(
        num_samples=50,
        img_size=config['data']['img_size']
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Verify theorems
    verifier = TheoremVerifier(model, dataloader, device)
    results = verifier.verify_all_theorems()
    
    return results


def verify_all_models():
    """Verify all models in results directory"""
    results_dir = "experiments/results"
    if not os.path.exists(results_dir):
        print(f"⚠️ Results directory not found: {results_dir}")
        return
    
    all_results = {}
    
    for file in os.listdir(results_dir):
        if file.endswith("_model.pt"):
            model_path = os.path.join(results_dir, file)
            print(f"\n" + "="*60)
            print(f"Verifying: {file}")
            print("="*60)
            
            try:
                results = verify_model(model_path)
                all_results[file] = results
                
                # Save individual results
                result_file = file.replace("_model.pt", "_verification.json")
                import json
                with open(os.path.join(results_dir, result_file), 'w') as f:
                    json.dump(results, f, indent=2)
                
            except Exception as e:
                print(f"❌ Error verifying {file}: {e}")
    
    # Save summary
    import json
    summary_path = os.path.join(results_dir, "verification_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Saved verification summary to {summary_path}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--all', action='store_true', help='Verify all models in results directory')
    
    args = parser.parse_args()
    
    if args.all:
        verify_all_models()
    elif args.model:
        verify_model(args.model)
    else:
        # Default: verify the core model
        verify_model("experiments/results/lephyjepa_core_model.pt")
