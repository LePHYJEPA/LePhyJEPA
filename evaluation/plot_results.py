"""
Plotting utilities for LePhyJEPA results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path


def plot_training_curve(losses_dict, save_path=None):
    """Plot training/validation loss curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training loss
    if 'train' in losses_dict:
        axes[0].plot(losses_dict['train'], label='Train', linewidth=2)
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Validation loss
    if 'val' in losses_dict:
        epochs = range(0, len(losses_dict['val']) * 2, 2)
        axes[1].plot(epochs, losses_dict['val'], label='Validation', 
                    linewidth=2, marker='o', markersize=4)
        
        if 'test' in losses_dict and losses_dict['test']:
            axes[1].axhline(y=losses_dict['test'][-1], color='red', 
                           linestyle='--', label=f'Test: {losses_dict["test"][-1]:.4f}')
        
        axes[1].set_title("Validation & Test Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved plot to {save_path}")
    
    plt.show()


def plot_ablation_results(results_path, save_path=None):
    """Plot ablation study results"""
    if isinstance(results_path, str):
        if results_path.endswith('.json'):
            with open(results_path, 'r') as f:
                results = json.load(f)
        elif results_path.endswith('.csv'):
            results = pd.read_csv(results_path).to_dict('records')
        else:
            raise ValueError("Unsupported file format")
    else:
        results = results_path
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Final Loss
    axes[0, 0].bar(df['config'], df['final_loss'], color='skyblue')
    axes[0, 0].set_title("Final Training Loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Theorem Compliance
    theorem1_scores = []
    for r in results:
        if isinstance(r['theorem1'], bool):
            theorem1_scores.append(1.0 if r['theorem1'] else 0.0)
        else:
            theorem1_scores.append(0.0)
    
    axes[0, 1].bar(df['config'], theorem1_scores, color='lightgreen')
    axes[0, 1].set_title("Theorem 1 Compliance")
    axes[0, 1].set_ylabel("Verified (1=True)")
    axes[0, 1].set_ylim(0, 1.2)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Collapse Risk
    axes[1, 0].bar(df['config'], df['collapse_risk'], color='salmon')
    axes[1, 0].set_title("Collapse Risk")
    axes[1, 0].set_ylabel("Risk (1=High)")
    axes[1, 0].set_ylim(0, 1.2)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary Heatmap
    summary_data = df[['final_loss', 'collapse_risk']].values.T
    im = axes[1, 1].imshow(summary_data, cmap='RdYlGn_r', aspect='auto')
    axes[1, 1].set_title("Performance Summary")
    axes[1, 1].set_xticks(range(len(df['config'])))
    axes[1, 1].set_xticklabels(df['config'], rotation=45)
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_yticklabels(['Loss', 'Collapse Risk'])
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved ablation plot to {save_path}")
    
    plt.show()


def plot_baseline_comparison(results_path, save_path=None):
    """Plot baseline comparison results"""
    if isinstance(results_path, str):
        df = pd.read_csv(results_path)
    else:
        df = results_path
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Test Loss comparison
    axes[0].bar(df['Model'], pd.to_numeric(df['Test Loss'], errors='coerce'), color='lightblue')
    axes[0].set_title("Test Loss Comparison")
    axes[0].set_ylabel("Loss")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Variance comparison
    axes[1].bar(df['Model'], pd.to_numeric(df['Variance'], errors='coerce'), color='lightgreen')
    axes[1].set_title("Feature Variance (Anti-Collapse)")
    axes[1].set_ylabel("Variance")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_results_summary(results_dir="experiments/results"):
    """Create a comprehensive summary of all results"""
    summary = {}
    
    # Find all result files
    result_files = list(Path(results_dir).glob("*.json"))
    result_files += list(Path(results_dir).glob("*.csv"))
    
    for file in result_files:
        if file.suffix == '.json':
            with open(file, 'r') as f:
                data = json.load(f)
        elif file.suffix == '.csv':
            data = pd.read_csv(file).to_dict('records')
        
        summary[file.name] = {
            'type': file.suffix[1:],
            'size': os.path.getsize(file),
            'data_keys': list(data.keys()) if isinstance(data, dict) else 'list'
        }
    
    # Save summary
    summary_path = Path(results_dir) / "results_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Created results summary: {summary_path}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['training', 'ablation', 'baseline', 'summary'], 
                       default='training', help='Type of plot to generate')
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.type == 'training':
        if args.input and os.path.exists(args.input):
            with open(args.input, 'r') as f:
                losses = json.load(f)
            plot_training_curve(losses, args.output)
    
    elif args.type == 'ablation':
        plot_ablation_results(args.input or "experiments/results/ablation_results.json", args.output)
    
    elif args.type == 'baseline':
        plot_baseline_comparison(args.input or "experiments/results/baseline_results.csv", args.output)
    
    elif args.type == 'summary':
        create_results_summary()
