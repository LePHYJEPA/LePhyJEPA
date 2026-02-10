#!/bin/bash

# Run baseline comparison experiment

echo "=================================================="
echo "RUNNING BASELINE COMPARISON EXPERIMENT"
echo "=================================================="

# Create results directory
mkdir -p experiments/results/figures
mkdir -p experiments/results/logs

# Run baseline comparison
echo ""
echo "1. Running baseline comparison..."
python training/train_extended.py --config configs/baseline.yaml

# Generate plots
echo ""
echo "2. Generating plots..."
python evaluation/plot_results.py --type baseline \
    --input experiments/results/baseline_results.csv \
    --output experiments/results/figures/baseline_comparison.png

# Create summary
echo ""
echo "3. Creating summary..."
python -c "
import pandas as pd
import json

# Load results
try:
    df = pd.read_csv('experiments/results/baseline_results.csv')
    print('Baseline Comparison Results:')
    print('='*50)
    print(df.to_string(index=False))
    
    # Find best model
    if 'Test Loss' in df.columns:
        df['Test Loss Num'] = pd.to_numeric(df['Test Loss'], errors='coerce')
        best_model = df.loc[df['Test Loss Num'].idxmin(), 'Model']
        best_loss = df['Test Loss Num'].min()
        print(f'\\n🏆 Best model: {best_model} (Loss: {best_loss:.4f})')
    
    # Save summary
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'num_models': len(df),
        'best_model': best_model if 'best_model' in locals() else None,
        'best_loss': float(best_loss) if 'best_loss' in locals() else None,
        'results': df.to_dict('records')
    }
    
    with open('experiments/results/baseline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
except Exception as e:
    print(f'Error creating summary: {e}')
"

echo ""
echo "=================================================="
echo "EXPERIMENT COMPLETE!"
echo "Results saved to: experiments/results/"
echo "=================================================="
