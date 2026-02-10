# LePhyJEPA: Physics-Informed Latent JEPA

Official implementation of "LePhyJEPA: Geometric and Semantic Learning Foundations"  
Extends LeJEPA with physics-informed constraints.

## Features
- Physics-regularized latent prediction
- Theorem-driven verification (non-collapse, physics compliance)
- Ablation studies & baseline comparisons
- Scalable architectures

## Quick Start
```bash
pip install -r requirements.txt
python training/train_core.py --config configs/core.yaml
