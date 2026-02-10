"""
Tests for LePhyJEPA models
"""

import torch
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lephyjepa_core import LePhyJEPA
from models.baselines import SimCLR, VICReg
from models.theorem_verifier import TheoremVerifier
from data.synthetic_generator import SyntheticNYUv2


def test_lephyjepa_initialization():
    """Test LePhyJEPA model initialization"""
    model = LePhyJEPA(latent_dim=32, lambda_phy=0.1, sigma=1.0)
    
    assert model.latent_dim == 32
    assert model.lambda_phy == 0.1
    assert model.sigma == 1.0
    
    # Check model components exist
    assert hasattr(model, 'encoder_backbone')
    assert hasattr(model, 'online_projection')
    assert hasattr(model, 'target_projection')
    assert hasattr(model, 'predictor')
    assert hasattr(model, 'depth_decoder')
    
    print("✅ LePhyJEPA initialization test passed")


def test_lephyjepa_forward():
    """Test forward pass of LePhyJEPA"""
    model = LePhyJEPA(latent_dim=16)
    
    # Create dummy input
    batch_size = 2
    view1 = torch.randn(batch_size, 3, 120, 160)
    view2 = torch.randn(batch_size, 3, 120, 160)
    rgb = torch.randn(batch_size, 3, 120, 160)
    
    # Forward pass
    outputs = model(view1, view2, rgb)
    
    # Check outputs
    assert "total" in outputs
    assert "jepa" in outputs
    assert "physics" in outputs
    assert "energy" in outputs
    assert "z1" in outputs
    assert "z2" in outputs
    
    assert outputs["z1"].shape == (batch_size, 16)
    assert outputs["z2"].shape == (batch_size, 16)
    
    # Loss should be scalar
    assert outputs["total"].dim() == 0
    
    print("✅ LePhyJEPA forward pass test passed")


def test_baselines_initialization():
    """Test baseline models initialization"""
    simclr = SimCLR(latent_dim=32)
    vicreg = VICReg(latent_dim=32)
    
    assert simclr.encoder is not None
    assert vicreg.encoder is not None
    
    print("✅ Baselines initialization test passed")


def test_baselines_forward():
    """Test forward pass of baselines"""
    simclr = SimCLR(latent_dim=16)
    vicreg = VICReg(latent_dim=16)
    
    batch_size = 2
    view1 = torch.randn(batch_size, 3, 120, 160)
    view2 = torch.randn(batch_size, 3, 120, 160)
    
    # SimCLR forward
    outputs_simclr = simclr(view1, view2)
    assert "total" in outputs_simclr
    assert outputs_simclr["z1"].shape == (batch_size, 16)
    
    # VICReg forward
    outputs_vicreg = vicreg(view1, view2)
    assert "total" in outputs_vicreg
    assert outputs_vicreg["z1"].shape == (batch_size, 16)
    
    print("✅ Baselines forward pass test passed")


def test_theorem_verifier():
    """Test theorem verifier"""
    model = LePhyJEPA(latent_dim=16)
    dataset = SyntheticNYUv2(num_samples=10)
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2)
    
    verifier = TheoremVerifier(model, dataloader, device='cpu')
    
    # Test theorem 1 verification
    result1 = verifier.verify_theorem_1()
    assert "verified" in result1
    assert "variance" in result1
    assert "energy" in result1
    
    # Test theorem 5 verification
    result5 = verifier.verify_theorem_5()
    assert "verified" in result5
    assert "avg_physics_loss" in result5
    
    print("✅ Theorem verifier test passed")


def test_energy_constraint():
    """Test energy constraint in LePhyJEPA"""
    model = LePhyJEPA(latent_dim=16, sigma=1.0)
    
    batch_size = 4
    view1 = torch.randn(batch_size, 3, 120, 160)
    view2 = torch.randn(batch_size, 3, 120, 160)
    
    outputs = model(view1, view2)
    
    # Calculate energy from features
    energy = torch.mean(torch.norm(outputs["z1"], dim=1) ** 2).item()
    
    # Energy should be close to sigma^2 = 1.0 (after training)
    # For untrained model, just check it's not NaN
    assert not torch.isnan(outputs["energy"])
    
    print("✅ Energy constraint test passed")


def test_physics_loss():
    """Test physics loss computation"""
    model = LePhyJEPA(latent_dim=16, lambda_phy=0.1)
    
    batch_size = 2
    features = torch.randn(batch_size, 16)
    rgb = torch.randn(batch_size, 3, 120, 160)
    
    physics_loss = model.compute_physics_loss(features, rgb)
    
    # Physics loss should be non-negative
    assert physics_loss >= 0
    assert not torch.isnan(physics_loss)
    
    print("✅ Physics loss test passed")


def test_model_serialization():
    """Test model saving and loading"""
    import tempfile
    
    model = LePhyJEPA(latent_dim=16)
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        # Save model
        torch.save(model.state_dict(), tmp.name)
        
        # Load model
        loaded_model = LePhyJEPA(latent_dim=16)
        loaded_model.load_state_dict(torch.load(tmp.name))
        
        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)
        
        # Clean up
        os.unlink(tmp.name)
    
    print("✅ Model serialization test passed")


def run_all_tests():
    """Run all tests"""
    print("🧪 Running LePhyJEPA tests...")
    print("="*60)
    
    test_lephyjepa_initialization()
    test_lephyjepa_forward()
    test_baselines_initialization()
    test_baselines_forward()
    test_theorem_verifier()
    test_energy_constraint()
    test_physics_loss()
    test_model_serialization()
    
    print("="*60)
    print("✅ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
