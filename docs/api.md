# LePhyJEPA API Documentation

## Core Models

### `LePhyJEPA` (Code 3)
```python
from models.lephyjepa_core import LePhyJEPA

model = LePhyJEPA(
    latent_dim=32,           # Dimension of latent space
    lambda_phy=0.1,         # Weight for physics loss
    sigma=1.0,              # Energy constraint parameter
    physics_type="depth_smoothness"
)

# Forward pass
outputs = model(view1, view2, rgb=None)
# Returns: {
#   'total': total_loss,
#   'jepa': jepa_loss,
#   'physics': physics_loss,
#   'energy': energy_loss,
#   'z1': online features,
#   'z2': target features,
#   'z1_pred': predicted features
# }
