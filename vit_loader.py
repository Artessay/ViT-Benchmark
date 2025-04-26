import torch
import torch.nn as nn
from vision_transformer import VisionTransformer

def create_vit_model(model_name: str, num_classes: int = 1000) -> VisionTransformer:
    """Create a Vision Transformer model with the specified weights."""
    model = VisionTransformer()

    weights_path = f"checkpoints/{model_name}.pth"
    model.load_state_dict(torch.load(weights_path))

    model.heads.head = nn.Linear(model.hidden_dim, num_classes)

    return model

def load_vit_model(model_path: str, num_classes: int = 1000) -> VisionTransformer:
    """Load a Vision Transformer model with the specified weights."""
    model = VisionTransformer(num_classes=num_classes)

    model.load_state_dict(torch.load(model_path))

    return model