"""
Stage 1: Bounding Box Detection Model

Predicts robot bounding box from full image.
Input: 256x256 RGB image
Output: 4 values [x_min, y_min, x_max, y_max] normalized to [0,1]
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BBoxModel(nn.Module):
    """
    Lightweight CNN for bounding box regression.
    Uses ResNet18 by default - sufficient for this simpler task.
    """
    
    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()
        
        self.backbone_name = backbone
        
        if backbone == 'resnet18':
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone == 'resnet34':
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove classification head
        self.backbone.fc = nn.Identity()
        
        # Regression head for bbox
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
            nn.Sigmoid()  # Output in [0,1] range
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        
        Returns:
            bbox: (B, 4) normalized bounding boxes [x_min, y_min, x_max, y_max]
        """
        features = self.backbone(x)
        bbox = self.head(features)
        return bbox
    
    def __repr__(self):
        return f"BBoxModel(backbone={self.backbone_name})"


def load_bbox_model(checkpoint_path, device='cuda'):
    """
    Load a trained BBoxModel from checkpoint.
    
    Args:
        checkpoint_path: path to .pt checkpoint file
        device: device to load model on
    
    Returns:
        model: loaded model in eval mode
        checkpoint: full checkpoint dict (contains history, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine backbone from checkpoint if available
    backbone = checkpoint.get('backbone', 'resnet18')
    
    model = BBoxModel(backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


if __name__ == '__main__':
    # Quick test
    model = BBoxModel()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
