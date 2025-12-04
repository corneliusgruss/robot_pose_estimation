"""
Stage 2: Keypoint Detection Model

Predicts 2D keypoint locations from cropped robot image.
Input: 512x512 RGB image (cropped to robot bbox)
Output: 12 values (x,y for 6 joints) normalized to [0,1]
"""

import torch
import torch.nn as nn
import torchvision.models as models


class KeypointModel(nn.Module):
    """
    CNN for 2D keypoint regression.
    Uses ResNet34 by default for better spatial feature extraction.
    """
    
    def __init__(self, num_keypoints=6, backbone='resnet34', pretrained=True):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.num_outputs = num_keypoints * 2  # x,y per keypoint
        self.backbone_name = backbone
        
        if backbone == 'resnet34':
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            feature_dim = 512
        elif backbone == 'resnet50':
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            feature_dim = 2048
        elif backbone == 'resnet18':
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove classification head
        self.backbone.fc = nn.Identity()
        
        # Regression head for keypoints
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_outputs),
            nn.Sigmoid()  # Output in [0,1] range
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        
        Returns:
            keypoints: (B, num_keypoints*2) normalized keypoint coordinates
                       Formatted as [x0, y0, x1, y1, ..., x5, y5]
        """
        features = self.backbone(x)
        keypoints = self.head(features)
        return keypoints
    
    def predict_structured(self, x):
        """
        Predict and reshape to (B, num_keypoints, 2).
        
        Returns:
            keypoints: (B, 6, 2) array of (x, y) coordinates
        """
        flat = self.forward(x)
        return flat.view(-1, self.num_keypoints, 2)
    
    def __repr__(self):
        return f"KeypointModel(num_keypoints={self.num_keypoints}, backbone={self.backbone_name})"


def load_keypoint_model(checkpoint_path, device='cuda'):
    """
    Load a trained KeypointModel from checkpoint.
    
    Args:
        checkpoint_path: path to .pt checkpoint file
        device: device to load model on
    
    Returns:
        model: loaded model in eval mode
        checkpoint: full checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine architecture from checkpoint if available
    num_keypoints = checkpoint.get('num_keypoints', 6)
    backbone = checkpoint.get('backbone', 'resnet34')
    
    model = KeypointModel(num_keypoints=num_keypoints, backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


if __name__ == '__main__':
    # Quick test
    model = KeypointModel(num_keypoints=6)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    y_structured = model.predict_structured(x)
    print(f"Input: {x.shape}")
    print(f"Output (flat): {y.shape}")
    print(f"Output (structured): {y_structured.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
