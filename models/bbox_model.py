import torch
import torch.nn as nn
import torchvision.models as models


class BBoxModel(nn.Module):

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

        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        bbox = self.head(features)
        return bbox

    def __repr__(self):
        return f"BBoxModel(backbone={self.backbone_name})"


def load_bbox_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = checkpoint.get('backbone', 'resnet18')

    model = BBoxModel(backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


if __name__ == '__main__':
    model = BBoxModel()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
