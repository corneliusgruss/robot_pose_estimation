import torch
import torch.nn as nn
import torchvision.models as models


class KeypointModel(nn.Module):

    def __init__(self, num_keypoints=6, backbone='resnet34', pretrained=True):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.num_outputs = num_keypoints * 2
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

        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_outputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        keypoints = self.head(features)
        return keypoints

    def predict_structured(self, x):
        flat = self.forward(x)
        return flat.view(-1, self.num_keypoints, 2)

    def __repr__(self):
        return f"KeypointModel(num_keypoints={self.num_keypoints}, backbone={self.backbone_name})"


def load_keypoint_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_keypoints = checkpoint.get('num_keypoints', 6)
    backbone = checkpoint.get('backbone', 'resnet34')

    model = KeypointModel(num_keypoints=num_keypoints, backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


if __name__ == '__main__':
    model = KeypointModel(num_keypoints=6)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    y_structured = model.predict_structured(x)
    print(f"Input: {x.shape}")
    print(f"Output (flat): {y.shape}")
    print(f"Output (structured): {y_structured.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
