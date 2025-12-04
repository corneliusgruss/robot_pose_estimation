# %% [markdown]
# # Train Stage 1: Bounding Box Detection
# 
# Train the bounding box model. Run this after Stage 2 is trained.

# %% Cell 1: Setup
import sys
sys.path.append('..')

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import config
from models import BBoxModel
from data import RobotKeypointDataset
from utils import train_stage1, plot_training_history, visualize_bbox_predictions, denormalize_image

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% Cell 2: Load Data
train_dataset = RobotKeypointDataset(
    data_dirs=config.TRAIN_DIRS,
    config=config
)

test_dataset = RobotKeypointDataset(
    data_dirs=[config.TEST_DIR],
    config=config
)

print(f"Train: {len(train_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")

# %% Cell 3: Create DataLoaders
# Can use larger batch size - smaller images
train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE * 2,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE * 2,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)

# %% Cell 4: Create Model
model = BBoxModel(
    backbone=config.STAGE1_BACKBONE,
    pretrained=True
)
print(model)
print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

# %% Cell 5: Train
model, history = train_stage1(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=config.EPOCHS_STAGE1,
    lr=config.LR,
    save_dir='../checkpoints',
    save_every=config.SAVE_EVERY,
    early_stop_patience=config.EARLY_STOP_PATIENCE
)

# %% Cell 6: Plot Training History
fig = plot_training_history(history, title_prefix='Stage 1: ')
plt.savefig('../checkpoints/stage1_history.png', dpi=150)
plt.show()

print("\n✓ Training complete!")
print(f"Best model saved to: ../checkpoints/stage1_best.pt")
print(f"Best validation IoU: {max(history['val_iou']):.3f}")

# %% Cell 7: Visualize Predictions
fig = visualize_bbox_predictions(test_dataset, model, device, num_samples=6, config=config)
plt.savefig('../checkpoints/stage1_predictions.png', dpi=150)
plt.show()

# %% Cell 8: IoU Distribution
from utils import compute_bbox_iou

model.eval()
ious = []

with torch.no_grad():
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        img = sample['img_stage1'].unsqueeze(0).to(device)
        pred = model(img)
        gt = sample['bbox'].unsqueeze(0).to(device)
        iou = compute_bbox_iou(pred, gt)
        ious.append(iou)

ious = np.array(ious)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(ious, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(ious), color='red', linestyle='--', label=f'Mean: {np.mean(ious):.3f}')
ax.axvline(np.median(ious), color='orange', linestyle='--', label=f'Median: {np.median(ious):.3f}')
ax.set_xlabel('IoU')
ax.set_ylabel('Count')
ax.set_title('BBox IoU Distribution')
ax.legend()
plt.tight_layout()
plt.savefig('../checkpoints/stage1_iou_dist.png', dpi=150)
plt.show()

print(f"\nIoU Statistics:")
print(f"  Mean:   {np.mean(ious):.3f}")
print(f"  Median: {np.median(ious):.3f}")
print(f"  >0.8:   {(ious > 0.8).mean()*100:.1f}%")
print(f"  >0.9:   {(ious > 0.9).mean()*100:.1f}%")
