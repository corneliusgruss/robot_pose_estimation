# %% [markdown]
# # Train Stage 2: Keypoint Detection
# 
# Train the keypoint regression model using ground-truth bounding boxes.
# This should be trained first, before Stage 1.

# %% Cell 1: Setup
import sys
sys.path.append('..')

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import config
from models import KeypointModel
from data import RobotKeypointDataset
from utils import train_stage2, plot_training_history, visualize_sample

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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

# %% Cell 3: Visualize Sample
sample = train_dataset[0]
fig = visualize_sample(sample, config=config)
plt.show()

# %% Cell 4: Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)

# %% Cell 5: Create Model
model = KeypointModel(
    num_keypoints=config.NUM_JOINTS,
    backbone=config.STAGE2_BACKBONE,
    pretrained=True
)
print(model)
print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

# %% Cell 6: Train
model, history = train_stage2(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=config.EPOCHS_STAGE2,
    lr=config.LR,
    save_dir='../checkpoints',
    save_every=config.SAVE_EVERY,
    early_stop_patience=config.EARLY_STOP_PATIENCE,
    img_size=config.STAGE2_SIZE
)

# %% Cell 7: Plot Training History
fig = plot_training_history(history, title_prefix='Stage 2: ')
plt.savefig('../checkpoints/stage2_history.png', dpi=150)
plt.show()

print("\n✓ Training complete!")
print(f"Best model saved to: ../checkpoints/stage2_best.pt")
print(f"Best validation error: {min(history['val_error']):.1f}px")

# %% Cell 8: Visualize Predictions
from utils import denormalize_image
import numpy as np

model.eval()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

indices = np.random.choice(len(test_dataset), 6, replace=False)

for ax, idx in zip(axes, indices):
    sample = test_dataset[idx]
    
    with torch.no_grad():
        img = sample['img_stage2'].unsqueeze(0).to(device)
        pred = model(img).cpu().numpy()[0]
    
    # Display
    img_display = denormalize_image(sample['img_stage2'])
    ax.imshow(img_display)
    
    # Ground truth
    gt = sample['keypoints'].numpy().reshape(-1, 2) * config.STAGE2_SIZE
    ax.scatter(gt[:, 0], gt[:, 1], c='lime', s=100, marker='o', label='GT')
    
    # Prediction
    pred_px = pred.reshape(-1, 2) * config.STAGE2_SIZE
    ax.scatter(pred_px[:, 0], pred_px[:, 1], c='red', s=100, marker='x', label='Pred')
    
    # Error
    error = np.linalg.norm(pred_px - gt, axis=1).mean()
    ax.set_title(f'Error: {error:.1f}px')
    ax.axis('off')

axes[0].legend()
plt.tight_layout()
plt.savefig('../checkpoints/stage2_predictions.png', dpi=150)
plt.show()
