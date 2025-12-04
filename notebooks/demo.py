# %% [markdown]
# # Robot Pose Estimation Demo
# 
# Two-stage CNN pipeline for UR10 robot arm pose estimation from single RGB images.
# 
# **Authors**: Cornelius Gruss, Devin Caulfield, Juan Rueda  
# **Course**: CS523 Deep Learning, Boston University

# %% Cell 1: Setup
import sys
sys.path.append('..')

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import config
from models import load_pipeline
from data import RobotKeypointDataset
from utils import visualize_pipeline_result, compute_auc, compute_add_error_scaled

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% Cell 2: Load Models
pipeline = load_pipeline(
    stage1_path='../checkpoints/stage1_best.pt',
    stage2_path='../checkpoints/stage2_best.pt',
    config=config,
    device=device
)
print("Pipeline loaded!")

# %% Cell 3: Load Test Dataset
test_dataset = RobotKeypointDataset(
    data_dirs=[config.TEST_DIR],
    config=config,
    load_3d=True
)
print(f"Test set: {len(test_dataset)} images")

# %% [markdown]
# ## Single Image Inference

# %% Cell 4: Run Inference on Single Image
def demo_inference(image_path, show_gt=True):
    """Run inference and display results."""
    # Get ground truth if available
    gt_kp = None
    for sample in test_dataset.samples:
        if sample['img_path'] == image_path:
            gt_kp = sample['keypoints']
            break
    
    # Run pipeline
    keypoints_2d, bbox = pipeline.predict(image_path)
    
    # Display
    img = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    
    # Bbox
    rect = plt.Rectangle(
        (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
        fill=False, edgecolor='yellow', linewidth=2
    )
    ax.add_patch(rect)
    
    # Skeleton
    colors = plt.cm.rainbow(np.linspace(0, 1, 6))
    for i in range(5):
        ax.plot([keypoints_2d[i, 0], keypoints_2d[i+1, 0]],
                [keypoints_2d[i, 1], keypoints_2d[i+1, 1]],
                'w-', linewidth=2, alpha=0.8)
    
    # Predicted keypoints
    for i, (name, color) in enumerate(zip(config.JOINT_NAMES, colors)):
        ax.scatter(keypoints_2d[i, 0], keypoints_2d[i, 1],
                   c=[color], s=200, marker='o', edgecolors='white', linewidths=2)
        ax.annotate(name, (keypoints_2d[i, 0] + 15, keypoints_2d[i, 1]),
                    color='white', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    # Ground truth
    if show_gt and gt_kp is not None:
        ax.scatter(gt_kp[:, 0], gt_kp[:, 1], c='lime', s=100, marker='x',
                   linewidths=3, label='Ground Truth')
        ax.legend(fontsize=12)
    
    ax.set_title('UR10 Pose Estimation', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print keypoints
    print("\nDetected Joint Positions (pixels):")
    for i, name in enumerate(config.JOINT_NAMES):
        print(f"  {name:12s}: ({keypoints_2d[i, 0]:6.1f}, {keypoints_2d[i, 1]:6.1f})")


# Demo on random image
sample_idx = np.random.randint(len(test_dataset))
demo_inference(test_dataset.samples[sample_idx]['img_path'])

# %% [markdown]
# ## Batch Evaluation

# %% Cell 5: Evaluate Full Pipeline
from tqdm import tqdm

add_errors = []
pixel_errors = []

for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
    sample = test_dataset.samples[i]
    img_path = sample['img_path']
    gt_2d = sample['keypoints']
    gt_3d = sample.get('positions_3d')
    
    # Predict
    pred_2d, _ = pipeline.predict(img_path)
    
    # 2D error
    px_err = np.linalg.norm(pred_2d - gt_2d, axis=1).mean()
    pixel_errors.append(px_err)
    
    # 3D error
    if gt_3d is not None:
        add_err, _, _ = compute_add_error_scaled(pred_2d, gt_2d, gt_3d)
        add_errors.append(add_err)

pixel_errors = np.array(pixel_errors)
add_errors = np.array(add_errors)

# %% Cell 6: Display Results
auc, thresholds, accuracies = compute_auc(add_errors)

print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"\n2D Pixel Error:  {pixel_errors.mean():.1f} px (median: {np.median(pixel_errors):.1f})")
print(f"3D ADD Error:    {add_errors.mean()*100:.1f} cm (median: {np.median(add_errors)*100:.1f} cm)")
print(f"AUC (0-30cm):    {auc:.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(thresholds * 100, accuracies * 100, 'b-', linewidth=2)
axes[0].fill_between(thresholds * 100, 0, accuracies * 100, alpha=0.3)
axes[0].set_xlabel('ADD Threshold (cm)', fontsize=12)
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title(f'ADD Accuracy Curve (AUC = {auc:.3f})', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 30])
axes[0].set_ylim([0, 100])

axes[1].hist(add_errors * 100, bins=30, alpha=0.7, edgecolor='black')
axes[1].axvline(add_errors.mean() * 100, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {add_errors.mean()*100:.1f}cm')
axes[1].set_xlabel('ADD Error (cm)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Error Distribution', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Method Overview
# 
# ### Two-Stage Pipeline
# 
# | Stage | Model | Input | Output |
# |-------|-------|-------|--------|
# | 1 | ResNet18 | 256×256 full image | Bounding box (4 values) |
# | 2 | ResNet34 | 512×512 crop | 6 keypoints (12 values) |
# 
# ### Training Data
# - 8,000 synthetic images (Isaac Sim)
# - 4 domain-randomized environments
# - UR10 robot arm
# 
# ### Results
# - Mean ADD Error: ~9.5 cm
# - AUC (0-30cm): ~0.70
# - Competitive with DREAM (<10cm)
