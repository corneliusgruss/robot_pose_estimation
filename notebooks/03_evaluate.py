# %% [markdown]
# # Evaluate Full Pipeline
# 
# Load both trained models and compute final metrics.

# %% Cell 1: Setup
import sys
sys.path.append('..')

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import config
from models import load_pipeline
from data import RobotKeypointDataset
from utils import (
    compute_add_error_scaled, compute_auc,
    visualize_pipeline_result, plot_error_distribution
)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% Cell 2: Load Pipeline
pipeline = load_pipeline(
    stage1_path='../checkpoints/stage1_best.pt',
    stage2_path='../checkpoints/stage2_best.pt',
    config=config,
    device=device
)
print("✓ Pipeline loaded")

# %% Cell 3: Load Test Data with 3D Positions
test_dataset = RobotKeypointDataset(
    data_dirs=[config.TEST_DIR],
    config=config,
    load_3d=True
)
print(f"Test set: {len(test_dataset)} samples")

# %% Cell 4: Evaluate Full Pipeline
print("\nEvaluating full pipeline...")

results = {
    'pixel_errors': [],
    'add_errors': [],
    'per_joint_2d': np.zeros(6),
    'per_joint_3d': np.zeros(6),
    'scales': [],
}

for i in tqdm(range(len(test_dataset))):
    sample = test_dataset.samples[i]
    img_path = sample['img_path']
    gt_2d = sample['keypoints']
    gt_3d = sample.get('positions_3d')
    
    # Predict
    pred_2d, bbox = pipeline.predict(img_path)
    
    # 2D errors
    errors_2d = np.linalg.norm(pred_2d - gt_2d, axis=1)
    results['pixel_errors'].append(errors_2d.mean())
    results['per_joint_2d'] += errors_2d
    
    # 3D errors
    if gt_3d is not None:
        add_error, per_joint_3d, scale = compute_add_error_scaled(pred_2d, gt_2d, gt_3d)
        results['add_errors'].append(add_error)
        results['per_joint_3d'] += per_joint_3d
        results['scales'].append(scale)

# Convert to arrays
results['pixel_errors'] = np.array(results['pixel_errors'])
results['add_errors'] = np.array(results['add_errors'])
results['per_joint_2d'] /= len(test_dataset)
results['per_joint_3d'] /= len(test_dataset)
results['avg_scale'] = np.mean(results['scales'])

# %% Cell 5: Evaluate Stage 2 Only (GT BBox)
print("\nEvaluating Stage 2 with GT bboxes...")

results_gt = {
    'pixel_errors': [],
    'add_errors': [],
}

for i in tqdm(range(len(test_dataset))):
    sample = test_dataset.samples[i]
    img_path = sample['img_path']
    gt_2d = sample['keypoints']
    gt_3d = sample.get('positions_3d')
    
    # Get GT bbox
    gt_bbox = test_dataset.compute_bbox(gt_2d)
    
    # Predict with GT bbox
    img = Image.open(img_path).convert('RGB')
    pred_2d = pipeline.predict_with_gt_bbox(img, gt_bbox)
    
    # Errors
    px_error = np.linalg.norm(pred_2d - gt_2d, axis=1).mean()
    results_gt['pixel_errors'].append(px_error)
    
    if gt_3d is not None:
        add_error, _, _ = compute_add_error_scaled(pred_2d, gt_2d, gt_3d)
        results_gt['add_errors'].append(add_error)

results_gt['pixel_errors'] = np.array(results_gt['pixel_errors'])
results_gt['add_errors'] = np.array(results_gt['add_errors'])

# %% Cell 6: Compute AUC
auc_full, thresholds, acc_full = compute_auc(results['add_errors'])
auc_gt, _, acc_gt = compute_auc(results_gt['add_errors'])

# %% Cell 7: Print Results
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print("\n--- Full Pipeline (Stage 1 + Stage 2) ---")
print(f"  2D Pixel Error:  {results['pixel_errors'].mean():.1f} px "
      f"(median: {np.median(results['pixel_errors']):.1f})")
print(f"  3D ADD Error:    {results['add_errors'].mean()*100:.1f} cm "
      f"(median: {np.median(results['add_errors'])*100:.1f} cm)")
print(f"  AUC (0-30cm):    {auc_full:.3f}")

print("\n--- Stage 2 Only (GT BBox) ---")
print(f"  2D Pixel Error:  {results_gt['pixel_errors'].mean():.1f} px")
print(f"  3D ADD Error:    {results_gt['add_errors'].mean()*100:.1f} cm")
print(f"  AUC (0-30cm):    {auc_gt:.3f}")

print("\n--- Per-Joint 2D Errors (Full Pipeline) ---")
for name, err in zip(config.JOINT_NAMES, results['per_joint_2d']):
    print(f"  {name:12s}: {err:.1f} px")

print("\n--- Per-Joint 3D Errors (Full Pipeline) ---")
for name, err in zip(config.JOINT_NAMES, results['per_joint_3d']):
    print(f"  {name:12s}: {err*100:.1f} cm")

# %% Cell 8: Plot Results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# AUC curves
axes[0].plot(thresholds * 100, acc_full * 100, 'b-', linewidth=2,
             label=f'Full Pipeline (AUC={auc_full:.3f})')
axes[0].plot(thresholds * 100, acc_gt * 100, 'g--', linewidth=2,
             label=f'GT BBox (AUC={auc_gt:.3f})')
axes[0].fill_between(thresholds * 100, 0, acc_full * 100, alpha=0.3)
axes[0].set_xlabel('ADD Threshold (cm)')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('ADD Accuracy Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 30])
axes[0].set_ylim([0, 100])

# Error histogram
axes[1].hist(results['add_errors'] * 100, bins=30, alpha=0.7, edgecolor='black')
axes[1].axvline(results['add_errors'].mean() * 100, color='red', linestyle='--',
                label=f'Mean: {results["add_errors"].mean()*100:.1f}cm')
axes[1].set_xlabel('ADD Error (cm)')
axes[1].set_ylabel('Count')
axes[1].set_title('Error Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Per-joint errors
x = np.arange(6)
width = 0.35
axes[2].bar(x - width/2, results['per_joint_2d'], width, label='2D (px)')
axes[2].bar(x + width/2, results['per_joint_3d'] * 100, width, label='3D (cm)')
axes[2].set_xticks(x)
axes[2].set_xticklabels(config.JOINT_NAMES, rotation=45)
axes[2].set_ylabel('Error')
axes[2].set_title('Per-Joint Errors')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../checkpoints/final_results.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Cell 9: Sample Predictions
print("\n--- Sample Predictions ---")
indices = [0, 100, 500, 1000, 1500]

for idx in indices:
    sample = test_dataset.samples[idx]
    fig = visualize_pipeline_result(
        sample['img_path'], 
        pipeline, 
        gt_keypoints=sample['keypoints'],
        config=config
    )
    plt.show()
    plt.close(fig)

# %% Cell 10: Summary for Paper
print("\n" + "=" * 70)
print("SUMMARY FOR PAPER/POSTER")
print("=" * 70)
print(f"""
Method: Two-Stage CNN Pipeline
  - Stage 1: {config.STAGE1_BACKBONE} bbox detector
  - Stage 2: {config.STAGE2_BACKBONE} keypoint regressor
  - Input: Single RGB image (1080×1080)
  - Output: 6 joint positions (2D keypoints)

Dataset:
  - Training: {sum(len(RobotKeypointDataset([d], config)) for d in config.TRAIN_DIRS)} synthetic images
  - Test: {len(test_dataset)} images (unseen environment)
  - Robot: UR10 (6-DOF manipulator)

Results on Test Set:
  - Mean ADD Error: {results['add_errors'].mean()*100:.1f} cm
  - Median ADD Error: {np.median(results['add_errors'])*100:.1f} cm
  - AUC (0-30cm): {auc_full:.3f}
  
  - 2D Pixel Error: {results['pixel_errors'].mean():.1f} px

Comparison:
  - Target for class: 20-30cm ✓
  - Our result: {results['add_errors'].mean()*100:.1f} cm
  - DREAM (state-of-art): <10cm
""")
