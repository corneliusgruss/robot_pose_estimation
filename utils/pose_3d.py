"""
3D Pose Estimation Utilities

This module provides functions for estimating 3D errors from 2D keypoint predictions.

IMPORTANT: The scaling-based approach used here is an approximation for evaluation
purposes only. It is NOT true 3D pose estimation.

Limitations of the scaling approach:
- Requires ground truth 3D positions to compute the scale factor
- Assumes roughly orthographic projection (constant depth)
- Does not account for perspective distortion at different depths
- Cannot be used for inference without ground truth

For true 3D pose estimation, you would need:
- Accurate camera intrinsics (focal length, principal point)
- Known 3D model of the robot (joint positions in robot frame)
- PnP solver (e.g., cv2.solvePnP) to estimate camera pose
"""

import numpy as np


def compute_add_error_scaled(pred_2d, gt_2d, gt_3d):
    """
    Estimate 3D error using scale from ground truth correspondences.

    This function computes an approximate 3D error by:
    1. Computing pairwise distances between joints in 2D and 3D (ground truth)
    2. Estimating a median scale factor (meters per pixel)
    3. Multiplying 2D pixel errors by this scale

    This is an APPROXIMATION that assumes roughly constant depth across all joints.
    It should only be used for evaluation when ground truth is available.

    Args:
        pred_2d: (6, 2) predicted 2D keypoints in pixels
        gt_2d: (6, 2) ground truth 2D keypoints in pixels
        gt_3d: (6, 3) ground truth 3D positions in meters

    Returns:
        add_error: mean 3D spatial error in meters (approximate)
        per_joint_errors: (6,) per-joint 3D errors in meters
        scale: estimated meters per pixel
    """
    # Compute pairwise distances in 2D and 3D for ground truth
    scales = []
    for i in range(6):
        for j in range(i + 1, 6):
            dist_2d = np.linalg.norm(gt_2d[i] - gt_2d[j])
            dist_3d = np.linalg.norm(gt_3d[i] - gt_3d[j])

            if dist_2d > 10:  # Avoid degenerate cases (joints too close in 2D)
                scales.append(dist_3d / dist_2d)

    # Use median scale to be robust to outliers
    scale = np.median(scales) if scales else 0.003  # Fallback ~3mm/px

    # Compute 2D error and convert to approximate 3D using scale
    errors_2d = np.linalg.norm(pred_2d - gt_2d, axis=1)
    errors_3d = errors_2d * scale

    return errors_3d.mean(), errors_3d, scale


def compute_auc(errors, max_threshold=0.30, num_steps=100):
    """
    Compute Area Under the Curve for ADD metric.

    The ADD (Average Distance of model points) metric measures pose accuracy
    by computing the percentage of predictions within various distance thresholds.

    Args:
        errors: array of ADD errors in meters
        max_threshold: maximum threshold in meters (default 0.30m = 30cm)
        num_steps: number of threshold steps for AUC computation

    Returns:
        auc: Area under curve (0-1, higher is better)
        thresholds: array of thresholds used (in meters)
        accuracies: accuracy (fraction of samples) at each threshold
    """
    thresholds = np.linspace(0, max_threshold, num_steps)
    accuracies = []

    for thresh in thresholds:
        acc = (errors <= thresh).mean()
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    auc = np.trapz(accuracies, thresholds) / max_threshold

    return auc, thresholds, accuracies
