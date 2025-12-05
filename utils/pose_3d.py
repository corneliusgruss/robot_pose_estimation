"""
3D Pose Estimation Utilities

This module provides functions for estimating 3D errors from 2D keypoint predictions.

Two approaches are provided:
1. Scaling approximation (compute_add_error_scaled) - fast but approximate
2. PnP-based estimation (compute_add_error_pnp) - more accurate, requires camera intrinsics

IMPORTANT: The scaling-based approach is an approximation for evaluation purposes only.
The PnP-based approach provides more accurate 3D error estimation but treats the
robot as a rigid body (which is an approximation for articulated robots).
"""

import numpy as np
import cv2


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


def compute_add_error_pnp(pred_2d, gt_3d, camera_matrix, dist_coeffs=None):
    """
    Compute ADD error using PnP pose estimation.

    This function estimates the 6D pose of the robot by solving the Perspective-n-Point
    problem, then computes the ADD (Average Distance of model points) error.

    The approach:
    1. Use GT 3D positions as the "object model" (robot reference frame)
    2. Solve PnP with predicted 2D keypoints to get pose (R, t)
    3. Transform model points by estimated pose
    4. Compare transformed points to GT 3D positions

    Note: This treats the robot as a rigid body, which is an approximation
    since robot arms are articulated. However, it provides a meaningful
    measure of 3D pose accuracy.

    Args:
        pred_2d: (N, 2) predicted 2D keypoints in pixels
        gt_3d: (N, 3) ground truth 3D positions in meters (robot frame)
        camera_matrix: (3, 3) camera intrinsic matrix
        dist_coeffs: distortion coefficients (default: None = no distortion)

    Returns:
        add_error: mean ADD error in meters (NaN if PnP fails)
        per_joint_errors: (N,) per-joint 3D errors in meters
        pose: dict with 'rvec', 'tvec', 'R' if successful, None otherwise
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float32)

    # Ensure correct types for OpenCV
    object_points = gt_3d.astype(np.float32).reshape(-1, 1, 3)
    image_points = pred_2d.astype(np.float32).reshape(-1, 1, 2)
    camera_matrix = camera_matrix.astype(np.float32)
    dist_coeffs = dist_coeffs.astype(np.float32)

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return np.nan, np.full(len(gt_3d), np.nan), None

    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)

    # Transform model points by estimated pose: P_cam = R @ P_model + t
    pred_3d = (R @ gt_3d.T).T + tvec.T

    # ADD error: distance between estimated 3D positions and GT 3D positions
    # Note: Both are now in camera frame
    per_joint_errors = np.linalg.norm(pred_3d - gt_3d, axis=1)
    add_error = per_joint_errors.mean()

    pose = {
        'rvec': rvec,
        'tvec': tvec,
        'R': R
    }

    return add_error, per_joint_errors, pose


def compute_reprojection_error(pred_2d, gt_3d, camera_matrix, dist_coeffs=None):
    """
    Compute reprojection error after PnP pose estimation.

    This provides a measure of how well the estimated pose explains the
    observed 2D keypoints.

    Args:
        pred_2d: (N, 2) predicted 2D keypoints in pixels
        gt_3d: (N, 3) ground truth 3D positions in meters
        camera_matrix: (3, 3) camera intrinsic matrix
        dist_coeffs: distortion coefficients (default: None)

    Returns:
        reproj_error: mean reprojection error in pixels
        per_joint_errors: (N,) per-joint reprojection errors
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float32)

    object_points = gt_3d.astype(np.float32).reshape(-1, 1, 3)
    image_points = pred_2d.astype(np.float32).reshape(-1, 1, 2)
    camera_matrix = camera_matrix.astype(np.float32)
    dist_coeffs = dist_coeffs.astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return np.nan, np.full(len(gt_3d), np.nan)

    # Project 3D points back to 2D
    projected_2d, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_2d = projected_2d.reshape(-1, 2)

    # Reprojection error
    per_joint_errors = np.linalg.norm(projected_2d - pred_2d, axis=1)
    reproj_error = per_joint_errors.mean()

    return reproj_error, per_joint_errors
