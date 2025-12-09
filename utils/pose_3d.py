import numpy as np
import cv2


def compute_add_error_scaled(pred_2d, gt_2d, gt_3d):
    # Compute pairwise distances in 2D and 3D for ground truth
    scales = []
    for i in range(6):
        for j in range(i + 1, 6):
            dist_2d = np.linalg.norm(gt_2d[i] - gt_2d[j])
            dist_3d = np.linalg.norm(gt_3d[i] - gt_3d[j])

            if dist_2d > 10:
                scales.append(dist_3d / dist_2d)

    scale = np.median(scales) if scales else 0.003
    # print(f"scale: {scale}")

    errors_2d = np.linalg.norm(pred_2d - gt_2d, axis=1)
    errors_3d = errors_2d * scale

    return errors_3d.mean(), errors_3d, scale


def compute_auc(errors, max_threshold=0.30, num_steps=100):
    thresholds = np.linspace(0, max_threshold, num_steps)
    accuracies = []

    for thresh in thresholds:
        acc = (errors <= thresh).mean()
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    auc = np.trapz(accuracies, thresholds) / max_threshold

    return auc, thresholds, accuracies


def compute_add_error_pnp(pred_2d, gt_3d, camera_matrix, dist_coeffs=None):
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
        return np.nan, np.full(len(gt_3d), np.nan), None

    R, _ = cv2.Rodrigues(rvec)

    # Transform model points by estimated pose
    pred_3d = (R @ gt_3d.T).T + tvec.T

    per_joint_errors = np.linalg.norm(pred_3d - gt_3d, axis=1)
    add_error = per_joint_errors.mean()

    pose = {
        'rvec': rvec,
        'tvec': tvec,
        'R': R
    }

    return add_error, per_joint_errors, pose


def compute_reprojection_error(pred_2d, gt_3d, camera_matrix, dist_coeffs=None):
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

    projected_2d, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_2d = projected_2d.reshape(-1, 2)

    per_joint_errors = np.linalg.norm(projected_2d - pred_2d, axis=1)
    reproj_error = per_joint_errors.mean()

    return reproj_error, per_joint_errors
