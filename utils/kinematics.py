"""
UR10 Kinematics Module

Forward and inverse kinematics for the Universal Robots UR10 manipulator.
Calibrated to match Isaac Sim's coordinate system.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from typing import Tuple, Optional


# Link lengths derived from Isaac Sim calibration
UR10_PARAMS = {
    'base_height': 0.1273,
    'L1': 0.2208,           # Shoulder offset
    'L2_x': -0.1713,        # Shoulder to elbow X offset
    'L2_y': -0.6127,        # Upper arm length
    'L3_y': -0.5723,        # Forearm length
    'L4_x': 0.1157,         # Wrist 1 to wrist 2 offset
    'L5_z': -0.1157,        # Wrist 2 to end effector
}

# Keep old params for reference
UR10_DH_PARAMS = UR10_PARAMS


def forward_kinematics(joint_angles: np.ndarray, angles_in_radians: bool = True) -> np.ndarray:
    """
    Compute forward kinematics for UR10 matching Isaac Sim's coordinate system.

    Args:
        joint_angles: (6,) array of joint angles
        angles_in_radians: If True, input is radians. If False, input is degrees.

    Returns:
        (6, 3) array of 3D positions for each keypoint in base frame (meters)
        Order: Base, Shoulder, Elbow, Wrist3, Wrist4, Wrist5
    """
    # Convert to radians if needed
    if angles_in_radians:
        t = joint_angles.copy()
    else:
        t = np.radians(joint_angles)

    # Link lengths
    L1 = UR10_PARAMS['L1']
    L2_x = UR10_PARAMS['L2_x']
    L2_y = UR10_PARAMS['L2_y']
    L3_y = UR10_PARAMS['L3_y']
    L4_x = UR10_PARAMS['L4_x']
    L5_z = UR10_PARAMS['L5_z']
    base_height = UR10_PARAMS['base_height']

    positions = [np.array([0, 0, base_height])]  # Base

    # --- Joint 1 (Base) ---
    # Rotation around Z with 90° offset
    R1 = Rotation.from_rotvec((t[0] + np.pi/2) * np.array([0, 0, 1]))

    # Shoulder Keypoint
    shoulder_keypoint = positions[0] + R1.apply([L1, 0, 0])
    positions.append(shoulder_keypoint)

    # --- Joint 2 (Shoulder) ---
    # Rotation around X
    R2 = R1 * Rotation.from_euler('x', t[1])

    # Forearm (Elbow) Position
    forearm_pos = shoulder_keypoint + R1.apply([L2_x, 0, 0]) + R2.apply([0, L2_y, 0])
    positions.append(forearm_pos)

    # --- Joint 3 (Elbow) ---
    # Rotation around X
    R3 = R2 * Rotation.from_euler('x', t[2])

    # Wrist 1 Position (Wrist3 in our naming)
    wrist1_pos = forearm_pos + R3.apply([0, L3_y, 0])
    positions.append(wrist1_pos)

    # --- Joint 4 (Wrist 1) ---
    # Rotation around X
    R4 = R3 * Rotation.from_euler('x', t[3])

    # Wrist 2 Position (Wrist4 in our naming)
    # Link is along X, unaffected by t[3] roll
    wrist2_pos = wrist1_pos + R3.apply([L4_x, 0, 0])
    positions.append(wrist2_pos)

    # --- Joint 5 (Wrist 2) ---
    # Rotation around Z (Roll)
    # The link to the next joint is along Z, so this rotation is around the link axis.
    R5 = R4 * Rotation.from_euler('z', t[4])

    # End effector position (Wrist5 in our naming)
    end_pos = wrist2_pos + R5.apply([0, 0, L5_z])
    positions.append(end_pos)

    return np.array(positions)


def get_joint_positions(joint_angles: np.ndarray, include_base: bool = True, angles_in_radians: bool = True) -> np.ndarray:
    """
    Get the 3D positions of the robot joints matching the keypoint convention.

    Keypoint order: Base, Shoulder, Elbow, Wrist3, Wrist4, Wrist5

    Args:
        joint_angles: (6,) array of joint angles
        include_base: If True, returns 6 positions. If False, returns 5 (no base).
        angles_in_radians: If True, input is radians. If False, input is degrees.

    Returns:
        (6, 3) or (5, 3) array of 3D joint positions in meters
    """
    positions = forward_kinematics(joint_angles, angles_in_radians=angles_in_radians)

    if include_base:
        return positions  # All 6 positions
    else:
        return positions[1:]  # Skip base


def project_3d_to_2d(points_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """
    Project 3D points to 2D using pinhole camera model.

    Assumes points are already in camera frame (z forward, x right, y down).

    Args:
        points_3d: (N, 3) array of 3D points in camera frame
        camera_matrix: (3, 3) camera intrinsic matrix

    Returns:
        (N, 2) array of 2D pixel coordinates
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    points_2d = np.zeros((len(points_3d), 2))

    for i, p in enumerate(points_3d):
        if p[2] > 0:  # Point in front of camera
            points_2d[i, 0] = fx * p[0] / p[2] + cx
            points_2d[i, 1] = fy * p[1] / p[2] + cy
        else:
            points_2d[i] = np.nan

    return points_2d


def solve_ik_reprojection(
    target_2d: np.ndarray,
    camera_matrix: np.ndarray,
    robot_to_camera: np.ndarray,
    initial_angles: Optional[np.ndarray] = None,
    joint_limits: Optional[np.ndarray] = None,
    max_iterations: int = 100,
) -> Tuple[np.ndarray, float, bool]:
    """
    Solve inverse kinematics by minimizing 2D reprojection error.

    Finds joint angles θ such that:
        project(transform(FK(θ))) ≈ target_2d

    Args:
        target_2d: (6, 2) target 2D keypoints in pixels
        camera_matrix: (3, 3) camera intrinsic matrix
        robot_to_camera: (4, 4) transformation from robot base to camera frame
        initial_angles: (6,) initial guess for joint angles (default: zeros)
        joint_limits: (6, 2) array of [min, max] for each joint (radians)
        max_iterations: Maximum optimization iterations

    Returns:
        angles: (6,) estimated joint angles in radians
        error: Final reprojection error (pixels, mean per joint)
        success: Whether optimization converged
    """
    if initial_angles is None:
        initial_angles = np.zeros(6)

    if joint_limits is None:
        # UR10 joint limits (radians) - approximately ±2π for all joints
        joint_limits = np.array([
            [-2*np.pi, 2*np.pi],  # Base
            [-2*np.pi, 2*np.pi],  # Shoulder
            [-2*np.pi, 2*np.pi],  # Elbow
            [-2*np.pi, 2*np.pi],  # Wrist 1
            [-2*np.pi, 2*np.pi],  # Wrist 2
            [-2*np.pi, 2*np.pi],  # Wrist 3
        ])

    def objective(angles):
        """Reprojection error objective function."""
        # Get 3D positions in robot frame
        positions_robot = get_joint_positions(angles, include_base=True)

        # Transform to camera frame
        positions_homo = np.hstack([positions_robot, np.ones((6, 1))])
        positions_camera = (robot_to_camera @ positions_homo.T).T[:, :3]

        # Project to 2D
        projected_2d = project_3d_to_2d(positions_camera, camera_matrix)

        # Compute reprojection error (sum of squared distances)
        error = np.sum((projected_2d - target_2d) ** 2)

        return error

    # Optimize
    bounds = [(joint_limits[i, 0], joint_limits[i, 1]) for i in range(6)]

    result = minimize(
        objective,
        initial_angles,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iterations, 'ftol': 1e-8}
    )

    final_angles = result.x
    final_error = np.sqrt(result.fun / 6)  # RMSE per joint

    return final_angles, final_error, result.success


def estimate_robot_to_camera_transform(
    gt_2d: np.ndarray,
    gt_3d: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """
    Estimate the transformation from robot base frame to camera frame
    using ground truth 2D-3D correspondences.

    Uses cv2.solvePnP internally.

    Args:
        gt_2d: (N, 2) ground truth 2D keypoints
        gt_3d: (N, 3) ground truth 3D positions (in robot frame)
        camera_matrix: (3, 3) camera intrinsic matrix

    Returns:
        (4, 4) homogeneous transformation matrix (robot to camera)
    """
    import cv2

    # solvePnP gives us camera-to-object transform
    # We want object(robot)-to-camera
    success, rvec, tvec = cv2.solvePnP(
        gt_3d.astype(np.float32).reshape(-1, 1, 3),
        gt_2d.astype(np.float32).reshape(-1, 1, 2),
        camera_matrix.astype(np.float32),
        np.zeros(5, dtype=np.float32),
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return np.eye(4)

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    return T


def solve_ik_from_2d(
    pred_2d: np.ndarray,
    gt_2d: np.ndarray,
    gt_3d: np.ndarray,
    gt_angles_deg: np.ndarray,
    camera_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Full IK pipeline: from predicted 2D keypoints to estimated joint angles and 3D positions.

    This function:
    1. Estimates robot-to-camera transform from GT 2D-3D correspondences
    2. Solves IK using predicted 2D keypoints
    3. Computes FK to get estimated 3D positions

    Args:
        pred_2d: (6, 2) predicted 2D keypoints from model
        gt_2d: (6, 2) ground truth 2D keypoints (for transform estimation)
        gt_3d: (6, 3) ground truth 3D positions (for transform estimation)
        gt_angles_deg: (6,) ground truth joint angles in DEGREES (as initial guess)
        camera_matrix: (3, 3) camera intrinsic matrix

    Returns:
        estimated_angles_deg: (6,) estimated joint angles in DEGREES
        estimated_3d: (6, 3) estimated 3D positions in robot frame
        reproj_error: reprojection error in pixels
    """
    # Convert GT angles to radians for IK optimization
    gt_angles_rad = np.radians(gt_angles_deg)

    # Step 1: Estimate robot-to-camera transform from GT
    robot_to_camera = estimate_robot_to_camera_transform(gt_2d, gt_3d, camera_matrix)

    # Step 2: Solve IK using predicted 2D (internally uses radians)
    estimated_angles_rad, reproj_error, success = solve_ik_reprojection(
        target_2d=pred_2d,
        camera_matrix=camera_matrix,
        robot_to_camera=robot_to_camera,
        initial_angles=gt_angles_rad,  # Use GT as initial guess
    )

    # Step 3: Compute FK to get 3D positions
    estimated_3d = get_joint_positions(estimated_angles_rad, include_base=True, angles_in_radians=True)

    # Convert estimated angles back to degrees for output
    estimated_angles_deg = np.degrees(estimated_angles_rad)

    return estimated_angles_deg, estimated_3d, reproj_error


def compute_angle_error(estimated_angles: np.ndarray, gt_angles: np.ndarray) -> np.ndarray:
    """
    Compute angle errors with proper circular wrapping.

    Args:
        estimated_angles: (6,) estimated joint angles in radians
        gt_angles: (6,) ground truth joint angles in radians

    Returns:
        (6,) angle errors in radians (always positive, wrapped to [0, π])
    """
    diff = estimated_angles - gt_angles
    # Proper circular difference using atan2
    wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return np.abs(wrapped)


def compute_add_error_ik(
    pred_2d: np.ndarray,
    gt_2d: np.ndarray,
    gt_3d: np.ndarray,
    gt_angles_deg: np.ndarray,
    camera_matrix: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Compute ADD error using full IK pipeline.

    This is the proper way to evaluate 3D pose accuracy for articulated robots.

    Args:
        pred_2d: (6, 2) predicted 2D keypoints
        gt_2d: (6, 2) ground truth 2D keypoints
        gt_3d: (6, 3) ground truth 3D positions
        gt_angles_deg: (6,) ground truth joint angles in DEGREES
        camera_matrix: (3, 3) camera intrinsic matrix

    Returns:
        add_error: Mean ADD error in meters
        per_joint_errors: (6,) per-joint 3D errors in meters
        angle_errors_deg: (6,) per-joint angle errors in DEGREES
        reproj_error: Final reprojection error in pixels
    """
    # Solve IK (returns angles in degrees)
    estimated_angles_deg, estimated_3d, reproj_error = solve_ik_from_2d(
        pred_2d, gt_2d, gt_3d, gt_angles_deg, camera_matrix
    )

    # Compute 3D position errors (ADD)
    per_joint_errors = np.linalg.norm(estimated_3d - gt_3d, axis=1)
    add_error = per_joint_errors.mean()

    # Compute angle errors with proper wrapping (convert to radians for wrapping, then back to degrees)
    angle_errors_rad = compute_angle_error(
        np.radians(estimated_angles_deg),
        np.radians(gt_angles_deg)
    )
    angle_errors_deg = np.degrees(angle_errors_rad)

    return add_error, per_joint_errors, angle_errors_deg, reproj_error


if __name__ == '__main__':
    # Quick test
    print("UR10 Kinematics Test (Isaac Sim calibrated)")
    print("=" * 50)

    # Test forward kinematics at home position (all zeros)
    home_angles = np.zeros(6)
    positions = get_joint_positions(home_angles)

    print("\nJoint positions at home (all angles = 0):")
    joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist3', 'Wrist4', 'Wrist5']
    for name, pos in zip(joint_names, positions):
        print(f"  {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")

    # Test at a different configuration
    test_angles = np.array([0, -np.pi/4, np.pi/2, -np.pi/4, np.pi/2, 0])
    positions = get_joint_positions(test_angles)

    print("\nJoint positions at test configuration:")
    for name, pos in zip(joint_names, positions):
        print(f"  {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")

    # Compute reach (end effector distance from base)
    reach = np.linalg.norm(positions[-1])
    print(f"\nEnd effector reach: {reach:.4f} m")
