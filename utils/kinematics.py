"""
UR10 Kinematics Module

Forward and inverse kinematics for the Universal Robots UR10 manipulator.
Used to convert between joint angles and 3D joint positions.

DH Parameters source: Universal Robots UR10 specification
https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional


# UR10 DH Parameters (meters, radians)
# Modified DH convention: [a, d, alpha]
# Joint i: a_i (link length), d_i (link offset), alpha_i (link twist)
UR10_DH_PARAMS = {
    'd1': 0.1273,      # Base height
    'a2': -0.612,      # Upper arm length (negative in DH convention)
    'a3': -0.5723,     # Forearm length (negative in DH convention)
    'd4': 0.163941,    # Wrist 1 offset
    'd5': 0.1157,      # Wrist 2 offset
    'd6': 0.0922,      # Wrist 3 offset (to end effector)
}

# Full DH table: [a, d, alpha] for each joint
# Using standard DH convention
UR10_DH_TABLE = np.array([
    [0,                  UR10_DH_PARAMS['d1'], np.pi/2],   # Joint 1
    [UR10_DH_PARAMS['a2'], 0,                  0],         # Joint 2
    [UR10_DH_PARAMS['a3'], 0,                  0],         # Joint 3
    [0,                  UR10_DH_PARAMS['d4'], np.pi/2],   # Joint 4
    [0,                  UR10_DH_PARAMS['d5'], -np.pi/2],  # Joint 5
    [0,                  UR10_DH_PARAMS['d6'], 0],         # Joint 6
])


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Compute the DH transformation matrix for a single joint.

    Uses standard DH convention:
    T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)

    Args:
        theta: Joint angle (radians)
        d: Link offset along z
        a: Link length along x
        alpha: Link twist around x

    Returns:
        4x4 homogeneous transformation matrix
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1]
    ])


def forward_kinematics(joint_angles: np.ndarray) -> np.ndarray:
    """
    Compute forward kinematics for UR10.

    Returns the 3D position of each joint frame origin.

    Args:
        joint_angles: (6,) array of joint angles in radians

    Returns:
        (6, 3) array of 3D positions for each joint in base frame (meters)
    """
    assert len(joint_angles) == 6, "UR10 has 6 joints"

    positions = []
    T = np.eye(4)  # Start at base frame

    for i in range(6):
        a, d, alpha = UR10_DH_TABLE[i]
        theta = joint_angles[i]

        # Apply DH transform for this joint
        T = T @ dh_transform(theta, d, a, alpha)

        # Extract position (translation component)
        positions.append(T[:3, 3].copy())

    return np.array(positions)


def forward_kinematics_all_frames(joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute forward kinematics returning all frame origins and rotations.

    Args:
        joint_angles: (6,) array of joint angles in radians

    Returns:
        positions: (7, 3) array - base + 6 joint positions
        rotations: (7, 3, 3) array - rotation matrices for each frame
    """
    positions = [np.array([0, 0, 0])]  # Base at origin
    rotations = [np.eye(3)]  # Base rotation

    T = np.eye(4)

    for i in range(6):
        a, d, alpha = UR10_DH_TABLE[i]
        theta = joint_angles[i]

        T = T @ dh_transform(theta, d, a, alpha)

        positions.append(T[:3, 3].copy())
        rotations.append(T[:3, :3].copy())

    return np.array(positions), np.array(rotations)


def get_joint_positions(joint_angles: np.ndarray, include_base: bool = True) -> np.ndarray:
    """
    Get the 3D positions of the robot joints matching the keypoint convention.

    Keypoint order: Base, Shoulder, Elbow, Wrist3, Wrist4, Wrist5

    Args:
        joint_angles: (6,) array of joint angles in radians
        include_base: If True, includes base position (returns 6 positions)
                     If False, returns only the 6 moving joint positions

    Returns:
        (6, 3) array of 3D joint positions in meters
    """
    positions, _ = forward_kinematics_all_frames(joint_angles)

    if include_base:
        # Return: Base, J1, J2, J3, J4, J5 (frames 0-5)
        # This matches: Base, Shoulder, Elbow, Wrist3, Wrist4, Wrist5
        return positions[:6]
    else:
        # Return only moving joint positions (frames 1-6)
        return positions[1:7]


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
    # Homogeneous projection: p = K @ P / P_z
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
    gt_angles: np.ndarray,
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
        gt_angles: (6,) ground truth joint angles (as initial guess)
        camera_matrix: (3, 3) camera intrinsic matrix

    Returns:
        estimated_angles: (6,) estimated joint angles in radians
        estimated_3d: (6, 3) estimated 3D positions in robot frame
        reproj_error: reprojection error in pixels
    """
    # Step 1: Estimate robot-to-camera transform from GT
    robot_to_camera = estimate_robot_to_camera_transform(gt_2d, gt_3d, camera_matrix)

    # Step 2: Solve IK using predicted 2D
    estimated_angles, reproj_error, success = solve_ik_reprojection(
        target_2d=pred_2d,
        camera_matrix=camera_matrix,
        robot_to_camera=robot_to_camera,
        initial_angles=gt_angles,  # Use GT as initial guess for faster convergence
    )

    # Step 3: Compute FK to get 3D positions
    estimated_3d = get_joint_positions(estimated_angles, include_base=True)

    return estimated_angles, estimated_3d, reproj_error


def compute_add_error_ik(
    pred_2d: np.ndarray,
    gt_2d: np.ndarray,
    gt_3d: np.ndarray,
    gt_angles: np.ndarray,
    camera_matrix: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Compute ADD error using full IK pipeline.

    This is the proper way to evaluate 3D pose accuracy for articulated robots.

    Args:
        pred_2d: (6, 2) predicted 2D keypoints
        gt_2d: (6, 2) ground truth 2D keypoints
        gt_3d: (6, 3) ground truth 3D positions
        gt_angles: (6,) ground truth joint angles
        camera_matrix: (3, 3) camera intrinsic matrix

    Returns:
        add_error: Mean ADD error in meters
        per_joint_errors: (6,) per-joint 3D errors in meters
        angle_errors: (6,) per-joint angle errors in radians
        reproj_error: Final reprojection error in pixels
    """
    # Solve IK
    estimated_angles, estimated_3d, reproj_error = solve_ik_from_2d(
        pred_2d, gt_2d, gt_3d, gt_angles, camera_matrix
    )

    # Compute 3D position errors (ADD)
    per_joint_errors = np.linalg.norm(estimated_3d - gt_3d, axis=1)
    add_error = per_joint_errors.mean()

    # Compute angle errors
    angle_errors = np.abs(estimated_angles - gt_angles)
    # Wrap to [-pi, pi]
    angle_errors = np.minimum(angle_errors, 2*np.pi - angle_errors)

    return add_error, per_joint_errors, angle_errors, reproj_error


if __name__ == '__main__':
    # Quick test
    print("UR10 Kinematics Test")
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
    print(f"Expected max reach: ~1.3 m (UR10 spec)")
