import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from typing import Tuple, Optional


# Link lengths from Isaac Sim calibration
UR10_PARAMS = {
    'base_height': 0.1273,
    'L1': 0.2208,
    'L2_x': -0.1713,
    'L2_y': -0.6127,
    'L3_y': -0.5723,
    'L4_x': 0.1157,
    'L5_z': -0.1157,
}

UR10_DH_PARAMS = UR10_PARAMS


def forward_kinematics(joint_angles: np.ndarray, angles_in_radians: bool = True) -> np.ndarray:
    if angles_in_radians:
        t = joint_angles.copy()
    else:
        t = np.radians(joint_angles)

    L1 = UR10_PARAMS['L1']
    L2_x = UR10_PARAMS['L2_x']
    L2_y = UR10_PARAMS['L2_y']
    L3_y = UR10_PARAMS['L3_y']
    L4_x = UR10_PARAMS['L4_x']
    L5_z = UR10_PARAMS['L5_z']
    base_height = UR10_PARAMS['base_height']

    positions = [np.array([0, 0, base_height])]

    R1 = Rotation.from_rotvec((t[0] + np.pi/2) * np.array([0, 0, 1]))
    shoulder_keypoint = positions[0] + R1.apply([L1, 0, 0])
    positions.append(shoulder_keypoint)

    R2 = R1 * Rotation.from_euler('x', t[1])
    forearm_pos = shoulder_keypoint + R1.apply([L2_x, 0, 0]) + R2.apply([0, L2_y, 0])
    positions.append(forearm_pos)

    R3 = R2 * Rotation.from_euler('x', t[2])
    wrist1_pos = forearm_pos + R3.apply([0, L3_y, 0])
    positions.append(wrist1_pos)

    R4 = R3 * Rotation.from_euler('x', t[3])
    wrist2_pos = wrist1_pos + R3.apply([L4_x, 0, 0])
    positions.append(wrist2_pos)

    R5 = R4 * Rotation.from_euler('z', t[4])
    end_pos = wrist2_pos + R5.apply([0, 0, L5_z])
    positions.append(end_pos)

    return np.array(positions)


def get_joint_positions(joint_angles: np.ndarray, include_base: bool = True, angles_in_radians: bool = True) -> np.ndarray:
    positions = forward_kinematics(joint_angles, angles_in_radians=angles_in_radians)
    if include_base:
        return positions
    else:
        return positions[1:]


def project_3d_to_2d(points_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    points_2d = np.zeros((len(points_3d), 2))

    for i, p in enumerate(points_3d):
        if p[2] > 0:
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

    if initial_angles is None:
        initial_angles = np.zeros(6)

    if joint_limits is None:
        joint_limits = np.array([
            [-2*np.pi, 2*np.pi],
            [-2*np.pi, 2*np.pi],
            [-2*np.pi, 2*np.pi],
            [-2*np.pi, 2*np.pi],
            [-2*np.pi, 2*np.pi],
            [-2*np.pi, 2*np.pi],
        ])

    def objective(angles):
        positions_robot = get_joint_positions(angles, include_base=True)
        positions_homo = np.hstack([positions_robot, np.ones((6, 1))])
        positions_camera = (robot_to_camera @ positions_homo.T).T[:, :3]
        projected_2d = project_3d_to_2d(positions_camera, camera_matrix)
        error = np.sum((projected_2d - target_2d) ** 2)
        return error

    bounds = [(joint_limits[i, 0], joint_limits[i, 1]) for i in range(6)]

    result = minimize(
        objective,
        initial_angles,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iterations, 'ftol': 1e-8}
    )

    final_angles = result.x
    final_error = np.sqrt(result.fun / 6)

    return final_angles, final_error, result.success


def estimate_robot_to_camera_transform(
    gt_2d: np.ndarray,
    gt_3d: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    import cv2

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

    gt_angles_rad = np.radians(gt_angles_deg)

    robot_to_camera = estimate_robot_to_camera_transform(gt_2d, gt_3d, camera_matrix)

    estimated_angles_rad, reproj_error, success = solve_ik_reprojection(
        target_2d=pred_2d,
        camera_matrix=camera_matrix,
        robot_to_camera=robot_to_camera,
        initial_angles=gt_angles_rad,
    )

    estimated_3d = get_joint_positions(estimated_angles_rad, include_base=True, angles_in_radians=True)
    estimated_angles_deg = np.degrees(estimated_angles_rad)

    return estimated_angles_deg, estimated_3d, reproj_error


def compute_angle_error(estimated_angles: np.ndarray, gt_angles: np.ndarray) -> np.ndarray:
    diff = estimated_angles - gt_angles
    wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return np.abs(wrapped)


def compute_add_error_ik(
    pred_2d: np.ndarray,
    gt_2d: np.ndarray,
    gt_3d: np.ndarray,
    gt_angles_deg: np.ndarray,
    camera_matrix: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, float]:

    estimated_angles_deg, estimated_3d, reproj_error = solve_ik_from_2d(
        pred_2d, gt_2d, gt_3d, gt_angles_deg, camera_matrix
    )

    per_joint_errors = np.linalg.norm(estimated_3d - gt_3d, axis=1)
    add_error = per_joint_errors.mean()

    angle_errors_rad = compute_angle_error(
        np.radians(estimated_angles_deg),
        np.radians(gt_angles_deg)
    )
    angle_errors_deg = np.degrees(angle_errors_rad)

    return add_error, per_joint_errors, angle_errors_deg, reproj_error


if __name__ == '__main__':
    print("UR10 Kinematics Test")
    print("=" * 50)

    home_angles = np.zeros(6)
    positions = get_joint_positions(home_angles)

    print("\nJoint positions at home (all angles = 0):")
    joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist3', 'Wrist4', 'Wrist5']
    for name, pos in zip(joint_names, positions):
        print(f"  {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")

    test_angles = np.array([0, -np.pi/4, np.pi/2, -np.pi/4, np.pi/2, 0])
    positions = get_joint_positions(test_angles)

    print("\nJoint positions at test configuration:")
    for name, pos in zip(joint_names, positions):
        print(f"  {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")

    reach = np.linalg.norm(positions[-1])
    print(f"\nEnd effector reach: {reach:.4f} m")
