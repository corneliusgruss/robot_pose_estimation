"""
Configuration for Robot Pose Estimation Project.
All hyperparameters in one place.
"""

import numpy as np

# =============================================================================
# PATHS - Update these for your environment
# =============================================================================
TRAIN_DIRS = [
    '../../Datasets/Train5000v2',
    '../../Datasets/Train1000v2-Room',
    '../../Datasets/Train1000v2-Warehouse',
    '../../Datasets/Train1000v2-Hospital',
]
TEST_DIR = '../../Datasets/Test2000v2'
CHECKPOINT_DIR = 'checkpoints'

# =============================================================================
# IMAGE SETTINGS
# =============================================================================
ORIG_SIZE = 1080              # Original image size
STAGE1_SIZE = 256             # Stage 1 input (bbox prediction)
STAGE2_SIZE = 512             # Stage 2 input (keypoint prediction)
BBOX_PADDING = 100            # Padding around keypoints for bbox

# =============================================================================
# MODEL SETTINGS
# =============================================================================
STAGE1_BACKBONE = 'resnet18'  # Lighter model for bbox
STAGE2_BACKBONE = 'resnet34'  # Heavier model for keypoints
NUM_JOINTS = 6

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4

EPOCHS_STAGE1 = 30
EPOCHS_STAGE2 = 50

# Checkpointing
SAVE_EVERY = 5                # Save checkpoint every N epochs
EARLY_STOP_PATIENCE = 15      # Stop if no improvement for N epochs

# =============================================================================
# JOINT DEFINITIONS
# =============================================================================
JOINT_NAMES = ['Base', 'Shoulder', 'Elbow', 'Wrist3', 'Wrist4', 'Wrist5']

# CSV column names for 2D keypoints
KEYPOINT_COLS_2D = [
    ('2D_J0_Base_x', '2D_J0_Base_y'),
    ('2D_J1_Shoulder_x', '2D_J1_Shoulder_y'),
    ('2D_J2_Elbow_x', '2D_J2_Elbow_y'),
    ('2D_J3_Wrist_x', '2D_J3_Wrist_y'),
    ('2D_J4_Wrist_x', '2D_J4_Wrist_y'),
    ('2D_J5_Wrist_x', '2D_J5_Wrist_y'),
]

# CSV column names for 3D positions
KEYPOINT_COLS_3D = [
    ('J0_Base_x', 'J0_Base_y', 'J0_Base_z'),
    ('J1_Shoulder_x', 'J1_Shoulder_y', 'J1_Shoulder_z'),
    ('J2_Elbow_x', 'J2_Elbow_y', 'J2_Elbow_z'),
    ('J3_Wrist_x', 'J3_Wrist_y', 'J3_Wrist_z'),
    ('J4_Wrist_x', 'J4_Wrist_y', 'J4_Wrist_z'),
    ('J5_Wrist_x', 'J5_Wrist_y', 'J5_Wrist_z'),
]

# =============================================================================
# CAMERA INTRINSICS
# These are estimated Isaac Sim defaults and may not be accurate.
# Currently NOT used for inference - the project outputs 2D keypoints.
# 3D error estimation uses a scaling approximation (see utils/pose_3d.py).
# =============================================================================
CAMERA_MATRIX = np.array([
    [600.0,   0.0, 540.0],  # fx, 0, cx
    [  0.0, 600.0, 540.0],  # 0, fy, cy
    [  0.0,   0.0,   1.0]
], dtype=np.float32)

DIST_COEFFS = np.zeros(5, dtype=np.float32)  # No distortion for synthetic

# =============================================================================
# METRICS SETTINGS
# =============================================================================
ADD_THRESHOLD_MAX = 0.30      # 30cm max threshold for AUC
AUC_NUM_STEPS = 100

# =============================================================================
# CONVENIENCE: Get config as dict (for saving with checkpoints)
# =============================================================================
def get_config_dict():
    """Return config as dictionary for serialization."""
    return {
        'train_dirs': TRAIN_DIRS,
        'test_dir': TEST_DIR,
        'orig_size': ORIG_SIZE,
        'stage1_size': STAGE1_SIZE,
        'stage2_size': STAGE2_SIZE,
        'bbox_padding': BBOX_PADDING,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'epochs_stage1': EPOCHS_STAGE1,
        'epochs_stage2': EPOCHS_STAGE2,
        'joint_names': JOINT_NAMES,
        'num_joints': NUM_JOINTS,
    }
