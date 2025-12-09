import numpy as np

TRAIN_DIRS = [
    '../../Datasets/Train5000v2',
    '../../Datasets/Train1000v2-Room',
    '../../Datasets/Train1000v2-Warehouse',
    '../../Datasets/Train1000v2-Hospital',
]
TEST_DIR = '../../Datasets/Test2000v2'
CHECKPOINT_DIR = 'checkpoints'

ORIG_SIZE = 1080
STAGE1_SIZE = 256
STAGE2_SIZE = 512
BBOX_PADDING = 100

STAGE1_BACKBONE = 'resnet18'
STAGE2_BACKBONE = 'resnet34'
NUM_JOINTS = 6

BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4

EPOCHS_STAGE1 = 30
EPOCHS_STAGE2 = 50

SAVE_EVERY = 5
EARLY_STOP_PATIENCE = 15

JOINT_NAMES = ['Base', 'Shoulder', 'Elbow', 'Wrist3', 'Wrist4', 'Wrist5']

KEYPOINT_COLS_2D = [
    ('2D_J0_Base_x', '2D_J0_Base_y'),
    ('2D_J1_Shoulder_x', '2D_J1_Shoulder_y'),
    ('2D_J2_Elbow_x', '2D_J2_Elbow_y'),
    ('2D_J3_Wrist_x', '2D_J3_Wrist_y'),
    ('2D_J4_Wrist_x', '2D_J4_Wrist_y'),
    ('2D_J5_Wrist_x', '2D_J5_Wrist_y'),
]

KEYPOINT_COLS_3D = [
    ('J0_Base_x', 'J0_Base_y', 'J0_Base_z'),
    ('J1_Shoulder_x', 'J1_Shoulder_y', 'J1_Shoulder_z'),
    ('J2_Elbow_x', 'J2_Elbow_y', 'J2_Elbow_z'),
    ('J3_Wrist_x', 'J3_Wrist_y', 'J3_Wrist_z'),
    ('J4_Wrist_x', 'J4_Wrist_y', 'J4_Wrist_z'),
    ('J5_Wrist_x', 'J5_Wrist_y', 'J5_Wrist_z'),
]

JOINT_ANGLE_COLS = [
    'J0_actual', 'J1_actual', 'J2_actual',
    'J3_actual', 'J4_actual', 'J5_actual'
]

CAMERA_MATRIX = np.array([
    [935.307424,   0.0,       540.0],
    [  0.0,      1281.77511,  540.0],
    [  0.0,        0.0,         1.0]
], dtype=np.float32)

DIST_COEFFS = np.zeros(5, dtype=np.float32)

ADD_THRESHOLD_MAX = 0.30
AUC_NUM_STEPS = 100

def get_config_dict():
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
