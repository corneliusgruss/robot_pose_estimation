"""
Utilities for Robot Pose Estimation
"""

from .visualization import (
    denormalize_image,
    visualize_sample,
    visualize_pipeline_result,
    visualize_bbox_predictions,
    plot_training_history,
    plot_error_distribution,
)

from .metrics import (
    compute_pixel_error,
    compute_bbox_iou,
    AverageMeter,
    evaluate_keypoint_model,
    evaluate_bbox_model,
)

from .pose_3d import (
    compute_add_error_scaled,
    compute_add_error_pnp,
    compute_reprojection_error,
    compute_auc,
)

from .training import (
    train_one_epoch_keypoints,
    train_one_epoch_bbox,
    train_stage1,
    train_stage2,
)

__all__ = [
    # Visualization
    'denormalize_image',
    'visualize_sample',
    'visualize_pipeline_result',
    'visualize_bbox_predictions',
    'plot_training_history',
    'plot_error_distribution',
    # Metrics (2D)
    'compute_pixel_error',
    'compute_bbox_iou',
    'AverageMeter',
    'evaluate_keypoint_model',
    'evaluate_bbox_model',
    # Pose 3D
    'compute_add_error_scaled',
    'compute_add_error_pnp',
    'compute_reprojection_error',
    'compute_auc',
    # Training
    'train_one_epoch_keypoints',
    'train_one_epoch_bbox',
    'train_stage1',
    'train_stage2',
]
