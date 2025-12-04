"""
Metrics for Robot Pose Estimation

Includes:
- 2D pixel error
- Bounding box IoU
- 3D ADD error
- AUC computation
"""

import torch
import numpy as np


# =============================================================================
# 2D METRICS
# =============================================================================

def compute_pixel_error(pred, target, img_size):
    """
    Compute per-joint pixel error.
    
    Args:
        pred: (batch, 12) predicted keypoints in [0,1]
        target: (batch, 12) ground truth keypoints in [0,1]
        img_size: size to convert to pixels
    
    Returns:
        mean_error: scalar, mean pixel error across all joints
        per_joint_error: (6,) mean pixel error per joint
    """
    pred_px = pred.view(-1, 6, 2) * img_size
    target_px = target.view(-1, 6, 2) * img_size
    
    # Euclidean distance per joint
    distances = torch.sqrt(((pred_px - target_px) ** 2).sum(dim=-1))  # (batch, 6)
    
    per_joint_error = distances.mean(dim=0)  # (6,)
    mean_error = distances.mean()
    
    return mean_error.item(), per_joint_error.detach().cpu().numpy()


def compute_bbox_iou(pred, target):
    """
    Compute IoU between predicted and target bboxes.
    
    Args:
        pred: (batch, 4) predicted bbox [x_min, y_min, x_max, y_max] in [0,1]
        target: (batch, 4) ground truth bbox
    
    Returns:
        mean_iou: scalar
    """
    # Intersection
    x_min = torch.max(pred[:, 0], target[:, 0])
    y_min = torch.max(pred[:, 1], target[:, 1])
    x_max = torch.min(pred[:, 2], target[:, 2])
    y_max = torch.min(pred[:, 3], target[:, 3])
    
    inter_w = (x_max - x_min).clamp(min=0)
    inter_h = (y_max - y_min).clamp(min=0)
    intersection = inter_w * inter_h
    
    # Union
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = pred_area + target_area - intersection
    
    iou = intersection / (union + 1e-6)
    return iou.mean().item()


# =============================================================================
# 3D METRICS
# =============================================================================

def compute_add_error_scaled(pred_2d, gt_2d, gt_3d):
    """
    Estimate 3D error using scale from GT correspondences.
    
    Avoids needing exact camera intrinsics by using the ratio of
    2D pixel distances to 3D spatial distances from ground truth.
    
    Args:
        pred_2d: (6, 2) predicted 2D keypoints in pixels
        gt_2d: (6, 2) ground truth 2D keypoints in pixels
        gt_3d: (6, 3) ground truth 3D positions in meters
    
    Returns:
        add_error: mean 3D spatial error in meters
        per_joint_errors: (6,) per-joint 3D errors in meters
        scale: estimated meters per pixel
    """
    # Compute pairwise distances in 2D and 3D for GT
    scales = []
    for i in range(6):
        for j in range(i+1, 6):
            dist_2d = np.linalg.norm(gt_2d[i] - gt_2d[j])
            dist_3d = np.linalg.norm(gt_3d[i] - gt_3d[j])
            
            if dist_2d > 10:  # Avoid degenerate cases
                scales.append(dist_3d / dist_2d)
    
    scale = np.median(scales) if scales else 0.003  # Fallback ~3mm/px
    
    # Compute 2D error and convert to 3D using scale
    errors_2d = np.linalg.norm(pred_2d - gt_2d, axis=1)
    errors_3d = errors_2d * scale
    
    return errors_3d.mean(), errors_3d, scale


def compute_auc(errors, max_threshold=0.30, num_steps=100):
    """
    Compute Area Under the Curve for ADD metric.
    
    Args:
        errors: array of ADD errors in meters
        max_threshold: maximum threshold (default 0.30m = 30cm)
        num_steps: number of threshold steps
    
    Returns:
        auc: Area under curve (0-1, higher is better)
        thresholds: array of thresholds used
        accuracies: accuracy at each threshold
    """
    thresholds = np.linspace(0, max_threshold, num_steps)
    accuracies = []
    
    for thresh in thresholds:
        acc = (errors <= thresh).mean()
        accuracies.append(acc)
    
    accuracies = np.array(accuracies)
    auc = np.trapz(accuracies, thresholds) / max_threshold
    
    return auc, thresholds, accuracies


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

class AverageMeter:
    """Track running average of a metric."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_keypoint_model(model, loader, device, img_size=512):
    """
    Evaluate keypoint model on a dataloader.
    
    Returns:
        loss: average MSE loss
        error: mean pixel error
        joint_errors: (6,) per-joint errors
    """
    model.eval()
    loss_meter = AverageMeter()
    error_meter = AverageMeter()
    joint_errors = np.zeros(6)
    
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch['img_stage2'].to(device)
            targets = batch['keypoints'].to(device)
            
            preds = model(imgs)
            loss = criterion(preds, targets)
            
            px_error, per_joint = compute_pixel_error(preds, targets, img_size)
            
            loss_meter.update(loss.item(), imgs.size(0))
            error_meter.update(px_error, imgs.size(0))
            joint_errors += per_joint * imgs.size(0)
    
    joint_errors /= len(loader.dataset)
    
    return loss_meter.avg, error_meter.avg, joint_errors


def evaluate_bbox_model(model, loader, device):
    """
    Evaluate bbox model on a dataloader.
    
    Returns:
        loss: average MSE loss
        iou: mean IoU
    """
    model.eval()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch['img_stage1'].to(device)
            targets = batch['bbox'].to(device)
            
            preds = model(imgs)
            loss = criterion(preds, targets)
            iou = compute_bbox_iou(preds, targets)
            
            loss_meter.update(loss.item(), imgs.size(0))
            iou_meter.update(iou, imgs.size(0))
    
    return loss_meter.avg, iou_meter.avg
