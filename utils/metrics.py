"""
Metrics for Robot Pose Estimation

Includes:
- 2D pixel error
- Bounding box IoU
- Evaluation helpers

For 3D metrics (ADD error, AUC), see pose_3d.py
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
