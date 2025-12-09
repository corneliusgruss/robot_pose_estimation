import torch
import numpy as np


def compute_pixel_error(pred, target, img_size):
    pred_px = pred.view(-1, 6, 2) * img_size
    target_px = target.view(-1, 6, 2) * img_size

    distances = torch.sqrt(((pred_px - target_px) ** 2).sum(dim=-1))

    per_joint_error = distances.mean(dim=0)
    mean_error = distances.mean()

    return mean_error.item(), per_joint_error.detach().cpu().numpy()


def compute_bbox_iou(pred, target):
    x_min = torch.max(pred[:, 0], target[:, 0])
    y_min = torch.max(pred[:, 1], target[:, 1])
    x_max = torch.min(pred[:, 2], target[:, 2])
    y_max = torch.min(pred[:, 3], target[:, 3])

    inter_w = (x_max - x_min).clamp(min=0)
    inter_h = (y_max - y_min).clamp(min=0)
    intersection = inter_w * inter_h

    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = pred_area + target_area - intersection

    iou = intersection / (union + 1e-6)
    return iou.mean().item()


class AverageMeter:

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
