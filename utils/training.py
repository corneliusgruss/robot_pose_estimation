import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import os

from .metrics import (
    compute_pixel_error, compute_bbox_iou,
    AverageMeter, evaluate_keypoint_model, evaluate_bbox_model
)


def train_one_epoch_keypoints(model, loader, optimizer, device, img_size=512):
    model.train()
    loss_meter = AverageMeter()
    error_meter = AverageMeter()

    criterion = nn.MSELoss()

    for batch in loader:
        imgs = batch['img_stage2'].to(device)
        targets = batch['keypoints'].to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            px_error, _ = compute_pixel_error(preds, targets, img_size)

        loss_meter.update(loss.item(), imgs.size(0))
        error_meter.update(px_error, imgs.size(0))
        # print(f"loss: {loss.item():.4f}, error: {px_error:.2f}")

    return loss_meter.avg, error_meter.avg


def train_one_epoch_bbox(model, loader, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()

    criterion = nn.MSELoss()

    for batch in loader:
        imgs = batch['img_stage1'].to(device)
        targets = batch['bbox'].to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            iou = compute_bbox_iou(preds, targets)

        loss_meter.update(loss.item(), imgs.size(0))
        iou_meter.update(iou, imgs.size(0))

    return loss_meter.avg, iou_meter.avg


def train_stage2(
    model,
    train_loader,
    val_loader,
    device,
    epochs=50,
    lr=1e-4,
    save_dir='checkpoints',
    save_every=5,
    early_stop_patience=15,
    img_size=512
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    history = {
        'train_loss': [], 'train_error': [],
        'val_loss': [], 'val_error': [],
        'joint_errors': [], 'lr': []
    }

    best_val_error = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    print(f"\nTraining Stage 2 for {epochs} epochs")
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")
    print("-" * 70)

    for epoch in range(epochs):
        train_loss, train_error = train_one_epoch_keypoints(
            model, train_loader, optimizer, device, img_size
        )

        val_loss, val_error, joint_errors = evaluate_keypoint_model(
            model, val_loader, device, img_size
        )

        scheduler.step(val_error)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_error'].append(train_error)
        history['val_loss'].append(val_loss)
        history['val_error'].append(val_error)
        history['joint_errors'].append(joint_errors)
        history['lr'].append(current_lr)

        is_best = val_error < best_val_error
        if is_best:
            best_val_error = val_error
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': val_error,
                'history': history,
                'backbone': getattr(model, 'backbone_name', 'resnet34'),
                'num_keypoints': model.num_keypoints,
            }, save_dir / 'stage2_best.pt')
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': val_error,
                'history': history,
            }, save_dir / f'stage2_epoch{epoch+1}.pt')

        marker = ' * BEST' if is_best else ''
        joint_str = ' '.join([f'{e:.1f}' for e in joint_errors])
        print(f"Epoch {epoch+1:3d} | "
              f"Train: {train_loss:.4f} / {train_error:.1f}px | "
              f"Val: {val_loss:.4f} / {val_error:.1f}px | "
              f"Joints: [{joint_str}]{marker}")

        if epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping: no improvement for {early_stop_patience} epochs")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("-" * 70)
    print(f"Best validation error: {best_val_error:.1f}px")

    return model, history


def train_stage1(
    model,
    train_loader,
    val_loader,
    device,
    epochs=30,
    lr=1e-4,
    save_dir='checkpoints',
    save_every=5,
    early_stop_patience=15
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    history = {
        'train_loss': [], 'train_iou': [],
        'val_loss': [], 'val_iou': [], 'lr': []
    }

    best_val_iou = 0
    best_model_state = None
    epochs_without_improvement = 0

    print(f"\nTraining Stage 1 for {epochs} epochs")
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")
    print("-" * 70)

    for epoch in range(epochs):
        train_loss, train_iou = train_one_epoch_bbox(
            model, train_loader, optimizer, device
        )

        val_loss, val_iou = evaluate_bbox_model(model, val_loader, device)

        scheduler.step(val_iou)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['lr'].append(current_lr)

        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'history': history,
                'backbone': getattr(model, 'backbone_name', 'resnet18'),
            }, save_dir / 'stage1_best.pt')
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'history': history,
            }, save_dir / f'stage1_epoch{epoch+1}.pt')

        marker = ' * BEST' if is_best else ''
        print(f"Epoch {epoch+1:3d} | "
              f"Train: {train_loss:.4f} / IoU {train_iou:.3f} | "
              f"Val: {val_loss:.4f} / IoU {val_iou:.3f}{marker}")

        if epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping: no improvement for {early_stop_patience} epochs")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("-" * 70)
    print(f"Best validation IoU: {best_val_iou:.3f}")

    return model, history
