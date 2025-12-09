import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def denormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img


def visualize_sample(sample, pred_keypoints=None, config=None):
    if config is None:
        stage1_size = 256
        stage2_size = 512
        joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist3', 'Wrist4', 'Wrist5']
    elif hasattr(config, 'STAGE1_SIZE'):
        stage1_size = config.STAGE1_SIZE
        stage2_size = config.STAGE2_SIZE
        joint_names = config.JOINT_NAMES
    else:
        stage1_size = config.get('stage1_size', 256)
        stage2_size = config.get('stage2_size', 512)
        joint_names = config.get('joint_names',
            ['Base', 'Shoulder', 'Elbow', 'Wrist3', 'Wrist4', 'Wrist5'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    img_full = denormalize_image(sample['img_stage1'])
    axes[0].imshow(img_full)

    bbox = sample['bbox'].numpy() * stage1_size
    x_min, y_min, x_max, y_max = bbox
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                          fill=False, edgecolor='lime', linewidth=2)
    axes[0].add_patch(rect)
    axes[0].set_title('Stage 1: Full Image + BBox')
    axes[0].axis('off')

    img_crop = denormalize_image(sample['img_stage2'])
    axes[1].imshow(img_crop)

    kp_gt = sample['keypoints'].numpy().reshape(-1, 2) * stage2_size
    axes[1].scatter(kp_gt[:, 0], kp_gt[:, 1], c='lime', s=100, marker='o',
                    label='Ground Truth', edgecolors='black', linewidths=1)

    if pred_keypoints is not None:
        kp_pred = pred_keypoints.reshape(-1, 2) * stage2_size
        axes[1].scatter(kp_pred[:, 0], kp_pred[:, 1], c='red', s=100, marker='x',
                        label='Predicted', linewidths=2)

    for i, name in enumerate(joint_names):
        axes[1].annotate(name, (kp_gt[i, 0] + 5, kp_gt[i, 1] - 5),
                         color='white', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    axes[1].legend(loc='upper right')
    axes[1].set_title('Stage 2: Cropped Image + Keypoints')
    axes[1].axis('off')

    plt.tight_layout()
    return fig


def visualize_pipeline_result(image_path, pipeline, gt_keypoints=None, config=None):
    if config is None:
        joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist3', 'Wrist4', 'Wrist5']
    elif hasattr(config, 'JOINT_NAMES'):
        joint_names = config.JOINT_NAMES
    else:
        joint_names = config.get('joint_names',
            ['Base', 'Shoulder', 'Elbow', 'Wrist3', 'Wrist4', 'Wrist5'])

    img = Image.open(image_path).convert('RGB')
    keypoints_2d, bbox = pipeline.predict(img)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)

    rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                          fill=False, edgecolor='yellow', linewidth=2, label='Pred BBox')
    ax.add_patch(rect)

    for i in range(5):
        ax.plot([keypoints_2d[i, 0], keypoints_2d[i+1, 0]],
                [keypoints_2d[i, 1], keypoints_2d[i+1, 1]],
                'w-', linewidth=2, alpha=0.8)

    ax.scatter(keypoints_2d[:, 0], keypoints_2d[:, 1], c='red', s=150, marker='x',
               linewidths=3, label='Predicted', zorder=5)

    if gt_keypoints is not None:
        gt_kp = np.array(gt_keypoints).reshape(6, 2)
        ax.scatter(gt_kp[:, 0], gt_kp[:, 1], c='lime', s=100, marker='o',
                   edgecolors='black', linewidths=1, label='Ground Truth', zorder=4)

    for i, name in enumerate(joint_names):
        ax.annotate(name, (keypoints_2d[i, 0] + 10, keypoints_2d[i, 1] - 10),
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

    ax.legend(loc='upper right', fontsize=12)
    ax.set_title('End-to-End Pipeline Prediction', fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    return fig


def visualize_bbox_predictions(dataset, model, device, num_samples=4, config=None):
    if config is None:
        stage1_size = 256
    elif hasattr(config, 'STAGE1_SIZE'):
        stage1_size = config.STAGE1_SIZE
    else:
        stage1_size = config.get('stage1_size', 256)

    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
    if num_samples == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        sample = dataset[idx]

        with torch.no_grad():
            img = sample['img_stage1'].unsqueeze(0).to(device)
            pred_bbox = model(img).cpu().numpy()[0]

        gt_bbox = sample['bbox'].numpy()

        img_display = denormalize_image(sample['img_stage1'])
        ax.imshow(img_display)

        gt = gt_bbox * stage1_size
        rect_gt = plt.Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1],
                                  fill=False, edgecolor='lime', linewidth=2)
        ax.add_patch(rect_gt)

        pred = pred_bbox * stage1_size
        rect_pred = plt.Rectangle((pred[0], pred[1]), pred[2]-pred[0], pred[3]-pred[1],
                                    fill=False, edgecolor='red', linewidth=2, linestyle='--')
        ax.add_patch(rect_pred)

        ax.set_title(f'Sample {idx}')
        ax.axis('off')

    plt.tight_layout()
    return fig


def plot_training_history(history, title_prefix=''):
    has_error = 'train_error' in history
    has_iou = 'train_iou' in history
    has_joint_errors = 'joint_errors' in history

    n_plots = 2 + (1 if has_joint_errors else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title_prefix}Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if has_error:
        axes[1].plot(epochs, history['train_error'], label='Train')
        axes[1].plot(epochs, history['val_error'], label='Val')
        axes[1].set_ylabel('Pixel Error')
        axes[1].set_title(f'{title_prefix}Pixel Error')
    elif has_iou:
        axes[1].plot(epochs, history['train_iou'], label='Train')
        axes[1].plot(epochs, history['val_iou'], label='Val')
        axes[1].set_ylabel('IoU')
        axes[1].set_title(f'{title_prefix}IoU')

    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if has_joint_errors:
        joint_errors = np.array(history['joint_errors'])
        joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist3', 'Wrist4', 'Wrist5']
        for i, name in enumerate(joint_names):
            axes[2].plot(epochs, joint_errors[:, i], label=name)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Pixel Error')
        axes[2].set_title(f'{title_prefix}Per-Joint Error')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_error_distribution(errors, title='Error Distribution'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.median(errors), color='red', linestyle='--',
                    label=f'Median: {np.median(errors):.1f}')
    axes[0].axvline(np.mean(errors), color='orange', linestyle='--',
                    label=f'Mean: {np.mean(errors):.1f}')
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Count')
    axes[0].set_title(title)
    axes[0].legend()

    sorted_errors = np.sort(errors)
    axes[1].plot(sorted_errors)
    axes[1].set_xlabel('Sample (sorted)')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Sorted Errors')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
