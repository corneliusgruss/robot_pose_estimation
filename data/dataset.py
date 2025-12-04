"""
Dataset for Robot Pose Estimation

Loads images and labels for two-stage training:
- Stage 1: Full images with bounding box labels
- Stage 2: Cropped images with keypoint labels
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path


class RobotKeypointDataset(Dataset):
    """
    Dataset for two-stage robot keypoint detection.
    
    Each sample provides:
        - img_stage1: resized full image for bbox prediction
        - img_stage2: cropped image for keypoint prediction
        - bbox: normalized bounding box [x_min, y_min, x_max, y_max]
        - keypoints: normalized keypoints relative to crop
        - keypoints_orig: original pixel coordinates
        - positions_3d: 3D joint positions (if available)
    """
    
    def __init__(self, data_dirs, config, load_3d=False):
        """
        Args:
            data_dirs: list of directories containing images and labels.csv
            config: config module or dict with image sizes, etc.
            load_3d: whether to load 3D positions
        """
        # Handle both module and dict config
        if hasattr(config, 'ORIG_SIZE'):
            self.orig_size = config.ORIG_SIZE
            self.stage1_size = config.STAGE1_SIZE
            self.stage2_size = config.STAGE2_SIZE
            self.bbox_padding = config.BBOX_PADDING
            self.keypoint_cols_2d = config.KEYPOINT_COLS_2D
            self.keypoint_cols_3d = config.KEYPOINT_COLS_3D if load_3d else None
        else:
            self.orig_size = config.get('orig_size', 1080)
            self.stage1_size = config.get('stage1_size', 256)
            self.stage2_size = config.get('stage2_size', 512)
            self.bbox_padding = config.get('bbox_padding', 75)
            self.keypoint_cols_2d = config.get('keypoint_cols_2d', [
                ('2D_J0_Base_x', '2D_J0_Base_y'),
                ('2D_J1_Shoulder_x', '2D_J1_Shoulder_y'),
                ('2D_J2_Elbow_x', '2D_J2_Elbow_y'),
                ('2D_J3_Wrist_x', '2D_J3_Wrist_y'),
                ('2D_J4_Wrist_x', '2D_J4_Wrist_y'),
                ('2D_J5_Wrist_x', '2D_J5_Wrist_y'),
            ])
            self.keypoint_cols_3d = config.get('keypoint_cols_3d') if load_3d else None
        
        self.load_3d = load_3d
        
        # Load all samples
        self.samples = []
        self._load_samples(data_dirs)
        
        # Transforms
        self.transform_stage1 = T.Compose([
            T.Resize((self.stage1_size, self.stage1_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_stage2 = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_samples(self, data_dirs):
        """Load samples from all directories."""
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            csv_path = data_path / 'labels.csv'
            
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping")
                continue
            
            df = pd.read_csv(csv_path)
            
            for idx, row in df.iterrows():
                # Parse "Sample N" -> N
                command = row['command']
                sample_num = int(command.replace('Sample ', ''))
                
                # Find image file
                img_pattern = f"capture{sample_num}_*.png"
                img_files = list(data_path.glob(img_pattern))
                
                if len(img_files) == 0:
                    continue
                
                img_path = img_files[0]
                
                # Extract 2D keypoints
                keypoints_2d = []
                for x_col, y_col in self.keypoint_cols_2d:
                    keypoints_2d.append([row[x_col], row[y_col]])
                keypoints_2d = np.array(keypoints_2d, dtype=np.float32)
                
                sample = {
                    'img_path': str(img_path),
                    'keypoints': keypoints_2d,
                }
                
                # Extract 3D positions if requested
                if self.load_3d and self.keypoint_cols_3d:
                    positions_3d = []
                    for x_col, y_col, z_col in self.keypoint_cols_3d:
                        positions_3d.append([row[x_col], row[y_col], row[z_col]])
                    sample['positions_3d'] = np.array(positions_3d, dtype=np.float32)
                
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {len(data_dirs)} directories")
    
    def compute_bbox(self, keypoints):
        """Compute bounding box from keypoints with padding."""
        x_coords = keypoints[:, 0]
        y_coords = keypoints[:, 1]
        
        x_min = max(0, np.min(x_coords) - self.bbox_padding)
        y_min = max(0, np.min(y_coords) - self.bbox_padding)
        x_max = min(self.orig_size, np.max(x_coords) + self.bbox_padding)
        y_max = min(self.orig_size, np.max(y_coords) + self.bbox_padding)
        
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['img_path']).convert('RGB')
        keypoints = sample['keypoints'].copy()
        
        # Compute ground-truth bbox
        bbox = self.compute_bbox(keypoints)
        
        # Stage 1: Full image resized
        img_stage1 = self.transform_stage1(img)
        
        # Stage 2: Crop to bbox, then resize
        x_min, y_min, x_max, y_max = bbox.astype(int)
        crop = img.crop((x_min, y_min, x_max, y_max))
        crop_resized = crop.resize((self.stage2_size, self.stage2_size), Image.BILINEAR)
        img_stage2 = self.transform_stage2(crop_resized)
        
        # Convert keypoints to crop-relative normalized coordinates
        crop_w = x_max - x_min
        crop_h = y_max - y_min
        keypoints_crop = keypoints.copy()
        keypoints_crop[:, 0] = (keypoints[:, 0] - x_min) / crop_w
        keypoints_crop[:, 1] = (keypoints[:, 1] - y_min) / crop_h
        keypoints_crop = keypoints_crop.flatten()  # Shape: (12,)
        
        # Normalize bbox to [0, 1]
        bbox_norm = bbox / self.orig_size
        
        result = {
            'img_stage1': img_stage1,
            'img_stage2': img_stage2,
            'bbox': torch.tensor(bbox_norm, dtype=torch.float32),
            'keypoints': torch.tensor(keypoints_crop, dtype=torch.float32),
            'keypoints_orig': torch.tensor(keypoints.flatten(), dtype=torch.float32),
            'img_path': sample['img_path'],
        }
        
        if self.load_3d and 'positions_3d' in sample:
            result['positions_3d'] = torch.tensor(sample['positions_3d'], dtype=torch.float32)
        
        return result


def create_dataloaders(train_dirs, test_dir, config, batch_size=None, num_workers=None):
    """
    Convenience function to create train and test dataloaders.
    
    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    from torch.utils.data import DataLoader
    
    # Handle config
    if hasattr(config, 'BATCH_SIZE'):
        batch_size = batch_size or config.BATCH_SIZE
        num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
    else:
        batch_size = batch_size or config.get('batch_size', 16)
        num_workers = num_workers if num_workers is not None else config.get('num_workers', 2)
    
    train_dataset = RobotKeypointDataset(train_dirs, config)
    test_dataset = RobotKeypointDataset([test_dir], config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


if __name__ == '__main__':
    # Quick test with default config
    import sys
    sys.path.append('..')
    import config
    
    dataset = RobotKeypointDataset(config.TRAIN_DIRS[:1], config)
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"img_stage1: {sample['img_stage1'].shape}")
    print(f"img_stage2: {sample['img_stage2'].shape}")
    print(f"bbox: {sample['bbox']}")
    print(f"keypoints: {sample['keypoints'].shape}")
