"""
Two-Stage Pipeline for Robot Pose Estimation

Chains Stage 1 (bbox) and Stage 2 (keypoints) for end-to-end inference.
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from .bbox_model import BBoxModel, load_bbox_model
from .keypoint_model import KeypointModel, load_keypoint_model


class TwoStagePipeline:
    """
    Complete inference pipeline: Raw image -> 2D keypoints
    
    Stage 1: Full image -> Bounding box
    Stage 2: Cropped image -> 2D keypoints
    """
    
    def __init__(self, model_stage1, model_stage2, config, device='cuda'):
        """
        Args:
            model_stage1: trained BBoxModel
            model_stage2: trained KeypointModel
            config: config module or dict with stage1_size, stage2_size, etc.
            device: torch device
        """
        self.model_stage1 = model_stage1.eval()
        self.model_stage2 = model_stage2.eval()
        self.device = device
        
        # Handle both module and dict config
        if hasattr(config, 'STAGE1_SIZE'):
            self.stage1_size = config.STAGE1_SIZE
            self.stage2_size = config.STAGE2_SIZE
            self.joint_names = config.JOINT_NAMES
        else:
            self.stage1_size = config.get('stage1_size', 256)
            self.stage2_size = config.get('stage2_size', 512)
            self.joint_names = config.get('joint_names', 
                ['Base', 'Shoulder', 'Elbow', 'Wrist3', 'Wrist4', 'Wrist5'])
        
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
    
    @torch.no_grad()
    def predict(self, image):
        """
        Run full pipeline on an image.
        
        Args:
            image: PIL Image, numpy array, or path string
        
        Returns:
            keypoints_2d: (6, 2) array in original image pixel coordinates
            bbox: (4,) array [x_min, y_min, x_max, y_max] in original coords
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        orig_w, orig_h = image.size
        
        # Stage 1: Predict bbox
        img_s1 = self.transform_stage1(image).unsqueeze(0).to(self.device)
        bbox_norm = self.model_stage1(img_s1).cpu().numpy()[0]
        
        # Convert to original image coordinates
        bbox = bbox_norm * np.array([orig_w, orig_h, orig_w, orig_h])
        x_min, y_min, x_max, y_max = bbox.astype(int)
        
        # Clamp to image bounds
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(orig_w, x_max), min(orig_h, y_max)
        
        # Stage 2: Crop and predict keypoints
        crop = image.crop((x_min, y_min, x_max, y_max))
        crop_resized = crop.resize((self.stage2_size, self.stage2_size))
        img_s2 = self.transform_stage2(crop_resized).unsqueeze(0).to(self.device)
        
        kp_norm = self.model_stage2(img_s2).cpu().numpy()[0].reshape(6, 2)
        
        # Convert keypoints back to original image coordinates
        crop_w, crop_h = x_max - x_min, y_max - y_min
        keypoints_2d = kp_norm * np.array([crop_w, crop_h]) + np.array([x_min, y_min])
        
        return keypoints_2d, bbox
    
    @torch.no_grad()
    def predict_with_gt_bbox(self, image, gt_bbox):
        """
        Use ground-truth bbox (for evaluating Stage 2 in isolation).
        
        Args:
            image: PIL Image or path
            gt_bbox: (4,) array [x_min, y_min, x_max, y_max] in pixels
        
        Returns:
            keypoints_2d: (6, 2) array in original image coordinates
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        x_min, y_min, x_max, y_max = gt_bbox.astype(int)
        
        crop = image.crop((x_min, y_min, x_max, y_max))
        crop_resized = crop.resize((self.stage2_size, self.stage2_size))
        img_s2 = self.transform_stage2(crop_resized).unsqueeze(0).to(self.device)
        
        kp_norm = self.model_stage2(img_s2).cpu().numpy()[0].reshape(6, 2)
        
        crop_w, crop_h = x_max - x_min, y_max - y_min
        keypoints_2d = kp_norm * np.array([crop_w, crop_h]) + np.array([x_min, y_min])
        
        return keypoints_2d
    
    def predict_dict(self, image):
        """
        Predict and return as a structured dictionary.
        
        Returns:
            dict with 'keypoints' and 'bbox' keys
        """
        keypoints_2d, bbox = self.predict(image)
        
        return {
            'bbox': {
                'x_min': float(bbox[0]),
                'y_min': float(bbox[1]),
                'x_max': float(bbox[2]),
                'y_max': float(bbox[3]),
            },
            'keypoints': {
                name: {'x': float(keypoints_2d[i, 0]), 'y': float(keypoints_2d[i, 1])}
                for i, name in enumerate(self.joint_names)
            }
        }


def load_pipeline(stage1_path, stage2_path, config, device='cuda'):
    """
    Convenience function to load complete pipeline from checkpoints.
    
    Args:
        stage1_path: path to Stage 1 checkpoint
        stage2_path: path to Stage 2 checkpoint
        config: config module or dict
        device: torch device
    
    Returns:
        pipeline: ready-to-use TwoStagePipeline
    """
    model_stage1, _ = load_bbox_model(stage1_path, device)
    model_stage2, _ = load_keypoint_model(stage2_path, device)
    
    pipeline = TwoStagePipeline(model_stage1, model_stage2, config, device)
    
    return pipeline


if __name__ == '__main__':
    print("Pipeline module - use load_pipeline() to create inference pipeline")
