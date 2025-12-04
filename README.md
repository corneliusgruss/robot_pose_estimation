# Robot Pose Estimation

Two-stage CNN pipeline for UR10 robot arm pose estimation from single RGB images.

**Authors**: Cornelius Gruss, Devin Caulfield, Juan Rueda  
**Course**: CS523 Deep Learning, Boston University  
**Date**: December 2025

## Overview

This project estimates the 6D pose of a UR10 robot arm from a single RGB image using a two-stage deep learning approach:

1. **Stage 1**: Bounding box detection (ResNet18)
2. **Stage 2**: 2D keypoint regression (ResNet34)

The 2D keypoints can be evaluated using a scaling-based 3D error approximation (see `utils/pose_3d.py` for details on limitations).

## Results

| Metric | Value |
|--------|-------|
| Mean ADD Error | 9.5 cm |
| Median ADD Error | 7.6 cm |
| AUC (0-30cm) | 0.701 |
| 2D Pixel Error | ~17 px |

## Project Structure

```
robot_pose_estimation/
├── config.py                 # All hyperparameters
├── models/
│   ├── bbox_model.py         # Stage 1: BBoxModel
│   ├── keypoint_model.py     # Stage 2: KeypointModel
│   └── pipeline.py           # TwoStagePipeline
├── data/
│   └── dataset.py            # RobotKeypointDataset
├── utils/
│   ├── visualization.py      # Plotting functions
│   ├── metrics.py            # 2D metrics, IoU
│   ├── pose_3d.py            # 3D error estimation (approximate)
│   └── training.py           # Training loops
├── checkpoints/              # Saved models
├── notebooks/
│   ├── 01_train_stage1.ipynb # Train bbox model
│   ├── 02_train_stage2.ipynb # Train keypoint model
│   └── 03_evaluate.ipynb     # Full evaluation
└── README.md
```

## Usage

### Training

1. Update paths in `config.py` to point to your dataset
2. Train Stage 1 (bounding box detection):
   - Open `notebooks/01_train_stage1.ipynb`
3. Train Stage 2 (keypoint regression, uses ground-truth bboxes):
   - Open `notebooks/02_train_stage2.ipynb`

Note: The two stages are independent during training. Stage 2 uses ground-truth bounding boxes, not Stage 1 predictions.

### Evaluation

Open `notebooks/03_evaluate.ipynb` to run the full pipeline evaluation.

### Inference

```python
from models import load_pipeline
import config

pipeline = load_pipeline(
    'checkpoints/stage1_best.pt',
    'checkpoints/stage2_best.pt',
    config,
    device='cuda'
)

# Run on image
keypoints_2d, bbox = pipeline.predict('path/to/image.png')

# Or get structured output
result = pipeline.predict_dict('path/to/image.png')
```

## Dataset

- **Training**: 8,000 synthetic images from Isaac Sim
  - 4 domain-randomized environments
  - Random textures, lighting, camera positions
- **Test**: 2,000 images from unseen environment
- **Robot**: Universal Robots UR10
- **Image size**: 1080×1080 RGB

## Architecture

### Stage 1: Bounding Box Detection
- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Input**: 256×256 resized image
- **Output**: 4 values [x_min, y_min, x_max, y_max] normalized to [0,1]
- **Metric**: IoU ~0.79

### Stage 2: Keypoint Regression
- **Backbone**: ResNet34 (pretrained on ImageNet)
- **Input**: 512×512 cropped image
- **Output**: 12 values (x,y for 6 joints) normalized to [0,1]
- **Metric**: ~17px error (with GT bbox)

## Key Design Decisions

1. **Two-stage approach**: Localizing the robot first allows Stage 2 to focus on a smaller region with higher resolution
2. **2D keypoints over direct angles**: CNNs excel at spatial localization; angle regression failed (68-79° error)
3. **Domain randomization**: Synthetic training data with varied textures/lighting enables generalization
4. **Increased bbox padding**: 75px padding (up from 50px) gives Stage 2 more context

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- pandas
- matplotlib
- PIL
- tqdm

## References

- Lee et al., "Camera-to-Robot Pose Estimation from a Single Image" (DREAM), ICRA 2020
- Tobin et al., "Domain Randomization for Transferring Deep Neural Networks", IROS 2017
