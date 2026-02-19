# Robot Pose Estimation

**Two-stage CNN pipeline for UR10 pose estimation from RGB images.**

Estimates 6-DOF joint angles of a UR10 robotic arm from a single RGB image using a detect-then-regress approach: Stage 1 localizes the robot with a bounding box, Stage 2 regresses 2D keypoints on the cropped region, then inverse kinematics recovers joint angles.

## Architecture

```
                         ┌──────────────────┐
  1080×1080 RGB ────────▶│  Stage 1         │
                         │  ResNet18        │──▶ Bounding Box (4D)
                         │  256×256 input   │         │
                         └──────────────────┘         │
                                                      ▼ crop
                         ┌──────────────────┐
  Cropped region ───────▶│  Stage 2         │
                         │  ResNet34        │──▶ 2D Keypoints (6×2)
                         │  512×512 input   │         │
                         └──────────────────┘         │
                                                      ▼
                         ┌──────────────────┐
                         │  Inverse         │
                         │  Kinematics      │──▶ Joint Angles (6D)
                         └──────────────────┘
```

## Results

| Metric | Value |
|--------|-------|
| Mean ADD (3D accuracy) | 6.23 cm |
| AUC @ 30 cm | 80.2% |
| Bounding Box IoU | 86.4% |
| Mean 2D Keypoint Error | 14.4 px |
| End-to-end Inference | 56 ms (18 FPS) |

### Comparison to Baselines

| Method | AUC @ 30cm | Inference (ms) |
|--------|-----------|----------------|
| DREAM-H | 79.2% | 66 |
| RoboPose | 84.7% | 571 |
| **Ours** | **80.2%** | **56** |

## Training

- **Data**: 8,000 synthetic RGB images from NVIDIA Isaac Sim
- **Domain randomization**: 10 arm textures, 100 floor textures, 4 environments (room, warehouse, hospital, clean), variable lighting, camera position, Gaussian noise + cutout augmentation
- **Stage 1**: ResNet18, 30 epochs, AdamW (lr=1e-4), MSE loss
- **Stage 2**: ResNet34, 50 epochs, AdamW (lr=1e-4), MSE loss
- **Hardware**: NVIDIA Tesla V100

## Project Structure

```
robot_pose_estimation/
├── models/
│   ├── bbox_model.py          # Stage 1: ResNet18 bounding box detector
│   ├── keypoint_model.py      # Stage 2: ResNet34 keypoint regressor
│   └── pipeline.py            # End-to-end two-stage pipeline
├── data/
│   └── dataset.py             # Dataset loading with domain randomization
├── utils/
│   ├── training.py            # Training loops for both stages
│   ├── metrics.py             # IoU, pixel error, ADD, AUC metrics
│   ├── kinematics.py          # UR10 forward kinematics + IK solver
│   ├── pose_3d.py             # 3D pose evaluation (ADD, PnP)
│   └── visualization.py       # Prediction visualization
├── notebooks/
│   ├── 01_train_stage1.ipynb  # BBox detection training
│   ├── 02_train_stage2.ipynb  # Keypoint regression training
│   └── 03_evaluate.ipynb      # Full evaluation pipeline
├── config.py                  # Hyperparameters and camera intrinsics
├── final_report.tex           # Full technical report
└── checkpoints/               # Trained model weights
```

## Usage

### Training

Run the Jupyter notebooks in order:
```
notebooks/01_train_stage1.ipynb  → trains bounding box detector
notebooks/02_train_stage2.ipynb  → trains keypoint regressor
notebooks/03_evaluate.ipynb      → full pipeline evaluation
```

### Inference

```python
from models.pipeline import load_pipeline

pipeline = load_pipeline(
    'checkpoints/stage1_best.pt',
    'checkpoints/stage2_best.pt',
    config, device='cuda'
)
keypoints, bbox = pipeline.predict(image_path)
```

### Dataset

Synthetic data generated in NVIDIA Isaac Sim: [HuggingFace Dataset](https://huggingface.co/datasets/Delta-Gear/UR10_Random_Pose)

## Academic Context

**CS523 Deep Learning** — Boston University

**Team**: Cornelius Gruss, Devin Caulfield, Juan Rueda
