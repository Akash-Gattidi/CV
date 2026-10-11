# Monocular Video to 3D Gaussian Splatting (3DGS)

A lightweight pipeline for transforming raw monocular walk-through videos of indoor spaces into 3D Gaussian Splatting scenes using DUSt3R geometry initialization and gsplat optimization.

Designed for consumer and low-VRAM GPUs with bounded autograd memory footprints.

---

## Getting Started

### 1. Bring Your Own Video
No sample videos, extracted frames, or pre-trained 3D scene files are bundled with this repository. Users must supply their own video capture.

* Place your video file inside an `input/` folder, for example `input/video.mp4`.
* For best results, capture continuous, smoothly moving walk-through video footage with minimal motion blur.

### 2. Installation
Ensure PyTorch with CUDA support is installed. Then install the project requirements:

```bash
pip install -r requirements.txt
```

Download the DUSt3R ViT backbone model weights:
```bash
python scripts/download_dust3r.py
```

### 3. Running Single-Room Reconstruction (Milestone 1)
To extract keyframes, estimate camera poses, and train a 3D Gaussian Splatting model:

```bash
python -m pipeline.m1_pipeline --video input/video.mp4 --iters 5000
```

### 4. Running Multi-Room Reconstruction (Milestone 2)
To reconstruct multi-room sequences through doorways and corridors:

```bash
# 1. Extract clean, blur-filtered keyframes from your multi-room video
python scripts/extract_double_room_frames.py --video input/video.mp4 --out_dir output/frames_double

# 2. Run joint geometry estimation, stratified seeding, and splat training
python -m pipeline.m2_pipeline
```

---

## Hardware & Architecture Requirements
* **GPU**: CUDA-compatible GPU (optimized for consumer low-VRAM GPUs).
* **Geometry Backbone**: DUSt3R ViT-Large with float32 weights and native mixed-precision autocast.
* **Splatting Engine**: Pure PyTorch differentiable rasterization with adaptive densification and opacity reset schedules.
