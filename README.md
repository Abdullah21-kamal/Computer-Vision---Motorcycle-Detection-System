# Motorcycle Detection System

A lightweight, production-leaning pipeline for detecting motorcycles in hierarchically organized image datasets using **Faster R-CNN (ResNet-50 FPN)** via PyTorch/TorchVision. The tool preserves the `recording → person → image` directory structure, saves annotated images, and emits a machine-readable CSV for downstream analysis.

## Features
- Automated motorcycle detection across large datasets (GPU-accelerated when available)
- Strict preservation of input hierarchy in outputs
- Annotated images with bounding boxes and confidence scores
- Per-image CSV report with motorcycle counts
- Simple CLI with sensible defaults

## Requirements
- Python 3.8+
- Windows, Linux, or macOS (CPU mode on macOS)
- Recommended: NVIDIA GPU with CUDA for acceleration

**Python packages**:
```bash
pip install torch torchvision pillow tqdm
# (Optional, GPU wheels) https://pytorch.org/get-started/locally/
```

## Installation
Clone the repository and install dependencies:
```bash
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_DIR>
pip install -r requirements.txt  # optional if you create one
```

## Directory Structure
```
input_root/
├── recording_001/
│   ├── person_A/
│   │   ├── image_001.jpg
│   │   └── image_002.png
│   └── person_B/
│       └── image_003.jpg
└── recording_002/
    └── person_C/
        └── photo_001.jpg
```

## Usage
Run the detector on a folder tree:
```bash
python detect_motorcycles_nested.py   --input_dir /path/to/input_root   --output_dir /path/to/output_root
```

### Key Arguments
- `--input_dir` (required): Root folder containing `recording/person/image` hierarchy
- `--output_dir` (required): Destination for CSV and annotated images
- `--conf` (optional, default: 0.5): Confidence threshold (0.0–1.0)
- `--max_size` (optional, default: 1536): Max image dimension for inference (0 = no resize)
- `--csv_name` (optional, default: motorcycle_detections.csv): Custom CSV name

### Examples
High-precision pass:
```bash
python detect_motorcycles_nested.py   --input_dir "D:\data\recordings\root"   --output_dir "D:\data\results"   --conf 0.75
```

Faster pass at lower resolution:
```bash
python detect_motorcycles_nested.py   --input_dir ./data/root   --output_dir ./results   --max_size 1024   --conf 0.6
```

## Outputs
```
output_root/
├── motorcycle_detections.csv
├── recording_001/
│   └── person_A/
│       ├── image_001.jpg   # annotated
│       └── image_002.png   # annotated
└── recording_002/
    └── person_C/
        └── photo_001.jpg   # annotated
```

**CSV schema** (`motorcycle_detections.csv`):
```
recording_id,person_id,image_id,num_motorcycles
```

## Notes
- The model uses COCO-pretrained weights and filters for the *motorcycle* class only.
- GPU will be used automatically if available; otherwise the script runs on CPU.
- Ensure the three-level folder hierarchy is respected; non-conforming paths are skipped.


## Citation
If you use this code in academic or public work, please cite the repository URL and version.
