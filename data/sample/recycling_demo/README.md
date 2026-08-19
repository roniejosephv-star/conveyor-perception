# Recycling Demo Dataset (bundled, offline)

This is a **demo subset** of the Roboflow recycling dataset
(workspace `zkf624`, project `-recycling`, version 3, CC BY 4.0).

## Stats
- 104 images (83 train / 21 val split of v3's test split)
- 4 classes: Glass, metal, plastic, vinyl
- YOLO segmentation format (polygon labels)
- 4.2M total — bundled in the repo for offline use

## Why bundled
Roboflow's S3 export has been intermittently broken as of Aug 2026
(NoSuchKey XML response on most projects). Bundling a small subset
guarantees the Colab demo can train on REAL recycling data without
depending on Roboflow's flaky export.

## Source
- Original: https://universe.roboflow.com/zkf624/-recycling/dataset/3
- License: CC BY 4.0 (commercial OK with attribution)
- v3 has 2404 images total; this is a 104-image subset for the demo.

## Layout
```
data/sample/recycling_demo/
├── data.yaml          # YOLO config (4 classes)
├── README.md
├── train/
│   ├── images/        # 83 JPG files
│   └── labels/        # 83 TXT files (YOLO seg polygons)
└── val/
    ├── images/        # 21 JPG files
    └── labels/        # 21 TXT files
```

## Usage in the Colab demo
```python
from pathlib import Path
DATA_YAML = Path('/content/conveyor-perception/data/sample/recycling_demo/data.yaml')
# Train: model.train(data=str(DATA_YAML), epochs=5, imgsz=320, batch=8)
# ~30s on T4, ~2min on CPU. Good enough to show the training flow + metrics.
```

## For full training
Use `scripts/download_dataset.py` when Roboflow S3 is up — fetches
the full 2404-image dataset. The bundled subset is the offline
fallback that GUARANTEES the demo works.
