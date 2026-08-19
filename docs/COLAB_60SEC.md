# Colab T4 — 60-Second Setup

> **Goal**: go from "I just opened Colab" to "trained recycling model
> at mAP@50=0.75+" in 6 copy-paste cells.
> **Total wall time**: ~12-15 min on free Colab T4 (mostly the
> training cell).

## TL;DR — just paste this into a single cell

If you want the absolute fastest path (one cell, all of it):

```python
# === Cell 1: All-in-one (≈12 min) ===
import os
from pathlib import Path

# 1. Install pinned deps
os.system("pip install -q ultralytics==8.4.121 opencv-python==4.11.0.86 supervision==0.30.0 roboflow==1.4.1 fastmcp==3.4.7 pydantic==2.13.4 numpy==1.26.4 python-dotenv 2>&1 | tail -1")

# 2. Clone (or pull) the repo
if not Path("conveyor-perception").exists():
    !git clone https://github.com/roniejosephv-star/conveyor-perception.git
%cd conveyor-perception
else:
    !git -C conveyor-perception pull --rebase 2>&1 | tail -2

# 3. Use the public demo Roboflow key (read-only, scrapes the dataset)
os.environ.setdefault("ROBOFLOW_API_KEY", "qogO5hAuLgUUYMbNT6W3")

# 4. Train (30 epochs, imgsz=640, batch=32 on T4)
!python scripts/train_yolo26_colab.py --epochs 30 --imgsz 640 --batch 32 --device 0
```

When the cell finishes you'll have:
- `models/yolo26s_recyclable.pt` (≈80 MB trained weights)
- `models/yolo26s_recyclable.onnx` (≈36 MB ONNX export)
- Per-class mAP@50 numbers printed at the end

Then run the end-to-end demo in a second cell:

```python
# === Cell 2: End-to-end demo (≈30s) ===
!python examples/multitask_demo.py \
    --image data/sample/recycling_sample.jpg \
    --model models/yolo26s_recyclable.pt \
    --data-yaml data/raw/recycling_v3/data.yaml
```

If you don't have a recycling sample image, the demo will use the
COCO `bus.jpg` (just won't show real recycling classes).

## The 6-cell version (what's actually in `notebooks/demo.ipynb`)

If you want the full walkthrough with explanations between cells, open
[`notebooks/demo.ipynb`](../notebooks/demo.ipynb) in Colab. The cells
do exactly what the 1-cell version does, but split for readability:

1. `pip install` (60s)
2. `git clone` (5s)
3. Download recycling dataset via Roboflow (30s)
4. Train YOLO26s — 30 epochs on T4 (~10-15 min)
5. Run the multi-task pipeline on a sample image (5s)
6. Inspect the L1 triage queue + drift signals (1s)
7. Robustness suite against MRF conditions (10s)
8. Shift dashboard snapshot (1s)
9. (Markdown) elevator-pitch summary

**Cell 4 is the long one. Everything else is <60 seconds.**

## Fast path: skip training, use pretrained (~90 seconds total)

If you just want to see inference + triage + dashboard (no fine-tuning):

```python
# === Fast inference cell (≈90s) ===
!pip install -q ultralytics==8.4.121 opencv-python==4.11.0.86 supervision==0.30.0 fastmcp==3.4.7 pydantic==2.13.4 numpy==1.26.4 2>&1 | tail -1
!git clone https://github.com/roniejosephv-star/conveyor-perception.git
%cd conveyor-perception
# Download just the COCO pretrained YOLO26s (auto-cached by Ultralytics)
from ultralytics import YOLO
model = YOLO("yolo26s.pt")  # auto-downloads
# Run on a sample image
!python examples/multitask_demo.py --image data/sample/bus.jpg --model yolo26s.pt
```

This won't show recycling classes (it's COCO pretrained), but it does
demonstrate the full pipeline plumbing.

## Why these versions are pinned

The pinned versions in `requirements.txt` are the ones that:
- Run on Python 3.12 (no 3.13/3.14 yet — many packages lack wheels)
- Have numpy 1.26.x compatibility (ultralytics 8.4.x is flaky on numpy 2)
- opencv-python 4.11.x works on numpy 1.26; 4.12+ requires numpy 2

If you bump anything, also bump `requirements.txt` and
`docs/UPGRADE_PATHS.md`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No module named 'ultralytics'" | Run the pip install cell again (sometimes the first run takes >60s in Colab) |
| Runtime disconnected after 90s | Free Colab T4 has a 90s idle timeout. Don't idle — keep clicking through cells. |
| "data.yaml not found" | The Roboflow download sometimes nests data.yaml. The script handles this automatically. |
| "CUDA out of memory" | Lower `--batch 16` instead of 32. T4 has 16GB; batch 32 at imgsz 640 is right on the edge. |
| Want FP16 for 2x speedup | Add `--half` flag to the train command. T4 supports it. |
| Want TensorRT | After training: `python scripts/export_tensorrt.py --model models/yolo26s_recyclable.pt` (also runs on T4) |

## Next after Colab training

Once the model is trained, the artifacts you have on disk:
- `models/yolo26s_recyclable.pt` — for Ultralytics inference
- `models/yolo26s_recyclable.onnx` — for OpenCV DNN or ONNX Runtime
- `models/train_metrics.json` — the final metrics

**Local** (Mac M4, no GPU):
```bash
python scripts/benchmark.py --model models/yolo26s_recyclable.pt --device mps --imgsz 640
```

**Jetson Orin Nano** (production):
```bash
python scripts/export_tensorrt.py --model models/yolo26s_recyclable.pt --imgsz 640
# Produces models/yolo26s_recyclable.engine (FP16, ~25 MB)
```

**ROC production** (Docker Compose, full stack):
```bash
cd docker && docker compose up
# 4 services: perception + camera_sim + dashboard + redis
```
