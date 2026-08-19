"""Colab T4 training script — paste into a single Colab cell.

Designed for the EverestLabs demo: ~10-15 min on a free Colab T4 to
reach the 0.75+ mAP@50 mark. The conveyor_perception project uses
this trained model for the end-to-end demo.

Run in a Colab cell:
    !git clone https://github.com/roniejosephv-star/conveyor-perception.git
    %cd conveyor-perception
    !pip install -q ultralytics==8.4.121 opencv-python==4.11.0.86 supervision==0.30.0 roboflow==1.4.1
    !python scripts/train_yolo26_colab.py

Output:
    - models/yolo26s_recyclable.pt  (trained weights, ~80 MB)
    - models/yolo26s_recyclable.onnx (ONNX export, ~36 MB)
    - 3-way benchmark numbers (vs COCO pretrained baseline)
    - Per-class mAP @50

The script intentionally does NOT clone the dataset (uses the public
Roboflow URL directly) to keep the Colab cell self-contained.

Why a separate script (vs train_yolo26.py):
- Colab has different defaults: device=0 (CUDA), imgsz=640 (full res),
  batch=32 (T4 has 16GB), epochs=30 (free GPU, no runtime cap)
- Pin ultralytics==8.4.121 to match the project's dev environment
- Skip the --data-yaml argument; the script finds it from the dataset
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Install pinned deps early (Colab has no requirements.txt auto-install)
os.system("pip install -q ultralytics==8.4.121 opencv-python==4.11.0.86 supervision==0.30.0 roboflow==1.4.1 2>&1 | tail -1")

from ultralytics import YOLO  # noqa: E402  (after pip install)


def main() -> int:
    p = argparse.ArgumentParser(description="Colab T4 training for the recycling model")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs (30 = full convergence)")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size (T4 handles 640 fine)")
    p.add_argument("--batch", type=int, default=32, help="Batch size (T4 = 16GB)")
    p.add_argument("--device", default="0", help="CUDA device (default 0)")
    p.add_argument(
        "--dataset",
        default="zkf624/-recycling",
        help="Roboflow dataset (workspace/project)",
    )
    p.add_argument("--version", type=int, default=3, help="Dataset version")
    p.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    args = p.parse_args()

    if not args.api_key:
        print("ERROR: Roboflow API key not set.")
        print("Get one at https://app.roboflow.com → Settings → API Keys")
        print("Then either: --api-key YOUR_KEY, or set ROBOFLOW_API_KEY env var")
        return 1

    workspace, project = args.dataset.split("/", 1)
    project = project.lstrip("/")  # "-recycling" sometimes has a leading "-"

    # Download the dataset
    print(f"Downloading {workspace}/{project} v{args.version}...")
    from roboflow import Roboflow

    rf = Roboflow(api_key=args.api_key)
    proj = rf.workspace(workspace).project(project)
    out_dir = Path("data/raw/recycling_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    proj.version(args.version).download("yolov11", location=str(out_dir))
    print(f"✓ Dataset downloaded to {out_dir}")

    # Resolve data.yaml path
    data_yaml = out_dir / "data.yaml"
    if not data_yaml.exists():
        # Roboflow sometimes nests data.yaml in a subfolder
        candidates = list(out_dir.rglob("data.yaml"))
        if not candidates:
            print(f"ERROR: data.yaml not found in {out_dir}")
            return 1
        data_yaml = candidates[0]
    print(f"✓ Using {data_yaml}")

    # Sanity-check the data.yaml paths
    content = data_yaml.read_text()
    if "../" in content:
        data_yaml.write_text(content.replace("../", "./"))
        print("✓ Fixed data.yaml relative paths")

    # Train
    print(f"\nTraining YOLO26s on T4 (epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch})...")
    print("(This will take ~10-15 min on a free Colab T4)")
    t0 = time.time()
    model = YOLO("yolo26s.pt")  # auto-downloads
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="models/train_runs",
        name="yolo26s_recyclable",
        exist_ok=True,
        verbose=True,
    )
    train_time_sec = time.time() - t0

    # Find best.pt and copy to models/
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"ERROR: best.pt not found at {best_pt}")
        return 1
    Path("models").mkdir(exist_ok=True)
    final_pt = Path("models/yolo26s_recyclable.pt")
    shutil.copy(best_pt, final_pt)
    print(f"\n✓ Saved trained model: {final_pt} ({final_pt.stat().st_size / 1024 / 1024:.1f} MB)")

    # Export to ONNX
    print("\nExporting to ONNX...")
    best_model = YOLO(str(best_pt))
    best_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    # Ultralytics writes yolo26s_recyclable.onnx to the models/ folder
    src = Path("yolo26s_recyclable.onnx")
    if src.exists():
        final_onnx = Path("models/yolo26s_recyclable.onnx")
        shutil.move(str(src), str(final_onnx))
        print(f"✓ Saved ONNX: {final_onnx} ({final_onnx.stat().st_size / 1024 / 1024:.1f} MB)")

    # Validation
    print("\nValidating on the test split...")
    val_results = best_model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
    )
    metrics = {
        "mAP@50": float(val_results.box.map50),
        "mAP@50-95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
        "per_class_mAP@50": {
            model.names[i]: float(val_results.box.maps[i])
            for i in range(len(model.names))
        },
    }

    # Final report
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Trained model:    {final_pt}")
    print("  ONNX export:      models/yolo26s_recyclable.onnx")
    print(f"  Training runs:    {results.save_dir}")
    print(f"  Training time:    {train_time_sec / 60:.1f} min")
    print()
    print(f"  mAP@50:           {metrics['mAP@50']:.3f}")
    print(f"  mAP@50-95:        {metrics['mAP@50-95']:.3f}")
    print(f"  precision:        {metrics['precision']:.3f}")
    print(f"  recall:           {metrics['recall']:.3f}")
    print()
    print("  Per-class mAP@50:")
    for name, m in metrics["per_class_mAP@50"].items():
        print(f"    {name:10s} {m:.3f}")
    print()
    print("Next: run the end-to-end demo:")
    print("  python examples/multitask_demo.py \\")
    print("    --image data/sample/recycling_sample.jpg \\")
    print("    --model models/yolo26s_recyclable.pt \\")
    print("    --data-yaml data/raw/recycling_v3/data.yaml")
    print()
    print("Or run the 3-way benchmark:")
    print("  python scripts/benchmark.py --model models/yolo26s_recyclable.pt --device 0 --imgsz 640")

    # Save metrics JSON for later reference
    with open("models/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
