"""Train YOLO26s on the Roboflow recycling dataset.

Pre-requisites:
- Run `python scripts/download_dataset.py` first (saves to data/raw/)
- Edit WORKSPACE and PROJECT below if your dataset name differs

Then run:
    # On Colab T4 (recommended for full 50 epochs in 15-20 min)
    python scripts/train_yolo26.py

    # Local Mac with MPS (slower, ~30-40 min for 50 epochs)
    python scripts/train_yolo26.py --device mps

The trained model is exported to models/yolo26s_recyclable.onnx.
The training log is saved to models/train_log.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def main() -> int:
    p = argparse.ArgumentParser(description="Train YOLO26s on a recycling dataset")
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument(
        "--device", default="cpu", help="Device: cpu, mps (Mac), 0 (CUDA), 0,1 (multi-GPU)"
    )
    p.add_argument("--data-yaml", default=None, help="Path to data.yaml (overrides download)")
    p.add_argument("--model", default="yolo26s.pt", help="Base model to fine-tune from")
    args = p.parse_args()

    # Resolve data.yaml
    if args.data_yaml:
        data_yaml = Path(args.data_yaml)
    else:
        meta_path = ROOT / "data" / "raw" / "dataset_meta.json"
        if not meta_path.exists():
            print(f"ERROR: {meta_path} not found. Run scripts/download_dataset.py first.")
            return 1
        meta = json.loads(meta_path.read_text())
        if meta.get("source") != "roboflow":
            print(f"ERROR: dataset_meta.json source is {meta.get('source')!r}, not 'roboflow'.")
            print("Run scripts/download_dataset.py to get a recycling dataset.")
            return 1
        # Find data.yaml inside the dataset directory
        candidates = list((Path(meta["location"])).rglob("data.yaml"))
        if not candidates:
            print(f"ERROR: No data.yaml found in {meta['location']}")
            return 1
        data_yaml = candidates[0]

    print(f"Training config:")
    print(f"  data.yaml:   {data_yaml}")
    print(f"  base model:  {args.model}")
    print(f"  epochs:       {args.epochs}")
    print(f"  imgsz:        {args.imgsz}")
    print(f"  batch:        {args.batch}")
    print(f"  device:       {args.device}")
    print()

    # Train
    from ultralytics import YOLO  # type: ignore

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "models" / "train_runs"),
        name="yolo26s_recyclable",
        exist_ok=True,
        verbose=True,
    )

    # Find best.pt
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"ERROR: best.pt not found at {best_pt}")
        return 1

    # Move to models/ + export to ONNX
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    final_pt = models_dir / "yolo26s_recyclable.pt"
    final_onnx = models_dir / "yolo26s_recyclable.onnx"

    import shutil

    shutil.copy(best_pt, final_pt)
    print(f"\nSaved trained model: {final_pt}")

    print("Exporting to ONNX...")
    best_model = YOLO(str(best_pt))
    best_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    src_onnx = Path("yolo26s_recyclable.onnx")
    if src_onnx.exists():
        shutil.move(str(src_onnx), str(final_onnx))
        print(f"Saved ONNX model: {final_onnx}")

    # Final report
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Trained model:  {final_pt}")
    print(f"  ONNX export:    {final_onnx}")
    print(f"  Training runs:  {results.save_dir}")
    print()
    print("Next step: run the conveyor pipeline with the trained model:")
    print(f"  python -m conveyor_perception.app.conveyor \\")
    print(f"      --source data/sample/video.mp4 \\")
    print(f"      --model {final_onnx.name} \\")
    print(f"      --data-yaml {data_yaml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
