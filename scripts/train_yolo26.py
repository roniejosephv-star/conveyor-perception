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


# CRITICAL Colab platform self-heal (Aug 2026):
# When this script is invoked via `subprocess.run([sys.executable, ...])` from a
# Jupyter cell, the subprocess Python may NOT have Colab's site-packages
# (/usr/local/lib/python3.12/dist-packages — where `%pip install` writes) on
# its sys.path. The kernel does, but a fresh Python invocation doesn't unless
# we add it explicitly. Symptom of regression (run-1787150113.json follow-up):
# the kernel could `import ultralytics` but the subprocess `from ultralytics
# import YOLO` raised ModuleNotFoundError. The fix: add the path BEFORE any
# other import. Also run site.main() to re-process .pth files.
import os
import site as _site

for _candidate in (
    "/usr/local/lib/python3.12/dist-packages",
    "/usr/local/lib/python3.11/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
    "/usr/lib/python3.12/site-packages",
):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
try:
    _site.main()
except Exception:
    pass


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
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from models/train_runs/yolo26s_recyclable/weights/last.pt "
        "(continues to the --epochs total, e.g. --epochs 30 from epoch 15 -> 30).",
    )
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
        # FALLBACK PATH: Roboflow S3 was broken, so download_dataset.py fell back
        # to COCO pretrained. In that case, we don't have a recycling dataset, but
        # we CAN still use yolo26s.pt as the base model for the pipeline (it just
        # detects 80 COCO classes instead of 4 recycling classes). We copy the
        # COCO weights to the recyclable path with a clear warning so the demo
        # still runs end-to-end.
        if meta.get("source") == "ultralytics_coco_pretrained":
            print("=" * 70)
            print("  ⚠ RECYCLING TRAINING SKIPPED — Roboflow S3 export is broken")
            print("=" * 70)
            print(f"  Cause: {meta.get('description', 'Roboflow download failed')}")
            print()
            print("  Falling back to YOLO26s COCO pretrained (80 classes).")
            print("  The pipeline will run end-to-end but won't detect recycling")
            print("  classes (Glass/metal/plastic/vinyl) — it'll detect COCO classes")
            print("  (person, car, etc.). To enable recycling training once Roboflow")
            print("  S3 is fixed, re-run scripts/download_dataset.py.")
            print()
            # Copy COCO weights to the recyclable path
            import shutil
            src_pt = Path(meta.get("model_path", "models/yolo26s.pt"))
            if not src_pt.is_absolute():
                src_pt = ROOT / src_pt
            if not src_pt.exists():
                # Trigger auto-download
                from ultralytics import YOLO  # type: ignore
                YOLO("yolo26s.pt")
                src_pt = ROOT / "yolo26s.pt"
            final_pt = ROOT / "models" / "yolo26s_recyclable.pt"
            final_pt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_pt, final_pt)
            # ONNX export
            from ultralytics import YOLO  # type: ignore
            print(f"  Exporting {final_pt.name} to ONNX...")
            model = YOLO(str(final_pt))
            model.export(format="onnx", imgsz=args.imgsz, simplify=True)
            src_onnx = Path("yolo26s_recyclable.onnx")
            final_onnx = ROOT / "models" / "yolo26s_recyclable.onnx"
            if src_onnx.exists():
                shutil.move(str(src_onnx), str(final_onnx))
                print(f"  ✓ Saved ONNX: {final_onnx}")
            print()
            print("=" * 70)
            print("  PIPELINE READY (COCO pretrained fallback)")
            print("=" * 70)
            return 0
        if meta.get("source") not in ("roboflow", "bundled_demo"):
            print(f"ERROR: dataset_meta.json source is {meta.get('source')!r}, not 'roboflow' or 'bundled_demo'.")
            print("Run scripts/download_dataset.py to get a recycling dataset.")
            return 1
        # Find data.yaml inside the dataset directory
        candidates = list((Path(meta["location"])).rglob("data.yaml"))
        if not candidates:
            print(f"ERROR: No data.yaml found in {meta['location']}")
            return 1
        data_yaml = candidates[0]

    print("Training config:")
    print(f"  data.yaml:   {data_yaml}")
    print(f"  base model:  {args.model}")
    print(f"  epochs:       {args.epochs}")
    print(f"  imgsz:        {args.imgsz}")
    print(f"  batch:        {args.batch}")
    print(f"  device:       {args.device}")
    print(f"  resume:       {args.resume}")
    print()

    # Train
    from ultralytics import YOLO  # type: ignore

    if args.resume:
        last_pt = ROOT / "models" / "train_runs" / "yolo26s_recyclable" / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"ERROR: --resume requested but {last_pt} not found.")
            print("Train from scratch first (drop --resume).")
            return 1
        print(f"Resuming from {last_pt}")
        # CRITICAL: pass data= so Ultralytics builds the trainer with the
        # right number of classes (nc=4 for recycling). Without this, the
        # checkpoint's nc=4 weights are silently overridden with nc=80 (COCO)
        # and the resume fails with a size-mismatch error on the Detect head.
        model = YOLO(str(last_pt))
        results = model.train(
            data=str(data_yaml),
            resume=True,
            project=str(ROOT / "models" / "train_runs"),
            name="yolo26s_recyclable",
            exist_ok=True,
            device=args.device,
            verbose=True,
        )
    else:
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
    print("  python -m conveyor_perception.app.conveyor \\")
    print("      --source data/sample/video.mp4 \\")
    print(f"      --model {final_onnx.name} \\")
    print(f"      --data-yaml {data_yaml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
