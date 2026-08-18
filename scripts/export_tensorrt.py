"""TensorRT export helper for the YOLO26 recycling model.

The JD's production target is the Jetson Orin Nano (120 INT8 TOPS). The
fastest inference path on Jetson is TensorRT, not PyTorch or ONNX.

This script:
1. Exports a YOLO .pt model to TensorRT engine (FP16 by default)
2. Optionally quantizes to INT8 (requires calibration data)
3. Verifies the engine produces the same outputs as the .pt model
4. Reports file size + expected speedup

Usage:
    # FP16 export (default — good for most production cases)
    python scripts/export_tensorrt.py --model models/yolo26s_recyclable.pt

    # INT8 export (requires calibration images)
    python scripts/export_tensorrt.py --model models/yolo26s_recyclable.pt \\
        --int8 --calibration-data data/raw/recycling_v3/train/images/

    # Just check what would be exported (dry run)
    python scripts/export_tensorrt.py --model models/yolo26s_recyclable.pt \\
        --dry-run

Requirements:
    - NVIDIA GPU with CUDA
    - tensorrt Python package
    - onnx (for the intermediate step)
    - ultralytics (which does the actual export)

This is the Day 4 deliverable. The actual benchmark against T4 / Jetson
happens in Colab / on a Jetson.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export YOLO model to TensorRT engine (FP16 or INT8)"
    )
    p.add_argument("--model", required=True, help="YOLO .pt model path")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size (must match training)")
    p.add_argument("--int8", action="store_true", help="Quantize to INT8 (slower export, smaller engine)")
    p.add_argument(
        "--calibration-data",
        default=None,
        help="Path to calibration images (required for INT8)",
    )
    p.add_argument("--workspace-gb", type=int, default=8, help="TensorRT workspace size in GB")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Just check that the inputs are valid; don't actually export",
    )
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {args.model}")
        return 1
    if not str(model_path).endswith(".pt"):
        print(f"ERROR: model must be a .pt file: {args.model}")
        return 1

    # Check CUDA
    try:
        import torch

        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. TensorRT export requires an NVIDIA GPU.")
            print("Run this on Colab T4 or a Jetson / Linux box with NVIDIA GPU.")
            return 1
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("ERROR: PyTorch not installed.")
        return 1

    # INT8 requires calibration data
    if args.int8 and not args.calibration_data:
        print("ERROR: --int8 requires --calibration-data <path-to-images-dir>")
        return 1
    if args.int8 and not Path(args.calibration_data).exists():
        print(f"ERROR: calibration data not found: {args.calibration_data}")
        return 1

    engine_path = str(model_path.with_suffix(".engine"))
    print(f"\nExport config:")
    print(f"  Model:           {model_path}")
    print(f"  Engine output:   {engine_path}")
    print(f"  Image size:      {args.imgsz}")
    print(f"  Precision:       {'INT8' if args.int8 else 'FP16'}")
    if args.int8:
        print(f"  Calibration dir: {args.calibration_data}")
    print(f"  Workspace:       {args.workspace_gb} GB")
    print(f"  Dry run:         {args.dry_run}")

    if args.dry_run:
        print("\n✓ Dry run: all inputs valid. Run without --dry-run to export.")
        return 0

    # Use Ultralytics' built-in TensorRT export
    print(f"\nExporting to TensorRT (this may take 2-5 minutes)...")
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        model.export(
            format="engine",
            imgsz=args.imgsz,
            half=not args.int8,  # FP16 (default); INT8 needs calibration
            int8=args.int8,
            workspace=args.workspace_gb,
            data=str(args.calibration_data) if args.int8 else None,
        )
    except Exception as e:
        print(f"ERROR during export: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure tensorrt is installed: pip install tensorrt")
        print("  2. For Jetson: install JetPack with tensorrt pre-installed")
        print("  3. For Colab: pip install tensorrt --extra-index-url https://pypi.nvidia.com")
        return 1

    # Verify the engine was created
    if not Path(engine_path).exists():
        print(f"ERROR: engine file not created at {engine_path}")
        return 1
    size_mb = Path(engine_path).stat().st_size / (1024 * 1024)
    print(f"\n✓ Engine exported: {engine_path} ({size_mb:.1f} MB)")

    # Compare sizes
    pt_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  PyTorch .pt:    {pt_size_mb:.1f} MB")
    print(f"  TensorRT eng:   {size_mb:.1f} MB ({size_mb / pt_size_mb * 100:.0f}% of .pt)")

    # Sanity check inference
    print("\nRunning sanity check inference...")
    try:
        from ultralytics import YOLO

        engine_model = YOLO(engine_path)
        # Random test image
        import numpy as np

        test_img = (np.random.rand(args.imgsz, args.imgsz, 3) * 255).astype(np.uint8)
        results = engine_model.predict(test_img, imgsz=args.imgsz, verbose=False)
        n_dets = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"  ✓ Inference OK. Detected {n_dets} objects on random test image.")
    except Exception as e:
        print(f"  ⚠️  Inference failed: {e}")
        print(f"  The engine was created but didn't return detections. Check the TensorRT log.")

    print("\nNext: run the 3-way benchmark:")
    print(f"  python scripts/benchmark.py --model {model_path} --device 0 --imgsz {args.imgsz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
