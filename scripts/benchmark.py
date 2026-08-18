"""3-way benchmark: PyTorch vs ONNX vs TensorRT.

Runs the same image through all 3 model variants and reports
latency / throughput / model size. The "real-time performance" story
for the interview.

Usage:
    # Local Mac (CPU + MPS):
    python scripts/benchmark.py --model models/yolo26s_recyclable.pt

    # On Colab T4 GPU:
    python scripts/benchmark.py --model models/yolo26s_recyclable.pt \\
        --device cuda --imgsz 640 --num-runs 100

    # Skip ONNX or TensorRT (for machines that can't run them):
    python scripts/benchmark.py --model models/yolo26s_recyclable.pt \\
        --skip-onnx --skip-tensorrt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(
        description="3-way benchmark: PyTorch vs ONNX vs TensorRT"
    )
    p.add_argument("--model", required=True, help="YOLO .pt model path")
    p.add_argument("--device", default="cpu", help="cpu, mps, cuda, 0")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--num-runs", type=int, default=30, help="Runs per backend")
    p.add_argument("--warmup-runs", type=int, default=5, help="Warmup runs (excluded)")
    p.add_argument("--image", default=None, help="Test image (default: random)")
    p.add_argument("--skip-onnx", action="store_true", help="Skip ONNX backend")
    p.add_argument("--skip-tensorrt", action="store_true", help="Skip TensorRT backend")
    p.add_argument("--output", default=None, help="Save report to JSON file")
    args = p.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: model not found: {args.model}")
        return 1

    # Load test image
    if args.image and Path(args.image).exists():
        image = cv2.imread(args.image)
    else:
        # Random image of the right size
        image = (np.random.rand(args.imgsz, args.imgsz, 3) * 255).astype(np.uint8)
    print(f"Image shape: {image.shape}")

    # Run the benchmarks
    from conveyor_perception.optimization import (
        benchmark_onnx,
        benchmark_pytorch,
        compare_results,
    )

    results = []

    # 1. PyTorch
    print(f"\n[1/3] PyTorch benchmark (device={args.device})...")
    try:
        r_pt = benchmark_pytorch(
            args.model,
            image,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs,
            device=args.device,
            imgsz=args.imgsz,
        )
        results.append(r_pt)
        print(f"  mean: {r_pt.mean_ms:.1f}ms, p95: {r_pt.p95_ms:.1f}ms, "
              f"throughput: {r_pt.throughput_fps:.1f} FPS")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 2. ONNX
    onnx_path = str(Path(args.model).with_suffix(".onnx"))
    if not args.skip_onnx and Path(onnx_path).exists():
        print(f"\n[2/3] ONNX benchmark (file={onnx_path})...")
        try:
            r_onnx = benchmark_onnx(
                onnx_path,
                image,
                num_runs=args.num_runs,
                warmup_runs=args.warmup_runs,
                imgsz=args.imgsz,
            )
            results.append(r_onnx)
            print(f"  mean: {r_onnx.mean_ms:.1f}ms, p95: {r_onnx.p95_ms:.1f}ms, "
                  f"throughput: {r_onnx.throughput_fps:.1f} FPS")
        except Exception as e:
            print(f"  FAILED: {e}")
    elif not args.skip_onnx:
        # Auto-export
        print(f"\n[2/3] ONNX not found, exporting from {args.model}...")
        try:
            from conveyor_perception.optimization import export_onnx

            onnx_path = export_onnx(args.model, imgsz=args.imgsz)
            r_onnx = benchmark_onnx(
                onnx_path,
                image,
                num_runs=args.num_runs,
                warmup_runs=args.warmup_runs,
                imgsz=args.imgsz,
            )
            results.append(r_onnx)
            print(f"  mean: {r_onnx.mean_ms:.1f}ms, p95: {r_onnx.p95_ms:.1f}ms, "
                  f"throughput: {r_onnx.throughput_fps:.1f} FPS")
        except Exception as e:
            print(f"  FAILED: {e}")

    # 3. TensorRT (only on NVIDIA GPU)
    engine_path = str(Path(args.model).with_suffix(".engine"))
    if not args.skip_tensorrt and Path(engine_path).exists():
        print(f"\n[3/3] TensorRT benchmark (file={engine_path})...")
        # TensorRT benchmark would be similar to ONNX but via torch_tensorrt
        # or Ultralytics YOLO.engine inference
        try:
            from ultralytics import YOLO

            model = YOLO(engine_path)
            # Warmup
            for _ in range(args.warmup_runs):
                model.predict(image, imgsz=args.imgsz, device=args.device, verbose=False)
            latencies = []
            for _ in range(args.num_runs):
                t0 = time.perf_counter()
                model.predict(image, imgsz=args.imgsz, device=args.device, verbose=False)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            from conveyor_perception.optimization import BenchmarkResult

            results.append(
                BenchmarkResult(
                    name=f"TensorRT ({args.device})",
                    model_path=engine_path,
                    num_runs=args.num_runs,
                    mean_ms=sum(latencies) / n,
                    p50_ms=sorted_lat[min(n - 1, int(n * 0.50))],
                    p95_ms=sorted_lat[min(n - 1, int(n * 0.95))],
                    p99_ms=sorted_lat[min(n - 1, int(n * 0.99))],
                    min_ms=min(latencies),
                    max_ms=max(latencies),
                    throughput_fps=1000 / (sum(latencies) / n),
                    peak_memory_mb=0.0,
                    model_size_mb=Path(engine_path).stat().st_size / (1024 * 1024),
                    device=args.device,
                    imgsz=args.imgsz,
                )
            )
        except Exception as e:
            print(f"  FAILED: {e}")
    elif not args.skip_tensorrt:
        print(f"\n[3/3] TensorRT engine not found at {engine_path}. Skipping.")
        print("  (Run on a machine with NVIDIA GPU + tensorrt to enable.)")

    # Report
    if not results:
        print("\nNo successful benchmarks.")
        return 1

    print("\n" + "=" * 70)
    print(compare_results(results))
    print("=" * 70)

    if args.output:
        report_data = {
            "model": args.model,
            "device": args.device,
            "imgsz": args.imgsz,
            "num_runs": args.num_runs,
            "results": [r.to_dict() for r in results],
        }
        with open(args.output, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\nReport saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
