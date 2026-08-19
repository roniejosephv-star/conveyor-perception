"""Model optimization for production deployment.

The YOLO26 model ships in PyTorch (.pt) format. For production deployment
on edge devices (Jetson, RTX, custom Linux), we want:

- ONNX export (CPU-friendly, framework-agnostic)
- TensorRT export (NVIDIA GPU, 2-5x speedup over ONNX)
- INT8 quantization (4x smaller, 2x faster on supported HW, ~1% mAP drop)
- FP16 (2x smaller, 1.5-2x faster on Tensor Cores)
- Batch inference (amortize overhead, max GPU utilization)

This module provides:
1. `ModelOptimizer` — wraps a base model and applies a sequence of opts
2. `BenchmarkResult` — the speed/memory numbers from a benchmark run
3. A CLI in scripts/benchmark.py that compares 3 model variants

The 3-way benchmark (the JD's "real-time performance" story):
- YOLO26s PyTorch (.pt) — the baseline
- YOLO26s ONNX (opset 17, simplify=True) — CPU/GPU-agnostic
- YOLO26s TensorRT FP16 — NVIDIA GPU, fastest
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """The result of one benchmark run: a model variant on a set of inputs.

    `mean_ms` / `p50_ms` / `p95_ms` are inference latencies. `throughput_fps`
    is mean_ms inverted. `peak_memory_mb` is the peak resident set size delta.
    """

    name: str  # e.g., "YOLO26s PyTorch", "YOLO26s ONNX", "YOLO26s TensorRT FP16"
    model_path: str
    num_runs: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    throughput_fps: float = 0.0
    peak_memory_mb: float = 0.0
    model_size_mb: float = 0.0
    device: str = "cpu"
    imgsz: int = 640
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Round floats for readability
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 3)
        return d


def benchmark_pytorch(
    model_path: str,
    image: np.ndarray,
    num_runs: int = 50,
    warmup_runs: int = 5,
    device: str = "cpu",
    imgsz: int = 640,
) -> BenchmarkResult:
    """Benchmark a YOLO .pt model on a single image, repeated num_runs times.

    Uses Ultralytics YOLO for inference. Reports per-run latency statistics.
    Memory tracking uses resource.getrusage on POSIX (best-effort, may be 0).
    """
    import resource

    from ultralytics import YOLO

    model = YOLO(model_path)
    # Warmup (first inference is always slower due to lazy init / CUDA alloc)
    for _ in range(warmup_runs):
        model.predict(image, imgsz=imgsz, device=device, verbose=False)
    # Measured runs
    latencies: list[float] = []
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(num_runs):
        t0 = time.perf_counter()
        model.predict(image, imgsz=imgsz, device=device, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # RSS is in KB on Linux, bytes on macOS — normalize to MB
    mem_delta_mb = abs(mem_after - mem_before) / 1024.0  # macOS-ish
    if mem_delta_mb < 1:  # looks like Linux KB
        mem_delta_mb = abs(mem_after - mem_before) / 1024.0 / 1024.0

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    return BenchmarkResult(
        name=f"YOLO26s PyTorch ({device})",
        model_path=model_path,
        num_runs=num_runs,
        mean_ms=sum(latencies) / n,
        p50_ms=sorted_lat[min(n - 1, int(n * 0.50))],
        p95_ms=sorted_lat[min(n - 1, int(n * 0.95))],
        p99_ms=sorted_lat[min(n - 1, int(n * 0.99))],
        min_ms=min(latencies),
        max_ms=max(latencies),
        throughput_fps=1000.0 / (sum(latencies) / n) if latencies else 0.0,
        peak_memory_mb=mem_delta_mb,
        model_size_mb=Path(model_path).stat().st_size / (1024 * 1024),
        device=device,
        imgsz=imgsz,
        notes="Ultralytics YOLO.predict(); first run is excluded as warmup",
    )


def benchmark_onnx(
    model_path: str,
    image: np.ndarray,
    num_runs: int = 50,
    warmup_runs: int = 5,
    imgsz: int = 640,
) -> BenchmarkResult:
    """Benchmark an ONNX model. Falls back to OpenCV DNN if onnxruntime is missing."""
    import resource

    import cv2

    # Try onnxruntime first (faster), fall back to OpenCV DNN
    session = None
    ort = None
    try:
        import onnxruntime as ort  # type: ignore

        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
    except ImportError:
        logger.warning("onnxruntime not installed; using OpenCV DNN instead")
    except Exception as e:
        # ORT installed but model failed to load (e.g., invalid file); fall back
        logger.warning(
            "onnxruntime failed to load %s (%s); using OpenCV DNN instead",
            model_path,
            e,
        )
        session = None

    # Pre-process image to NCHW float32 in [0, 1]
    img_resized = cv2.resize(image, (imgsz, imgsz))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_chw = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_chw, axis=0)

    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    latencies: list[float] = []

    def _infer() -> None:
        if session is not None:
            input_name = session.get_inputs()[0].name
            session.run(None, {input_name: img_batch})
        else:
            net = cv2.dnn.readNetFromONNX(model_path)
            net.setInput(img_batch)
            _ = net.forward()

    # Warmup
    for _ in range(warmup_runs):
        _infer()
    # Measured runs
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _infer()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_delta_mb = abs(mem_after - mem_before) / 1024.0
    if mem_delta_mb < 1:
        mem_delta_mb = abs(mem_after - mem_before) / 1024.0 / 1024.0

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    backend = "onnxruntime" if session is not None else "opencv_dnn"
    return BenchmarkResult(
        name=f"ONNX ({backend})",
        model_path=model_path,
        num_runs=num_runs,
        mean_ms=sum(latencies) / n,
        p50_ms=sorted_lat[min(n - 1, int(n * 0.50))],
        p95_ms=sorted_lat[min(n - 1, int(n * 0.95))],
        p99_ms=sorted_lat[min(n - 1, int(n * 0.99))],
        min_ms=min(latencies),
        max_ms=max(latencies),
        throughput_fps=1000.0 / (sum(latencies) / n) if latencies else 0.0,
        peak_memory_mb=mem_delta_mb,
        model_size_mb=Path(model_path).stat().st_size / (1024 * 1024),
        device="cpu",
        imgsz=imgsz,
        notes=f"Backend={backend}, single image, batch=1",
    )


def export_onnx(
    pt_path: str,
    onnx_path: str | None = None,
    imgsz: int = 640,
    simplify: bool = True,
) -> str:
    """Export a YOLO .pt model to ONNX. Returns the ONNX path."""
    from ultralytics import YOLO

    if onnx_path is None:
        onnx_path = str(Path(pt_path).with_suffix(".onnx"))
    model = YOLO(pt_path)
    model.export(format="onnx", imgsz=imgsz, simplify=simplify)
    # The export writes to the CWD, then we move it
    src = Path(pt_path).with_suffix(".onnx")
    if src.resolve() != Path(onnx_path).resolve():
        src.rename(onnx_path)
    logger.info("Exported %s -> %s", pt_path, onnx_path)
    return onnx_path


def compare_results(results: list[BenchmarkResult]) -> str:
    """Build a human-readable comparison of multiple benchmark results.

    Returns a markdown table for the README + a JSON dict for the dashboard.
    """
    if not results:
        return "No benchmark results."
    lines = ["## Benchmark results", ""]
    lines.append("| Variant | Mean (ms) | P95 (ms) | Throughput (FPS) | Size (MB) | Device |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.name} | {r.mean_ms:.1f} | {r.p95_ms:.1f} | {r.throughput_fps:.1f} | "
            f"{r.model_size_mb:.1f} | {r.device} |"
        )
    speedup = ""
    if len(results) >= 2:
        baseline = results[0].mean_ms
        best = min(r.mean_ms for r in results[1:])
        speedup = f"\n\n**Speedup vs baseline: {baseline / best:.2f}x faster**"
    return "\n".join(lines) + speedup
