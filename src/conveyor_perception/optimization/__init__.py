"""Optimization layer.

Provides model export (ONNX, TensorRT-stub), benchmarking, and the 3-way
comparison that backs the "real-time performance" story in the interview.

The TensorRT path is documented but not executed in this environment
(no NVIDIA GPU). The code paths exist so a Colab T4 / Jetson Orin Nano
deployment can use them.
"""

from conveyor_perception.optimization.model_optimizer import (
    BenchmarkResult,
    benchmark_onnx,
    benchmark_pytorch,
    compare_results,
    export_onnx,
)

__all__ = [
    "BenchmarkResult",
    "benchmark_onnx",
    "benchmark_pytorch",
    "compare_results",
    "export_onnx",
]
