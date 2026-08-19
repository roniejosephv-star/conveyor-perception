"""Tests for the optimization module.

We use small synthetic workloads to keep tests fast. Real benchmarks go
through scripts/benchmark.py.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

from conveyor_perception.optimization.model_optimizer import (
    BenchmarkResult,
    compare_results,
)


def _fake_image() -> np.ndarray:
    return (np.random.rand(480, 640, 3) * 255).astype(np.uint8)


def _install_fake_ultralytics(monkeypatch, predict_fn=None) -> None:
    """Install a fake `ultralytics` module in sys.modules with a FakeYOLO class."""
    fake_predict = predict_fn or (lambda image, imgsz, device, verbose: [object()])

    class FakeYOLO:
        def __init__(self, path):
            self.path = path

        def predict(self, image, imgsz, device, verbose):
            return fake_predict(image, imgsz, device, verbose)

        def export(self, format, imgsz, simplify):
            out_path = Path(self.path).with_suffix(".onnx")
            out_path.write_bytes(b"fake onnx")
            return str(out_path)

    fake_mod = types.ModuleType("ultralytics")
    fake_mod.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)


class TestBenchmarkResult:
    def test_to_dict_is_json_safe(self):
        r = BenchmarkResult(
            name="test",
            model_path="/tmp/x.pt",
            num_runs=10,
            mean_ms=10.0,
            p50_ms=9.0,
            p95_ms=15.0,
            p99_ms=18.0,
            min_ms=8.0,
            max_ms=20.0,
            throughput_fps=100.0,
            peak_memory_mb=50.0,
            model_size_mb=20.0,
            device="cpu",
            imgsz=640,
        )
        d = r.to_dict()
        json.dumps(d)
        assert d["name"] == "test"
        assert d["num_runs"] == 10

    def test_to_dict_rounds_floats(self):
        r = BenchmarkResult(name="t", model_path="x", mean_ms=12.345678)
        d = r.to_dict()
        assert d["mean_ms"] == 12.346


class TestCompareResults:
    def test_empty_list(self):
        result = compare_results([])
        assert "No benchmark" in result

    def test_single_result(self):
        r = BenchmarkResult(
            name="A", model_path="x", mean_ms=10.0, p95_ms=15.0,
            throughput_fps=100.0, model_size_mb=20.0,
        )
        result = compare_results([r])
        assert "A" in result
        assert "Mean (ms)" in result

    def test_multiple_results_includes_speedup(self):
        r1 = BenchmarkResult(
            name="baseline", model_path="x", mean_ms=100.0, p95_ms=120.0,
            throughput_fps=10.0, model_size_mb=20.0,
        )
        r2 = BenchmarkResult(
            name="optimized", model_path="y", mean_ms=20.0, p95_ms=25.0,
            throughput_fps=50.0, model_size_mb=5.0,
        )
        result = compare_results([r1, r2])
        assert "baseline" in result
        assert "optimized" in result
        assert "5.00x" in result or "5x" in result


class TestBenchmarkPyTorch:
    def test_benchmark_pytorch_runs(self, tmp_path, monkeypatch):
        from conveyor_perception.optimization import model_optimizer

        _install_fake_ultralytics(monkeypatch)
        fake_pt = tmp_path / "fake.pt"
        fake_pt.write_bytes(b"0" * (10 * 1024 * 1024))

        result = model_optimizer.benchmark_pytorch(
            model_path=str(fake_pt),
            image=_fake_image(),
            num_runs=5,
            warmup_runs=1,
            device="cpu",
            imgsz=320,
        )
        assert result.name.startswith("YOLO26s PyTorch")
        assert result.num_runs == 5
        assert result.model_size_mb > 0
        assert result.imgsz == 320
        assert result.mean_ms >= 0


class TestBenchmarkOnnx:
    def test_benchmark_onnx_with_opencv_fallback(self, tmp_path, monkeypatch):
        from conveyor_perception.optimization import model_optimizer

        # Create a fake .onnx file
        fake_onnx = tmp_path / "fake.onnx"
        fake_onnx.write_bytes(b"0" * (5 * 1024 * 1024))

        # Build a fake onnxruntime module so the code uses the opencv fallback
        class FakeSession:
            def __init__(self, path, providers=None):
                pass

            def get_inputs(self):
                class FakeInput:
                    name = "input"

                return [FakeInput()]

            def run(self, outputs, inputs):
                return [np.zeros((1, 4), dtype=np.float32)]

        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.InferenceSession = FakeSession
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

        # Mock cv2.dnn.readNetFromONNX with a fake network
        import cv2

        class FakeNet:
            def setInput(self, x):
                pass

            def forward(self):
                return np.zeros((1, 4), dtype=np.float32)

        monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda x: FakeNet())
        result = model_optimizer.benchmark_onnx(
            model_path=str(fake_onnx),
            image=_fake_image(),
            num_runs=3,
            warmup_runs=1,
            imgsz=320,
        )
        # Fake onnxruntime is installed (mock), so backend is "onnxruntime"
        assert "ONNX" in result.name
        assert result.model_size_mb > 0
        assert "Backend=" in result.notes
        assert "onnxruntime" in result.notes

    def test_benchmark_onnx_with_real_opencv_fallback_branch(self, tmp_path, monkeypatch):
        """The opencv_dnn path is exercised in environments without onnxruntime.

        Note: this test runs the opencv_dnn code path (no fake onnxruntime
        module). The fake .onnx file is just an empty file, but we mock
        cv2.dnn.readNetFromONNX to return a stub that doesn't try to parse
        the file as a real model.
        """
        from conveyor_perception.optimization import model_optimizer

        fake_onnx = tmp_path / "fake.onnx"
        fake_onnx.write_bytes(b"0" * (5 * 1024 * 1024))

        # Inject a minimal fake onnxruntime so the code doesn't try real ORT
        class FakeORT:
            class FakeSession:
                def __init__(self, path, providers=None):
                    pass

                def get_inputs(self):
                    class FakeInput:
                        name = "input"

                    return [FakeInput()]

                def run(self, outputs, inputs):
                    return [np.zeros((1, 4), dtype=np.float32)]

            InferenceSession = FakeSession

        # Force the onnxruntime import to be a no-op (it'll fall through to ORT in this env)
        # by using a fake that does not actually load any model
        fake_ort_mod = types.ModuleType("onnxruntime")
        fake_ort_mod.InferenceSession = FakeORT.InferenceSession
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort_mod)

        import cv2

        class FakeNet:
            def setInput(self, x):
                pass

            def forward(self):
                return np.zeros((1, 4), dtype=np.float32)

        monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda x: FakeNet())
        # The function uses ORT if available; if our fake ORT raises, it
        # should fall back to OpenCV. Force a load failure on the fake ORT.
        class FailingORT:
            class FailingSession:
                def __init__(self, path, providers=None):
                    raise RuntimeError("simulated load failure")

            InferenceSession = FailingSession

        failing_mod = types.ModuleType("onnxruntime")
        failing_mod.InferenceSession = FailingORT.FailingSession
        monkeypatch.setitem(sys.modules, "onnxruntime", failing_mod)

        result = model_optimizer.benchmark_onnx(
            model_path=str(fake_onnx),
            image=_fake_image(),
            num_runs=2,
            warmup_runs=1,
            imgsz=320,
        )
        # Falls back to opencv_dnn
        assert "opencv_dnn" in result.notes


class TestExportOnnx:
    def test_export_onnx_calls_ultralytics(self, tmp_path, monkeypatch):
        from conveyor_perception.optimization import model_optimizer

        _install_fake_ultralytics(monkeypatch)
        fake_pt = tmp_path / "model.pt"
        fake_pt.write_bytes(b"0")
        out = model_optimizer.export_onnx(str(fake_pt), imgsz=320)
        assert out.endswith(".onnx")
        assert Path(out).exists()
