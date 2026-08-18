"""End-to-end multitask pipeline showcase.

Runs the full stack on a single image or short video, showing every layer
of the system in action:

    1. Detection (YOLO26 or COCO)
    2. Tracking (ByteTrack via supervision)
    3. Drift monitoring (KS test on confidence, z-score on counts, MAD on latency)
    4. L1 triage (7 severity rules)
    5. Predictive maintenance hints (drift signals → actionable advice)

This is the "20-second elevator pitch" of the framework: run it, see all
4 framework abstractions working together.

Usage:
    # With a custom recycling model (after running scripts/train_yolo26.py)
    python examples/multitask_demo.py --image data/sample/conveyor_frame.jpg \\
        --model models/yolo26s_recyclable.onnx \\
        --data-yaml data/raw/recycling_v3/data.yaml

    # With COCO pretrained
    python examples/multitask_demo.py --image data/sample/bus.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from conveyor_perception.core.drift_monitor import DriftMonitor
from conveyor_perception.core.tracking_pipeline import TrackingPipeline
from conveyor_perception.multitask.pipeline import MultitaskPipeline
from conveyor_perception.perception.detector import COCO_CLASSES, Detector
from conveyor_perception.predictive_maintenance.advisor import (
    DriftSignal,
    MaintenanceAdvisor,
)
from conveyor_perception.triage.agent import L1TriageAgent


def build_detector(args: argparse.Namespace) -> Detector:
    if args.model:
        class_names = None
        if args.data_yaml:
            from conveyor_perception.perception.detector import _parse_yolo_classes

            class_names = _parse_yolo_classes(Path(args.data_yaml))
        return Detector(
            model_path=args.model,
            class_names=class_names,
            conf_threshold=args.conf,
        )
    return Detector.from_coco_pretrained(conf_threshold=args.conf)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def build_drift_signals_from_result(result) -> list[DriftSignal]:
    """Convert the pipeline's drift status dict into DriftSignal objects
    that the MaintenanceAdvisor can consume."""
    signals = []
    drift = result.drift_signals
    if not drift or not drift.get("active"):
        return signals
    name = drift.get("drift_type", "unknown")
    signals.append(
        DriftSignal(
            name=name,
            active=True,
            p_value=drift.get("p_value"),
            z_score=drift.get("z_score"),
            mad_value=drift.get("mad_value"),
            current=drift.get("current"),
            baseline=drift.get("baseline"),
            extra={"message": drift.get("message", "")},
        )
    )
    return signals


def main() -> int:
    p = argparse.ArgumentParser(
        description="End-to-end multitask pipeline demo (detect → track → drift → triage → PM)"
    )
    p.add_argument("--image", required=True, help="Path to input image (jpg/png)")
    p.add_argument("--model", default=None, help="YOLO26 ONNX model (default: COCO pretrained)")
    p.add_argument("--data-yaml", default=None, help="Roboflow data.yaml for class names")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--frames", type=int, default=1, help="Number of times to run the same frame (for accumulating drift signals)")
    p.add_argument("--quiet", action="store_true", help="Suppress per-frame details")
    args = p.parse_args()

    print("=" * 70)
    print("CONVEYOR PERCEPTION — MULTITASK PIPELINE DEMO")
    print("=" * 70)
    print(f"Image:       {args.image}")
    print(f"Model:       {args.model or 'COCO pretrained (YOLO26s, 80 classes)'}")
    print(f"Data yaml:   {args.data_yaml or '(none, using COCO classes)'}")
    print(f"Confidence:  {args.conf}")
    print(f"Frames:      {args.frames} (running same image N times to accumulate drift signals)")
    print()

    # 1. Build the components
    detector = build_detector(args)
    tracker = TrackingPipeline()
    drift = DriftMonitor(baseline_window=50, min_samples_for_drift=20)
    triage = L1TriageAgent()
    advisor = MaintenanceAdvisor()
    pipeline = MultitaskPipeline(detector, tracker, drift, triage)

    # 2. Load the image
    image = load_image(args.image)
    print(f"Image shape: {image.shape}")
    print()

    # 3. Run the pipeline
    last_result = None
    for i in range(args.frames):
        result = pipeline.step(image)
        last_result = result

    if not args.quiet and args.frames > 0:
        # Detailed per-frame dump for the first frame
        r = last_result
        print(f"\n--- Frame {r.frame_idx} ---")
        print(f"Inference time: {r.inference_ms:.2f}ms")
        print(f"Detections:      {len(r.detections)}")
        for d in r.detections:
            tid = f" (track {d['track_id']})" if d.get("track_id") is not None else ""
            print(f"  - {d['class_name']}: {d['confidence']:.2f} at {d['bbox']}{tid}")

        print(f"\nDrift signals:   {r.drift_signals.get('active', False)}")
        if r.drift_signals.get("active"):
            print(f"  - type:     {r.drift_signals.get('drift_type')}")
            print(f"  - severity: {r.drift_signals.get('severity')}")
            print(f"  - message:  {r.drift_signals.get('message')}")

        print(f"\nAlerts: {len(r.alerts)}")
        for a in r.alerts:
            print(
                f"  - [{a['severity'].upper():9s}] {a['class_name']:10s} "
                f"conf={a['confidence']:.2f} reason='{a['metadata'].get('reason', '')[:60]}'"
            )

    # 4. Predictive maintenance hints
    signals = build_drift_signals_from_result(last_result)
    hints = advisor.advise(signals)
    if hints:
        print("\n--- PREDICTIVE MAINTENANCE HINTS ---")
        for h in hints:
            print(f"  [{h.severity.upper():8s}] {h.action}")
            print(f"             Why: {h.why}")
    else:
        print("\nNo predictive maintenance hints (no drift signals active).")

    # 5. Aggregate stats
    stats = pipeline.get_stats()
    print("\n--- AGGREGATE STATS ---")
    print(json.dumps(stats, indent=2))

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
