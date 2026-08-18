"""End-to-end conveyor pipeline — the showcase entry point.

Reads a video (file or webcam), runs detection + tracking, emits events,
prints a summary at the end. This is the one-command demo the recruiter
runs to see the whole system in action.

Usage:
    # On a sample video file
    python -m conveyor_perception.app.conveyor --source data/sample/video.mp4

    # On a webcam (source=0)
    python -m conveyor_perception.app.conveyor --source 0

    # With a custom-trained recycling model
    python -m conveyor_perception.app.conveyor --source 0 \\
        --model models/recyclable_yolo26s.onnx \\
        --data-yaml data/raw/recyclable-waste/data.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..core.detection_pipeline import Detection
from ..perception.detector import COCO_CLASSES, Detector
from ..perception.track import Tracker

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end conveyor perception pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        required=True,
        help="Video source: path to file, or 0 for webcam",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Path to YOLO26 ONNX model. Default: COCO pretrained (auto-download).",
    )
    p.add_argument(
        "--data-yaml",
        default=None,
        help="Path to data.yaml (for custom class names from a Roboflow dataset).",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold (default 0.3 for demos, 0.5 for production)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Optional path to save annotated video (e.g. output/conveyor.mp4)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N frames (useful for benchmarks). Default: all.",
    )
    p.add_argument(
        "--no-tracker",
        action="store_true",
        help="Skip tracking (detection only).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-frame logging.",
    )
    return p.parse_args()


def load_class_names(args: argparse.Namespace) -> list[str]:
    """Resolve the class names list based on args."""
    if args.data_yaml:
        from ..perception.detector import _parse_yolo_classes

        return _parse_yolo_classes(Path(args.data_yaml))
    return COCO_CLASSES


def build_detector(args: argparse.Namespace, class_names: list[str]) -> Detector:
    if args.model:
        return Detector(
            model_path=args.model,
            class_names=class_names,
            conf_threshold=args.conf,
        )
    return Detector.from_coco_pretrained(conf_threshold=args.conf)


def open_source(source: str) -> cv2.VideoCapture:
    """Open a video file or webcam."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        if not Path(source).exists():
            raise FileNotFoundError(f"Source not found: {source}")
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return cap


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    class_names = load_class_names(args)
    logger.info(
        "Loaded %d class names (source: %s)",
        len(class_names),
        args.data_yaml or "COCO pretrained (80 classes)",
    )

    detector = build_detector(args, class_names)
    tracker = None if args.no_tracker else Tracker(frame_rate=30)
    cap = open_source(args.source)

    # Output video writer
    writer: cv2.VideoWriter | None = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        logger.info("Writing annotated output to %s", args.output)

    # Stats
    frame_count = 0
    total_detections = 0
    total_events = 0
    severity_counts = {"routine": 0, "attention": 0, "escalate": 0}
    inference_times_ms: list[float] = []
    track_ids_seen: set[int] = set()

    logger.info("Pipeline ready. Press Ctrl-C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            if args.max_frames and frame_count > args.max_frames:
                break

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            t1 = time.perf_counter()
            inference_times_ms.append((t1 - t0) * 1000)

            if tracker is not None:
                detections = tracker.update(detections)
                for d in detections:
                    if d.track_id is not None:
                        track_ids_seen.add(d.track_id)

            total_detections += len(detections)
            for d in detections:
                total_events += 1
                # Simple severity heuristic (used in interview; can be replaced
                # with the predictive_maintenance module on Day 2).
                if d.confidence < 0.4:
                    severity = "attention"
                elif d.confidence < 0.7:
                    severity = "routine"
                else:
                    severity = "routine"  # high-conf = routine
                severity_counts[severity] += 1

            if writer is not None:
                _, annotated = detector.detect_and_draw(frame)
                if tracker is not None:
                    for d in detections:
                        if d.track_id is not None:
                            x1, y1, x2, y2 = [int(round(v)) for v in d.bbox]
                            cv2.putText(
                                annotated,
                                f"ID {d.track_id}",
                                (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                writer.write(annotated)

            if not args.quiet and frame_count % 30 == 0:
                logger.info(
                    "Frame %d: %d detections, %.1fms/frame",
                    frame_count,
                    len(detections),
                    inference_times_ms[-1],
                )
    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl-C)")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    # Print summary
    avg_inference_ms = (
        sum(inference_times_ms) / len(inference_times_ms) if inference_times_ms else 0.0
    )
    p50 = (
        sorted(inference_times_ms)[len(inference_times_ms) // 2] if inference_times_ms else 0.0
    )
    p95 = (
        sorted(inference_times_ms)[int(len(inference_times_ms) * 0.95)]
        if inference_times_ms
        else 0.0
    )
    summary = {
        "frames_processed": frame_count,
        "total_detections": total_detections,
        "total_events": total_events,
        "unique_track_ids": len(track_ids_seen),
        "severity_counts": severity_counts,
        "inference_ms": {
            "mean": round(avg_inference_ms, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "min": round(min(inference_times_ms), 2) if inference_times_ms else 0.0,
            "max": round(max(inference_times_ms), 2) if inference_times_ms else 0.0,
        },
        "source": args.source,
        "model": args.model or "COCO pretrained (YOLO26s)",
    }
    print("\n" + "=" * 60)
    print("CONVEYOR PIPELINE SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
