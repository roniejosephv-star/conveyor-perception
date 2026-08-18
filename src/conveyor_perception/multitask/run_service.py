"""Long-running service: run the multitask pipeline in a loop.

This is the production entry point. It:
1. Loads the detector + tracker + drift + triage + dashboard
2. Opens a video source (file, webcam, RTSP, or gRPC stream)
3. Runs the pipeline on each frame
4. Pushes the result to the dashboard via HTTP (in production) or in-process
5. Logs to stdout for the operator

Usage:
    # Local
    python -m conveyor_perception.multitask.run_service

    # Docker (via docker-compose.yml)
    docker compose up perception

Configuration via env vars (production pattern):
    SOURCE:    video source (file path, 0 for webcam, rtsp://...)
    MODEL:     YOLO .pt or .onnx model path
    DATA_YAML: Roboflow data.yaml (for class names)
    CONF:      confidence threshold (default 0.25)
    DASHBOARD: dashboard service URL (default: in-process)
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def main() -> int:
    # Configure logging
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Read config from env
    source = os.environ.get("SOURCE", "data/sample/bus.jpg")
    model_path = os.environ.get("MODEL", "models/yolo26s_recyclable.onnx")
    data_yaml = os.environ.get("DATA_YAML", "data/raw/recycling_v3/data.yaml")
    conf = float(os.environ.get("CONF", "0.25"))
    dashboard_url = os.environ.get("DASHBOARD", "")  # empty = in-process

    # Build the pipeline components
    logger.info("Loading model from %s", model_path)
    from conveyor_perception.core.drift_monitor import DriftMonitor
    from conveyor_perception.core.tracking_pipeline import TrackingPipeline
    from conveyor_perception.monitoring.dashboard import MonitoringDashboard
    from conveyor_perception.multitask.pipeline import MultitaskPipeline
    from conveyor_perception.perception.detector import Detector
    from conveyor_perception.triage.agent import L1TriageAgent

    if not Path(model_path).exists():
        logger.error("Model not found: %s", model_path)
        logger.error("Did you run scripts/train_yolo26.py first?")
        return 1

    class_names = None
    if data_yaml and Path(data_yaml).exists():
        from conveyor_perception.perception.detector import _parse_yolo_classes

        class_names = _parse_yolo_classes(Path(data_yaml))
        logger.info("Loaded %d class names from %s", len(class_names), data_yaml)
    else:
        logger.warning("No data_yaml provided, using COCO classes")

    detector = Detector(
        model_path=model_path,
        class_names=class_names,
        conf_threshold=conf,
    )
    tracker = TrackingPipeline()
    drift = DriftMonitor(baseline_window=500)
    triage = L1TriageAgent()
    dashboard = MonitoringDashboard()

    pipeline = MultitaskPipeline(detector, tracker, drift, triage)

    # Open the source
    import cv2

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        if not Path(source).exists():
            logger.error("Source not found: %s", source)
            return 1
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("Could not open source: %s", source)
        return 1

    logger.info("Pipeline running on %s", source)
    logger.info("Press Ctrl-C to stop")

    # Handle SIGINT gracefully
    running = True

    def _stop(*_args: Any) -> None:
        nonlocal running
        running = False
        logger.info("Stopping...")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    frame_count = 0
    last_dashboard_push = 0.0
    dashboard_push_interval = 5.0  # seconds

    while running:
        ok, frame = cap.read()
        if not ok:
            # End of file or webcam disconnect
            if source.isdigit():
                logger.warning("Webcam disconnected, retrying in 1s")
                time.sleep(1.0)
                continue
            logger.info("End of source")
            break

        frame_count += 1
        result = pipeline.step(frame)
        dashboard.record_frame(result)

        # Push to external dashboard every N seconds
        now = time.time()
        if dashboard_url and (now - last_dashboard_push) > dashboard_push_interval:
            try:
                import requests

                requests.post(
                    f"{dashboard_url}/frame",
                    json=result.to_dict(),
                    timeout=5.0,
                )
                last_dashboard_push = now
            except Exception as e:
                logger.warning("Dashboard push failed: %s", e)
                last_dashboard_push = now  # don't spam

        if frame_count % 30 == 0:
            logger.info(
                "Frame %d: %d detections, %.1fms/frame, queue=%d",
                frame_count,
                len(result.detections),
                result.inference_ms,
                triage.get_stats().pushed,
            )

    cap.release()
    logger.info("Stopped after %d frames", frame_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
