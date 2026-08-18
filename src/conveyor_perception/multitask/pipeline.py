"""Multitask pipeline — runs detection + tracking + drift + triage on each frame.

This is the "multitask" pattern in the JD: a single pipeline that produces
multiple outputs from one input (a frame). The architecture is:

    frame
      → Detector         (per-frame object detection)
      → Tracker          (per-frame ID assignment, motion, age)
      → DriftMonitor     (cumulative statistical signals)
      → L1TriageAgent    (severity classification + alert queue)

Each component is a separate concern with a clean interface. The pipeline
is the wiring. Swap any component for a production version without touching
the others.

Why a "multitask" pipeline (vs a single multitask neural network):
- A real multitask net would share a backbone across heads. That requires
  retraining. The pipeline approach uses off-the-shelf models and stitches
  them with clean Python. For the prototype/demo, this is the right trade.
- Each component is independently testable (we have 99 tests proving it).
- Swapping the detector for a Roboflow-trained model is a one-line change.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from conveyor_perception.core.detection_pipeline import Detection
from conveyor_perception.core.drift_monitor import DriftMonitor, ProductionSignal
from conveyor_perception.core.tracking_pipeline import TrackingPipeline
from conveyor_perception.triage.agent import L1TriageAgent
from conveyor_perception.triage.severity import DetectionContext

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """The output of MultitaskPipeline.step(frame).

    One per input frame. Carries every output the downstream consumers
    might need. The structure is JSON-serializable for logging.
    """

    frame_idx: int
    timestamp: float
    inference_ms: float
    detections: list[dict[str, Any]] = field(default_factory=list)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    drift_signals: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "inference_ms": round(self.inference_ms, 3),
            "detections": self.detections,
            "tracks": self.tracks,
            "drift_signals": self.drift_signals,
            "alerts": self.alerts,
            "metadata": self.metadata,
        }


def _detection_to_dict(d: Detection) -> dict[str, Any]:
    """Convert a Detection dataclass to a JSON-safe dict."""
    return {
        "class_id": d.class_id,
        "class_name": d.class_name,
        "confidence": d.confidence,
        "bbox": list(d.bbox) if d.bbox is not None else None,
        "track_id": d.track_id,
    }


class MultitaskPipeline:
    """The end-to-end pipeline. Wires the four core abstractions together.

    Usage:
        >>> detector = Detector.from_roboflow_dataset(...)
        >>> tracker = TrackingPipeline()
        >>> drift = DriftMonitor(baseline_window=500)
        >>> triage = L1TriageAgent()
        >>> pipeline = MultitaskPipeline(detector, tracker, drift, triage)
        >>> for frame in frames:
        ...     result = pipeline.step(frame)
        ...     process(result)

    The detector's `detect()` method must return a dict with key 'detections'
    whose value is a list[Detection]. The tracker then takes those detections
    and returns them with track_id populated.
    """

    def __init__(
        self,
        detector: Any,
        tracker: TrackingPipeline,
        drift_monitor: DriftMonitor,
        triage_agent: L1TriageAgent,
        confidence_threshold: float | None = None,
    ):
        self.detector = detector
        self.tracker = tracker
        self.drift_monitor = drift_monitor
        self.triage_agent = triage_agent
        # The threshold is informational; the actual filtering is done by the
        # detector at construction time. We keep it here so the API is symmetric
        # with the per-frame CLI and for future per-frame threshold support.
        self.confidence_threshold = confidence_threshold
        self._frame_count = 0
        # Per-track motion history: track_id -> deque of (timestamp, cx, cy)
        self._track_history: dict[int, deque] = {}

    def step(self, frame: Any) -> FrameResult:
        """Process a single frame. Returns a FrameResult with all outputs."""
        t0 = time.perf_counter()
        self._frame_count += 1

        # 1. Detect (detector API: detect(frame) returns a list of Detection)
        # Support both list[Detection] and a list of dicts.
        raw_dets = self.detector.detect(frame)
        if isinstance(raw_dets, dict):
            raw_dets = raw_dets.get("detections", [])
        # Normalize to Detection objects (the tracker's input type)
        detections: list[Detection] = []
        for d in raw_dets:
            if isinstance(d, Detection):
                detections.append(d)
            else:
                # Treat as dict-like; build a Detection
                bbox = d.get("bbox") if hasattr(d, "get") else None
                detections.append(
                    Detection(
                        class_id=d.get("class_id", -1) if hasattr(d, "get") else -1,
                        class_name=d.get("class_name", "unknown") if hasattr(d, "get") else "unknown",
                        confidence=d.get("conf", 0.0) if hasattr(d, "get") else 0.0,
                        bbox=tuple(bbox) if bbox is not None else (0.0, 0.0, 0.0, 0.0),
                        track_id=None,
                    )
                )

        # 2. Track (consume Detection list, returns same list with track_id)
        tracked = self.tracker.update(detections)
        now = time.time()
        for d in tracked:
            if d.track_id is not None:
                self._record_motion(d.track_id, d, now)

        # 3. Drift detection — one ProductionSignal per detection
        inference_ms_so_far = (time.perf_counter() - t0) * 1000.0
        for d in tracked:
            self.drift_monitor.update(
                ProductionSignal(
                    class_id=d.class_id,
                    confidence=d.confidence,
                    inference_time_ms=inference_ms_so_far,  # same for the frame
                    timestamp=now,
                )
            )
        # check_drift returns the most-severe DriftAlert or None
        drift_alert = self.drift_monitor.check_drift()
        # Build a small dict for the result + the active signal name for triage
        if drift_alert is not None:
            details = drift_alert.details or {}
            drift_status = {
                "active": True,
                "drift_type": drift_alert.drift_type,
                "severity": drift_alert.severity,
                "message": details.get("message", ""),
                "p_value": details.get("p_value"),
                "z_score": details.get("z_score"),
                "mad_value": details.get("mad_value"),
                "details": details,
            }
        else:
            drift_status = {"active": False}
        active_drift_names = (drift_alert.drift_type,) if drift_alert is not None else ()
        drift_active = drift_alert is not None

        # 4. Triage — for each detection, build a context and run the agent
        alerts: list[dict[str, Any]] = []
        for d in tracked:
            ctx = DetectionContext(
                class_name=d.class_name,
                confidence=d.confidence,
                bbox_area_ratio=self._bbox_area_ratio(d.bbox, frame),
                track_id=d.track_id,
                track_age_sec=self._track_age(d.track_id, now),
                track_motion_px=self._track_motion(d.track_id),
                drift_active=drift_active,
                drift_signals=active_drift_names,
                metadata={"frame_idx": self._frame_count},
            )
            alert = self.triage_agent.push_detection(ctx)
            alerts.append(alert.to_dict())

        inference_ms = (time.perf_counter() - t0) * 1000.0

        return FrameResult(
            frame_idx=self._frame_count,
            timestamp=now,
            inference_ms=inference_ms,
            detections=[_detection_to_dict(d) for d in tracked],
            tracks=[
                {
                    "track_id": d.track_id,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": list(d.bbox) if d.bbox is not None else None,
                }
                for d in tracked
                if d.track_id is not None
            ],
            drift_signals=drift_status,
            alerts=alerts,
            metadata={"frame_shape": self._frame_shape(frame)},
        )

    def _record_motion(self, track_id: int, det: Detection, now: float) -> None:
        if track_id not in self._track_history:
            self._track_history[track_id] = deque(maxlen=20)
        bbox = det.bbox
        if bbox is None or len(bbox) != 4:
            return
        try:
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        except (TypeError, ValueError):
            return
        self._track_history[track_id].append((now, cx, cy))

    def _track_age(self, track_id: int | None, now: float) -> float:
        if track_id is None or track_id not in self._track_history:
            return 0.0
        history = self._track_history[track_id]
        if not history:
            return 0.0
        return now - history[0][0]

    def _track_motion(self, track_id: int | None) -> float:
        """Return recent motion in px/sec. |now_pos - prev_pos| / dt."""
        if track_id is None or track_id not in self._track_history:
            return 0.0
        history = self._track_history[track_id]
        if len(history) < 2:
            return 0.0
        t_now, x_now, y_now = history[-1]
        t_prev, x_prev, y_prev = history[0]
        if t_now - t_prev < 1e-6:
            return 0.0
        dx = x_now - x_prev
        dy = y_now - y_prev
        return (dx * dx + dy * dy) ** 0.5 / (t_now - t_prev)

    @staticmethod
    def _bbox_area_ratio(bbox: Any, frame: Any) -> float:
        if bbox is None or frame is None:
            return 0.0
        try:
            if len(bbox) != 4:
                return 0.0
            x1, y1, x2, y2 = bbox
            box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            if hasattr(frame, "shape") and len(frame.shape) >= 2:
                h, w = frame.shape[0], frame.shape[1]
                frame_area = float(w * h)
                if frame_area <= 0:
                    return 0.0
                return min(1.0, box_area / frame_area)
        except (TypeError, ValueError, IndexError):
            return 0.0
        return 0.0

    @staticmethod
    def _frame_shape(frame: Any) -> tuple | None:
        if hasattr(frame, "shape"):
            try:
                return tuple(int(d) for d in frame.shape)
            except Exception:
                return None
        return None

    def get_stats(self) -> dict[str, Any]:
        """Aggregate stats across the pipeline."""
        triage_stats = self.triage_agent.get_stats().to_dict()
        return {
            "frames_processed": self._frame_count,
            "active_tracks": len(self._track_history),
            "triage": triage_stats,
        }
