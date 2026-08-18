"""Multi-object tracker — ByteTrack wrapper for the perception layer.

Wraps the core `TrackingPipeline` with a thin API. The tracker adds
stable track_ids to Detection objects across frames.

Use:
    >>> from conveyor_perception.perception.track import Tracker
    >>> tracker = Tracker()
    >>> for frame in video:
    ...     detections = detector.detect(frame)
    ...     tracked = tracker.update(detections)
    ...     for d in tracked:
    ...         if d.track_id is not None:
    ...             print(f"ID {d.track_id}: {d.class_name}")
"""

from __future__ import annotations

from ..core.detection_pipeline import Detection
from ..core.tracking_pipeline import TrackingPipeline


class Tracker:
    """ByteTrack multi-object tracker with a clean public API.

    Wraps the core TrackingPipeline. By default uses the IoU fallback
    (no supervision dependency) for portability. Pass use_supervision=True
    in production to get the full ByteTrack 2-stage matching.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
        min_box_area: float = 100.0,
        use_supervision: bool = True,
    ):
        """Initialize the tracker.

        Args:
            track_thresh: High-confidence threshold for first-stage matching.
            track_buffer: Frames to keep a lost track alive before retirement.
            match_thresh: IoU threshold for matching detections to tracks.
            frame_rate: Assumed FPS (used for motion prediction in ByteTrack).
            min_box_area: Minimum bbox area to consider for tracking.
            use_supervision: Use supervision.ByteTrack (full ByteTrack 2-stage
                matching) vs. the IoU fallback. Default True; set False
                for lightweight environments or unit tests.
        """
        self.use_supervision = use_supervision
        # Try the supervision path first; fall back to IoU if unavailable.
        # The TrackingPipeline's force_fallback flag is set lazily based
        # on what the import succeeds with.
        self.pipeline = TrackingPipeline(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
            min_box_area=min_box_area,
            force_fallback=not use_supervision,
        )

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Update tracker with detections from the current frame.

        Args:
            detections: List of Detection objects from the Detector.

        Returns:
            The same list with track_id populated on each Detection.
        """
        return self.pipeline.update(detections)
