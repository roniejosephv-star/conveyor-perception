"""TrackingPipeline — ByteTrack multi-object tracking.

Stable IDs across frames. Uses Ultralytics' built-in ByteTrack via the
.track() method. The tracker is a thin wrapper: take detections, return
detections with stable track_id.

Why this abstraction exists:
- The conveyor moves. Objects must be tracked across frames so the robot
  pick has a consistent identity.
- ByteTrack is the right choice for conveyor (motion-based, no re-ID network,
  robust to occlusion via 2-stage matching).
- The wrapper means swapping trackers (BoT-SORT, OC-SORT) is a 1-line change.

ByteTrack config (defaults from Ultralytics bytetrack.yaml):
- track_thresh: 0.5 — high-confidence threshold for first-stage matching
- track_buffer: 30 — frames to keep a lost track alive
- match_thresh: 0.8 — IoU threshold for matching
- frame_rate: 30 — assumed FPS, used for motion prediction
"""

from __future__ import annotations

import logging

import numpy as np

from .detection_pipeline import Detection

logger = logging.getLogger(__name__)


class TrackingPipeline:
    """ByteTrack multi-object tracker.

    Takes a list of Detection objects per frame, returns the same list with
    stable track_id assigned to each detection. Objects that disappear from
    the frame (occluded, off-belt) keep their ID for `track_buffer` frames
    before being retired.

    Example:
        >>> tracker = TrackingPipeline(frame_rate=30)
        >>> for frame in video:
        ...     detections = detector.infer(frame)
        ...     tracked = tracker.update(detections)
        ...     for d in tracked:
        ...         print(f"ID {d.track_id}: {d.class_name} @ {d.bbox}")
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
        min_box_area: float = 100.0,
        force_fallback: bool = False,
    ):
        """Initialize the tracker.

        Args:
            track_thresh: High-confidence threshold for first-stage matching.
            track_buffer: Frames to keep a lost track alive before retirement.
            match_thresh: IoU threshold for matching detections to tracks.
            frame_rate: Assumed FPS (used for Kalman motion prediction).
            min_box_area: Minimum bbox area (pixels^2) to consider for tracking.
                Filters out spurious detections.
            force_fallback: If True, skip the supervision.ByteTrack import and
                use the simple IoU tracker. Useful for tests and lightweight
                Docker images where supervision is unavailable.
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.min_box_area = min_box_area
        self.force_fallback = force_fallback
        # Lazy init: ByteTrack is imported on first update() to keep the
        # import cost off the critical path.
        self._tracker = None
        self._next_track_id = 1
        # Fallback: simple IoU-based tracker if ByteTrack isn't installed.
        # Useful for testing and for environments where ultralytics isn't
        # available (e.g., lightweight Docker images).
        self._tracks: dict[int, dict] = {}

    def _ensure_tracker(self) -> None:
        """Lazy-import ByteTrack on first use (unless force_fallback)."""
        if self._tracker is not None:
            return
        if self.force_fallback:
            self._tracker = None
            return
        try:
            # Ultralytics 8.4.x ships ByteTrack as a tracker config.
            # We use the underlying BYTETracker from the supervision package
            # (or ultralytics' internal) when available.
            #
            # supervision 0.30.x: ByteTrack uses minimum_consecutive_frames
            # and minimum_iou_threshold. The 0.28+ API also has a frame_rate
            # parameter (we pass None since our frame_rate is for motion
            # prediction, not internal gating).
            from supervision import ByteTrack  # type: ignore

            # supervision >=0.28 renamed minimum_iou_threshold -> minimum_matching_threshold.
            # Try the new name first, fall back to the old one for older versions.
            try:
                self._tracker = ByteTrack(  # type: ignore[assignment]
                    minimum_consecutive_frames=1,
                    minimum_matching_threshold=self.match_thresh,
                )
            except TypeError:
                # Old API — unreachable on supervision >=0.28. The
                # `minimum_iou_threshold` kwarg was removed in 0.28; the
                # `type: ignore` silences mypy on a known-bad signature.
                self._tracker = ByteTrack(  # type: ignore[call-arg,assignment]
                    minimum_consecutive_frames=1,
                    minimum_iou_threshold=self.match_thresh,
                )
            logger.info("TrackingPipeline: using supervision.ByteTrack")
        except (ImportError, TypeError) as e:
            logger.warning(
                "supervision.ByteTrack unavailable (%s); using simple IoU tracker. "
                "Install with: pip install supervision>=0.30.0",
                e,
            )
            self._tracker = None

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Update tracker with detections from the current frame.

        Args:
            detections: List of Detection objects from DetectionPipeline.

        Returns:
            The same list with track_id populated on each Detection.
            Detections below min_box_area are filtered out.
        """
        # Filter out tiny detections
        filtered = [d for d in detections if self._bbox_area(d.bbox) >= self.min_box_area]
        self._ensure_tracker()
        if self._tracker is not None:
            return self._update_supervision(filtered)
        return self._update_iou_fallback(filtered)

    def _update_supervision(self, detections: list[Detection]) -> list[Detection]:
        """Use supervision.ByteTrack for tracking."""
        import supervision as sv  # type: ignore

        if not detections:
            return []
        assert self._tracker is not None  # caller guarantees this
        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)
        sv_detections = sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=class_id
        )
        tracked = self._tracker.update_with_detections(sv_detections)
        # ByteTrack may not return a tracker_id for new tracks on the first
        # frame (it needs `minimum_consecutive_frames` of history to confirm).
        # We preserve all input detections and look up tracker_id by index,
        # defaulting to None for tracks that aren't confirmed yet.
        tracker_ids = (
            list(tracked.tracker_id) if tracked.tracker_id is not None else []
        )
        result: list[Detection] = []
        for i, det in enumerate(detections):
            tid = tracker_ids[i] if i < len(tracker_ids) else None
            det.track_id = int(tid) if tid is not None else None
            result.append(det)
        return result

    def _update_iou_fallback(self, detections: list[Detection]) -> list[Detection]:
        """Simple IoU-based tracker as a fallback.

        Not as good as ByteTrack (no Kalman prediction, no two-stage matching),
        but works without supervision and is useful for tests + lightweight
        Docker images. For production, use the supervision path.
        """

        if not detections:
            # Increment age of all tracks; retire dead ones
            for tid in list(self._tracks.keys()):
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.track_buffer:
                    del self._tracks[tid]
            return []
        # Match detections to existing tracks via IoU
        matched: set[int] = set()
        new_tracks: dict[int, dict] = {}
        for det in detections:
            best_iou = 0.0
            best_tid: int | None = None
            for tid, track in self._tracks.items():
                if tid in matched:
                    continue
                if track["class_id"] != det.class_id:
                    continue
                iou = self._iou(det.bbox, track["bbox"])
                if iou > best_iou and iou >= self.match_thresh:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                det.track_id = best_tid
                new_tracks[best_tid] = {
                    "bbox": det.bbox,
                    "class_id": det.class_id,
                    "age": 0,
                }
                matched.add(best_tid)
            else:
                # New track
                tid = self._next_track_id
                self._next_track_id += 1
                det.track_id = tid
                new_tracks[tid] = {
                    "bbox": det.bbox,
                    "class_id": det.class_id,
                    "age": 0,
                }
        # Carry over unmatched tracks with incremented age
        for tid, track in self._tracks.items():
            if tid not in matched:
                track["age"] += 1
                if track["age"] <= self.track_buffer:
                    new_tracks[tid] = track
        self._tracks = new_tracks
        return detections

    @staticmethod
    def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _iou(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0.0
