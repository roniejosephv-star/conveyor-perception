"""Robustness test suite.

Runs the detector on a set of MRF-condition augmentations and reports
detection success rate + accuracy degradation per condition. This is the
"chaotic environments" requirement from the JD — the model needs to work
when the conditions aren't perfect.

Output: a RobustnessReport with one row per augmentation:
- name, description
- detection count (mean over N runs)
- mean confidence
- inference time (ms)
- degradation flag (e.g., <50% of baseline detections = "broken")

Usage:
    >>> suite = RobustnessTestSuite(detector, baseline_image)
    >>> report = suite.run()
    >>> print(report.to_markdown())
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from conveyor_perception.robustness.augmentations import (
    MRF_AUGMENTATIONS,
    AugmentationSpec,
)

logger = logging.getLogger(__name__)


@dataclass
class AugmentationResult:
    """The result of running the detector on one augmented image.

    `detection_count`: how many objects the detector found
    `mean_confidence`: mean of all detection confidences (0 if no detections)
    `inference_ms`: per-image inference time
    `detection_ratio`: detection_count / baseline_count (1.0 = same as baseline)
    `broken`: True if detection_count < 50% of baseline
    """

    name: str
    description: str
    args_summary: str
    detection_count: int = 0
    mean_confidence: float = 0.0
    inference_ms: float = 0.0
    detection_ratio: float = 0.0  # 1.0 = same as baseline
    broken: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 3)
        return d


@dataclass
class RobustnessReport:
    """The full report from RobustnessTestSuite.run().

    One row per augmentation + a baseline row at the top. The summary
    is the count of broken / degraded / OK augmentations.
    """

    baseline: AugmentationResult
    results: list[AugmentationResult] = field(default_factory=list)
    broken_count: int = 0
    degraded_count: int = 0  # <80% of baseline
    ok_count: int = 0  # >=80% of baseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "broken": self.broken_count,
                "degraded": self.degraded_count,
                "ok": self.ok_count,
                "total": len(self.results),
            },
        }

    def to_markdown(self) -> str:
        """Build a markdown table for the README or report."""
        lines = [
            "# Robustness Test Report",
            "",
            "| Augmentation | Args | Detections | Ratio | Mean Conf | Latency (ms) | Status |",
            "|---|---|---|---|---|---|---|",
        ]
        # Baseline row first
        b = self.baseline
        lines.append(
            f"| **BASELINE** (no aug) | — | {b.detection_count} | 1.00 | "
            f"{b.mean_confidence:.2f} | {b.inference_ms:.1f} | OK |"
        )
        for r in self.results:
            status = "❌ broken" if r.broken else ("⚠️ degraded" if r.detection_ratio < 0.80 else "✅ ok")
            lines.append(
                f"| {r.name} | {r.args_summary} | {r.detection_count} | "
                f"{r.detection_ratio:.2f} | {r.mean_confidence:.2f} | "
                f"{r.inference_ms:.1f} | {status} |"
            )
        lines.append("")
        lines.append(
            f"**Summary:** {self.broken_count} broken, {self.degraded_count} degraded, "
            f"{self.ok_count} ok (out of {len(self.results)})"
        )
        return "\n".join(lines)


class RobustnessTestSuite:
    """The test suite. Run the detector on each MRF augmentation.

    Args:
        detector: any object with a `detect(frame)` method that returns a list of Detection
        baseline_image: a single image used as the baseline (no augmentation)
        runs_per_augmentation: how many times to run each augmentation (default 1, increase for noise)
    """

    def __init__(
        self,
        detector: Any,
        baseline_image: np.ndarray,
        runs_per_augmentation: int = 1,
    ):
        self.detector = detector
        self.baseline_image = baseline_image
        self.runs_per_augmentation = max(1, int(runs_per_augmentation))

    def _measure_one(self, image: np.ndarray) -> tuple[int, float, float]:
        """Run the detector on one image. Return (count, mean_conf, inference_ms)."""
        t0 = time.perf_counter()
        dets = self.detector.detect(image)
        t1 = time.perf_counter()
        # dets may be a list[Detection] or a dict
        if isinstance(dets, dict):
            dets = dets.get("detections", [])
        count = len(dets)
        if count > 0:
            # Try to get confidences; works for Detection dataclass and dict
            confs = []
            for d in dets:
                if hasattr(d, "confidence"):
                    confs.append(d.confidence)
                elif isinstance(d, dict):
                    confs.append(d.get("conf", d.get("confidence", 0.0)))
            mean_conf = statistics.mean(confs) if confs else 0.0
        else:
            mean_conf = 0.0
        return count, mean_conf, (t1 - t0) * 1000.0

    def run(
        self,
        augmentations: list[AugmentationSpec] | None = None,
    ) -> RobustnessReport:
        """Run the full suite. Returns a RobustnessReport."""
        if augmentations is None:
            augmentations = MRF_AUGMENTATIONS

        # Baseline: run on the unmodified image
        baseline_count, baseline_conf, baseline_ms = self._measure_one(self.baseline_image)
        baseline = AugmentationResult(
            name="baseline",
            description="No augmentation (raw image)",
            args_summary="—",
            detection_count=baseline_count,
            mean_confidence=baseline_conf,
            inference_ms=baseline_ms,
            detection_ratio=1.0,
        )

        # Augmented: run on each MRF condition
        results: list[AugmentationResult] = []
        for spec in augmentations:
            try:
                aug_image = spec.fn(self.baseline_image)
            except Exception as e:
                logger.warning("Augmentation %s failed: %s", spec.name, e)
                continue
            # Average over N runs (for stochastic augmentations like noise/occlusion)
            counts, confs, mss = [], [], []
            for _ in range(self.runs_per_augmentation):
                c, conf, ms = self._measure_one(aug_image)
                counts.append(c)
                confs.append(conf)
                mss.append(ms)
            avg_count = int(statistics.mean(counts))
            avg_conf = statistics.mean(confs)
            avg_ms = statistics.mean(mss)
            # detection_ratio: relative to baseline (1.0 = same)
            ratio = avg_count / baseline_count if baseline_count > 0 else 0.0
            broken = ratio < 0.50 and baseline_count > 0
            results.append(
                AugmentationResult(
                    name=spec.name,
                    description=spec.description,
                    args_summary=spec.args_summary,
                    detection_count=avg_count,
                    mean_confidence=avg_conf,
                    inference_ms=avg_ms,
                    detection_ratio=ratio,
                    broken=broken,
                )
            )

        # Tally
        broken = sum(1 for r in results if r.broken)
        degraded = sum(1 for r in results if not r.broken and r.detection_ratio < 0.80)
        ok = sum(1 for r in results if r.detection_ratio >= 0.80)
        return RobustnessReport(
            baseline=baseline,
            results=results,
            broken_count=broken,
            degraded_count=degraded,
            ok_count=ok,
        )
