"""DriftMonitor — production observability for perception models.

Watches production signals (per-class confidence, throughput, latency,
class count) and fires alerts when the live distribution drifts from
baseline. The 3 signals (per JD bullet 6):
1. Per-class confidence distribution drift (KS test)
2. Per-class count anomaly (z-score on rolling window)
3. Latency regression (median absolute deviation)

Why this abstraction exists:
- Models degrade in production. The conveyor's material mix changes.
  Lighting changes. New product types appear. The DriftMonitor catches
  these before they cause pick failures.
- The JD asks for "monitor in production, catch drift, design retraining
  loops". This is the answer: signals + thresholds + trigger.
- A real retraining loop is: DriftMonitor fires → ROC confirms → new data
  added to training set → next retrain. The DriftMonitor is the first step.

Why KS test (not just mean shift):
- The mean of per-class confidence can stay flat while the distribution
  shifts (e.g., from unimodal to bimodal). KS test catches both.
- scipy.stats.ks_2samp is fast, well-tested, and gives a p-value.
- Threshold: p < 0.05 = drift detected (5% significance).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProductionSignal:
    """A single production signal from the perception pipeline.

    Attributes:
        class_id: Integer class index of the detected object.
        confidence: Detection confidence in [0, 1].
        inference_time_ms: Time taken for this inference (milliseconds).
        timestamp: Unix timestamp (seconds) of the signal.
    """

    class_id: int
    confidence: float
    inference_time_ms: float
    timestamp: float


@dataclass
class DriftAlert:
    """An alert fired by the DriftMonitor.

    Attributes:
        drift_type: One of "confidence", "class_count", "latency".
        severity: One of "info", "warn", "critical".
        details: Free-form dict with metric values, threshold, p-value, etc.
        timestamp: When the alert was fired.
    """

    drift_type: str
    severity: str
    details: dict[str, Any]
    timestamp: float


@dataclass
class ClassBaseline:
    """Baseline statistics for a single class. Updated as new signals arrive."""

    class_id: int
    confidences: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    count_window: deque[int] = field(default_factory=lambda: deque(maxlen=100))

    @property
    def mean_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)

    @property
    def mean_count(self) -> float:
        if not self.count_window:
            return 0.0
        return sum(self.count_window) / len(self.count_window)


class DriftMonitor:
    """Production drift monitor for the perception pipeline.

    Watches 3 signals:
    1. Per-class confidence distribution drift (KS test on baseline vs. recent)
    2. Per-class count anomaly (z-score on rolling count window)
    3. Latency regression (MAD on recent inference times)

    Configure the baseline window size and drift threshold, then call
    update() per detection and check_drift() periodically (e.g., every 100
    signals or every minute).

    Example:
        >>> monitor = DriftMonitor(baseline_window=500, drift_threshold=0.05)
        >>> for detection in detections:
        ...     monitor.update(ProductionSignal(
        ...         class_id=detection.class_id,
        ...         confidence=detection.confidence,
        ...         inference_time_ms=12.3,
        ...         timestamp=time.time(),
        ...     ))
        >>> alert = monitor.check_drift()
        >>> if alert:
        ...     print(f"DRIFT: {alert.drift_type} ({alert.severity})")
    """

    def __init__(
        self,
        baseline_window: int = 500,
        drift_threshold: float = 0.05,
        count_z_threshold: float = 3.0,
        latency_mad_threshold: float = 3.0,
        min_samples_for_drift: int = 100,
    ):
        """Initialize the monitor.

        Args:
            baseline_window: Number of recent signals to use for the rolling
                baseline. Larger = more stable, slower to detect real drift.
            drift_threshold: p-value threshold for the KS test. Default 0.05
                (5% significance — same as the industry standard).
            count_z_threshold: z-score threshold for class count anomalies.
            latency_mad_threshold: MAD multiplier for latency regression.
            min_samples_for_drift: Minimum samples per class before drift
                detection kicks in. Prevents false positives during warmup.
        """
        self.baseline_window = baseline_window
        self.drift_threshold = drift_threshold
        self.count_z_threshold = count_z_threshold
        self.latency_mad_threshold = latency_mad_threshold
        self.min_samples_for_drift = min_samples_for_drift
        self._baselines: dict[int, ClassBaseline] = {}
        self._recent_latencies: deque[float] = deque(maxlen=baseline_window)
        self._class_count_window: dict[int, deque[int]] = {}
        self._count_window_size = 100
        self._alerts: list[DriftAlert] = []

    def update(self, signal: ProductionSignal) -> None:
        """Record a new production signal.

        Args:
            signal: The ProductionSignal from the perception pipeline.
        """
        if signal.class_id not in self._baselines:
            self._baselines[signal.class_id] = ClassBaseline(class_id=signal.class_id)
        self._baselines[signal.class_id].confidences.append(signal.confidence)
        self._recent_latencies.append(signal.inference_time_ms)
        # Tick the class count window (per-class count per tick)
        if signal.class_id not in self._class_count_window:
            self._class_count_window[signal.class_id] = deque(maxlen=self._count_window_size)
        # We increment on update; reset_count_window() is called at boundaries.
        window = self._class_count_window[signal.class_id]
        if window:
            window[-1] += 1
        else:
            window.append(1)

    def reset_count_window(self) -> None:
        """Call this at window boundaries (e.g., every 100 signals) to
        re-evaluate the class count anomaly."""
        for cid in self._class_count_window:
            self._class_count_window[cid].append(0)

    def check_drift(self) -> DriftAlert | None:
        """Check all 3 signals for drift. Returns the most severe alert, or None.

        Call this periodically (e.g., every 100 signals or every minute).
        Returns the highest-severity alert found. Lower-severity alerts are
        still recorded in self._alerts for the audit trail.
        """
        alerts: list[DriftAlert] = []
        for cid, baseline in self._baselines.items():
            if len(baseline.confidences) < self.min_samples_for_drift:
                continue
            alert = self._check_confidence_drift(cid, baseline)
            if alert is not None:
                alerts.append(alert)
        for cid, counts in self._class_count_window.items():
            if len(counts) < 10:  # need history
                continue
            alert = self._check_count_anomaly(cid, counts)
            if alert is not None:
                alerts.append(alert)
        if len(self._recent_latencies) >= self.min_samples_for_drift:
            alert = self._check_latency_drift()
            if alert is not None:
                alerts.append(alert)
        self._alerts.extend(alerts)
        if not alerts:
            return None
        # Return the highest severity
        severity_order = {"info": 0, "warn": 1, "critical": 2}
        alerts.sort(key=lambda a: severity_order.get(a.severity, 0), reverse=True)
        return alerts[0]

    def _check_confidence_drift(
        self, class_id: int, baseline: ClassBaseline
    ) -> DriftAlert | None:
        """KS test on per-class confidence. Returns alert if p<threshold."""
        try:
            from scipy.stats import ks_2samp  # type: ignore
        except ImportError:
            logger.warning("scipy not installed; skipping confidence drift check")
            return None
        confidences = list(baseline.confidences)
        n = len(confidences)
        if n < 2 * self.min_samples_for_drift:
            return None
        # Compare the first half (baseline) to the second half (recent)
        midpoint = n // 2
        baseline_sample = confidences[:midpoint]
        recent_sample = confidences[midpoint:]
        if len(baseline_sample) < 30 or len(recent_sample) < 30:
            return None
        result = ks_2samp(baseline_sample, recent_sample)
        if result.pvalue < self.drift_threshold:
            return DriftAlert(
                drift_type="confidence",
                severity="warn",
                details={
                    "class_id": class_id,
                    "ks_statistic": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "baseline_mean": sum(baseline_sample) / len(baseline_sample),
                    "recent_mean": sum(recent_sample) / len(recent_sample),
                },
                timestamp=_now(),
            )
        return None

    def _check_count_anomaly(
        self, class_id: int, counts: deque
    ) -> DriftAlert | None:
        """Z-score on per-class counts. Returns alert if |z|>threshold."""
        n = len(counts)
        if n < 10:
            return None
        counts_list = list(counts)
        mean = sum(counts_list[:-1]) / max(1, n - 1)
        variance = sum((c - mean) ** 2 for c in counts_list[:-1]) / max(1, n - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        latest = counts_list[-1]
        if std == 0:
            return None
        z = (latest - mean) / std
        if abs(z) > self.count_z_threshold:
            severity = "critical" if abs(z) > self.count_z_threshold * 2 else "warn"
            return DriftAlert(
                drift_type="class_count",
                severity=severity,
                details={
                    "class_id": class_id,
                    "z_score": round(z, 2),
                    "current_count": latest,
                    "baseline_mean": round(mean, 2),
                    "baseline_std": round(std, 2),
                },
                timestamp=_now(),
            )
        return None

    def _check_latency_drift(self) -> DriftAlert | None:
        """Median Absolute Deviation (MAD) on inference latencies."""
        latencies = list(self._recent_latencies)
        n = len(latencies)
        if n < self.min_samples_for_drift:
            return None
        sorted_lat = sorted(latencies)
        median = sorted_lat[n // 2]
        deviations = sorted(abs(latency - median) for latency in latencies)
        mad = deviations[n // 2]
        if mad == 0:
            return None
        # Modified z-score (Iglewicz & Hoaglin)
        modified_z = 0.6745 * (latencies[-1] - median) / mad
        if abs(modified_z) > self.latency_mad_threshold:
            severity = "critical" if abs(modified_z) > 6.0 else "warn"
            return DriftAlert(
                drift_type="latency",
                severity=severity,
                details={
                    "modified_z_score": round(modified_z, 2),
                    "current_latency_ms": latencies[-1],
                    "median_ms": round(median, 2),
                    "mad_ms": round(mad, 2),
                },
                timestamp=_now(),
            )
        return None

    def get_health(self) -> dict[str, Any]:
        """Return a snapshot of the monitor's health for dashboards."""
        return {
            "num_classes": len(self._baselines),
            "total_signals": sum(
                len(b.confidences) for b in self._baselines.values()
            ),
            "recent_latency_count": len(self._recent_latencies),
            "alerts_fired": len(self._alerts),
            "alerts_by_type": _count_by(self._alerts, "drift_type"),
            "alerts_by_severity": _count_by(self._alerts, "severity"),
        }


def _now() -> float:
    """Unix timestamp in seconds."""
    import time

    return time.time()


def _count_by(items: list, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        k = getattr(item, key, None)
        if k is not None:
            counts[k] = counts.get(k, 0) + 1
    return counts
