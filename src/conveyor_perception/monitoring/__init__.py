"""Monitoring layer.

Aggregates real-time metrics from perception + triage + drift into a
single dashboard. The ROC operator reads the per-shift report at the
start of every shift. The dashboard also auto-detects when retraining
should be triggered.
"""

from conveyor_perception.monitoring.dashboard import (
    MonitoringDashboard,
    ShiftReport,
)

__all__ = ["MonitoringDashboard", "ShiftReport"]
