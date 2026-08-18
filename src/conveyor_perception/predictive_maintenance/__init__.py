"""Predictive maintenance layer.

Turns drift signals from the core DriftMonitor (or any compatible source)
into actionable maintenance hints. The advisor is rule-based and auditable.
The L1 triage surface can route hints through the same alert queue.

Why a separate layer:
- DriftMonitor reports statistical signals (KS p-value, z-score, MAD).
- The advisor translates those into human-readable actions.
- This separation lets the rules evolve without touching the math.
"""

from conveyor_perception.predictive_maintenance.advisor import (
    DriftSignal,
    MaintenanceAdvisor,
    MaintenanceHint,
)

__all__ = [
    "DriftSignal",
    "MaintenanceAdvisor",
    "MaintenanceHint",
]
