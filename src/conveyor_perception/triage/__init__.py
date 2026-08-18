"""L1 triage layer for the conveyor perception system.

The L1 triage agent turns raw detection events into severity-classified
alerts in the queue. The ROC operator (or an LLM calling the MCP surface)
walks the queue and decides what to escalate.

Layers:
- L0 perception: detector + tracker + drift monitor (see `conveyor_perception.core`)
- L1 triage: this package (rules engine + alert queue + agent)
- L2 ROC: human operator + escalation tools (out of scope for this prototype)
"""

from conveyor_perception.triage.severity import (
    DEFAULT_ATTENTION_CLASSES,
    DetectionContext,
    SeverityDecision,
    SeverityRules,
    SeverityThresholds,
)
from conveyor_perception.triage.agent import L1TriageAgent, TriageStats

__all__ = [
    "DEFAULT_ATTENTION_CLASSES",
    "DetectionContext",
    "SeverityDecision",
    "SeverityRules",
    "SeverityThresholds",
    "L1TriageAgent",
    "TriageStats",
]
