"""ROC dashboard (FastAPI web service).

The dashboard reads the perception service's frame results and serves:
- /snapshot         — live counters
- /shift-report     — supervisor's 8am view
- /alerts           — current alert queue
- /frame (POST)     — push a new frame result
- /reset (POST)     — reset counters (start of shift)

In production, this would be backed by a database (Redis/Postgres) instead
of the in-memory MonitoringDashboard. The interface stays the same.
"""

from conveyor_perception.dashboard.app import (
    FASTAPI_AVAILABLE,
    get_dashboard,
)

__all__ = ["FASTAPI_AVAILABLE", "get_dashboard"]
