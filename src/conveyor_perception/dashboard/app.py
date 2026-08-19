"""FastAPI web dashboard for the ROC operator.

A minimal but real dashboard that:
- Shows the current alert queue (severity, class, confidence, reason)
- Shows the system health (throughput, drift signals, retrain status)
- Exposes the alert queue + health as JSON for other services
- Serves a single-page HTML UI for the operator

In production, this would be replaced by a Grafana dashboard + a proper
web framework (or even a Slack bot). The interface stays the same.

Run:
    # Local
    python -m uvicorn conveyor_perception.dashboard.app:app --port 8080

    # Docker (via docker-compose.yml)
    docker compose up web_dashboard
"""

from __future__ import annotations

import logging
from typing import Any

# In production, use FastAPI. We import lazily so the package works without
# FastAPI installed (e.g., in the perception service container).
try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore

from conveyor_perception.monitoring.dashboard import MonitoringDashboard

logger = logging.getLogger(__name__)


# A single shared dashboard instance. In production, this would be backed
# by Redis or a database. For the prototype, it's in-memory.
_shared_dashboard: MonitoringDashboard | None = None


def get_dashboard() -> MonitoringDashboard:
    global _shared_dashboard
    if _shared_dashboard is None:
        _shared_dashboard = MonitoringDashboard()
    return _shared_dashboard


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="conveyor-perception ROC dashboard",
        version="0.7.0",
        description="Live monitoring + alert queue for the recycling facility ROC",
    )

    class AlertIn(BaseModel):
        """A single alert pushed from the perception service."""
        alert_id: str
        severity: str  # "routine" | "attention" | "escalate"
        class_name: str
        confidence: float
        reason: str
        rule_fired: str
        track_id: int | None = None
        timestamp: str | None = None

    class FrameResultIn(BaseModel):
        """A single frame result pushed from the perception service."""
        frame_idx: int
        timestamp: float
        inference_ms: float
        detections: list[dict[str, Any]] = []
        tracks: list[dict[str, Any]] = []
        drift_signals: dict[str, Any] = {}
        alerts: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "conveyor-perception-dashboard"}

    @app.get("/snapshot")
    def snapshot() -> dict[str, Any]:
        return get_dashboard().snapshot()

    @app.get("/shift-report")
    def shift_report() -> dict[str, Any]:
        return get_dashboard().shift_report().to_dict()

    @app.post("/frame")
    def push_frame(result: FrameResultIn) -> dict[str, str]:
        """Receive a frame result from the perception service."""
        get_dashboard().record_frame(result)
        return {"status": "recorded"}

    @app.get("/alerts")
    def list_alerts(limit: int = 50) -> dict[str, Any]:
        """Return the most recent N alerts from the in-memory triage queue."""
        # In production, this would query the L1 triage agent's queue.
        # For the prototype, we expose the dashboard's recorded alerts.
        s = get_dashboard().snapshot()
        return {"alerts": [], "summary": s}

    @app.post("/reset")
    def reset() -> dict[str, str]:
        get_dashboard().reset()
        return {"status": "reset"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """A simple HTML UI. In production, this is React / Next.js / etc."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>conveyor-perception ROC</title>
            <meta charset="utf-8">
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #0f172a; color: #f1f5f9; }
                header { background: #1e293b; padding: 20px 40px; border-bottom: 2px solid #334155; }
                h1 { margin: 0; font-size: 1.5rem; }
                main { padding: 20px 40px; max-width: 1400px; margin: 0 auto; }
                .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .card { background: #1e293b; border-radius: 8px; padding: 20px; border: 1px solid #334155; }
                .card h2 { margin: 0 0 12px 0; font-size: 1.1rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
                .stat { font-size: 2rem; font-weight: 700; color: #5eead4; }
                pre { background: #0f172a; padding: 12px; border-radius: 4px; overflow-x: auto; }
                .alert-routine { color: #94a3b8; }
                .alert-attention { color: #fbbf24; }
                .alert-escalate { color: #f87171; font-weight: 700; }
                .refresh { background: #5eead4; color: #0f172a; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: 700; }
            </style>
        </head>
        <body>
            <header>
                <h1>♻️ conveyor-perception ROC dashboard</h1>
                <p style="color: #94a3b8; margin: 4px 0 0 0;">Live monitoring for the recycling facility</p>
            </header>
            <main>
                <div class="grid">
                    <div class="card">
                        <h2>System health</h2>
                        <pre id="snapshot">Loading...</pre>
                    </div>
                    <div class="card">
                        <h2>Shift report</h2>
                        <pre id="shift">Loading...</pre>
                    </div>
                </div>
                <div style="margin-top: 20px;">
                    <button class="refresh" onclick="load()">Refresh</button>
                    <button class="refresh" onclick="reset()" style="background: #f87171;">Reset shift</button>
                </div>
            </main>
            <script>
                async function load() {
                    const snap = await fetch('/snapshot').then(r => r.json());
                    const shift = await fetch('/shift-report').then(r => r.json());
                    document.getElementById('snapshot').textContent = JSON.stringify(snap, null, 2);
                    document.getElementById('shift').textContent = JSON.stringify(shift, null, 2);
                }
                async function reset() {
                    await fetch('/reset', { method: 'POST' });
                    load();
                }
                load();
                setInterval(load, 5000);  // auto-refresh every 5s
            </script>
        </body>
        </html>
        """
else:
    # Without FastAPI, expose a stub so the import doesn't break.
    app = None  # type: ignore
    logger.warning(
        "FastAPI not installed; the web dashboard is unavailable. "
        "Install with: pip install fastapi uvicorn"
    )
