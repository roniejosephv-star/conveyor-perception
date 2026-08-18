"""Tests for MCPTriageSurface and InMemoryAlertQueue."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from conveyor_perception.core.triage_surface import (
    Alert,
    InMemoryAlertQueue,
    MCPTriageSurface,
)


def _alert(id: str, class_name: str = "PET", severity: str = "routine"):
    return Alert(
        id=id,
        timestamp=datetime.now(tz=timezone.utc),
        class_name=class_name,
        confidence=0.85,
        severity=severity,
    )


class TestInMemoryAlertQueue:
    def test_push_and_get_recent(self):
        q = InMemoryAlertQueue()
        for i in range(5):
            q.push(_alert(id=f"a{i}"))
        recent = q.get_recent(3)
        assert len(recent) == 3
        # Newest first
        assert recent[0].id == "a4"
        assert recent[1].id == "a3"
        assert recent[2].id == "a2"

    def test_classify_unknown_returns_unknown(self):
        q = InMemoryAlertQueue()
        q.push(_alert("a1", severity="attention"))
        assert q.classify("a1") == "attention"
        assert q.classify("nope") == "unknown"

    def test_escalate_marks_severity(self):
        q = InMemoryAlertQueue()
        q.push(_alert("a1", severity="routine"))
        q.escalate("a1", "high confidence but wrong class")
        assert q.classify("a1") == "escalate"
        recent = q.get_recent(1)
        assert recent[0].metadata["escalation_reason"] == "high confidence but wrong class"

    def test_get_health_returns_basics(self):
        q = InMemoryAlertQueue()
        for i in range(3):
            q.push(_alert(f"a{i}"))
        health = q.get_health()
        assert health["queue_size"] == 3
        assert "throughput_per_min" in health
        assert "drift_indicators" in health

    def test_log_resolution(self):
        q = InMemoryAlertQueue()
        q.push(_alert("a1"))
        q.log_resolution("a1", "auto-resolved")
        assert q._resolutions["a1"] == "auto-resolved"


class TestMCPTriageSurface:
    def test_surface_constructible(self):
        """Smoke test: the surface is constructible with a queue."""
        surface = MCPTriageSurface("test", InMemoryAlertQueue())
        assert surface.name == "test"
        assert surface.alert_source is not None

    def test_ensure_server_raises_without_fastmcp(self, monkeypatch):
        """If FastMCP is not importable, ensure_server should raise clearly.

        We simulate the missing import by removing `fastmcp` from sys.modules
        (and any cached submodule) so the from-import inside _ensure_server
        raises ImportError. The surface should translate that to RuntimeError.
        """
        import sys

        # Save the real module (if present) and replace with None
        real_fastmcp = sys.modules.get("fastmcp")
        sys.modules["fastmcp"] = None
        # Clear any cached submodule imports
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("fastmcp."):
                monkeypatch.delitem(sys.modules, mod_name, raising=False)
        try:
            surface = MCPTriageSurface("test", InMemoryAlertQueue())
            with pytest.raises(RuntimeError, match="FastMCP not installed"):
                surface._ensure_server()
        finally:
            # Restore
            if real_fastmcp is not None:
                sys.modules["fastmcp"] = real_fastmcp
            else:
                sys.modules.pop("fastmcp", None)

    def test_validate_reason_strips_control_chars(self):
        result = MCPTriageSurface._validate_reason("hello\x00world")
        assert result == "helloworld"

    def test_validate_reason_rejects_empty(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            MCPTriageSurface._validate_reason("   ")

    def test_validate_reason_rejects_too_long(self):
        with pytest.raises(ValueError, match="<= 500 chars"):
            MCPTriageSurface._validate_reason("a" * 501)

    def test_validate_action_accepts_allowed(self):
        for action in [
            "auto-resolved",
            "paged-on-call",
            "escalated-to-l2",
            "deferred",
            "false-positive",
        ]:
            assert MCPTriageSurface._validate_action(action) == action

    def test_validate_action_rejects_unknown(self):
        with pytest.raises(ValueError, match="action must be one of"):
            MCPTriageSurface._validate_action("made-up-action")
