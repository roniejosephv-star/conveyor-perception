"""Tests for notebooks/colab_session.py.

The session helpers are designed to be importable in vanilla Python
(not just Colab). These tests cover the pure-Python parts:

- SessionState log / error / metric round-trips
- get_state() / reset_state() singleton behavior
- The cell() context manager: success path, error path
- to_json() / to_dict() / summary_table()
- REMEDIATION_HINTS + hint_for()
- env_check() with mocked /proc/meminfo

Colab-specific bits (ipywidgets, google.colab, google.generativeai)
are NOT tested here — they're exercised manually in the notebook.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the notebooks/ dir importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

# Import after sys.path tweak
from colab_session import (  # type: ignore[import-not-found]  # noqa: E402
    REMEDIATION_HINTS,
    SessionState,
    cell,
    env_check,
    get_state,
    hint_for,
    reset_state,
    run_cell,
)


@pytest.fixture(autouse=True)
def _reset_state():
    """Each test gets a fresh SessionState singleton."""
    reset_state()
    yield
    reset_state()


class TestSessionState:
    def test_default_toggles_count(self):
        s = SessionState()
        # 4 abstractions + 8 modules (7 JD modules + 1 bonus triage)
        assert len(s.toggles) == 12
        assert all(s.toggles.values()), "all toggles should default to True"

    def test_log_appends_entry(self):
        s = SessionState()
        s.log("c1", action="install", detail="x")
        assert len(s.logs) == 1
        entry = s.logs[0]
        assert entry["cell_id"] == "c1"
        assert entry["action"] == "install"
        assert entry["detail"] == "x"
        assert "ts" in entry

    def test_error_captures_stack(self):
        s = SessionState()
        try:
            raise ValueError("boom")
        except ValueError as e:
            s.error("c1", e, hint="check the input")
        assert len(s.errors) == 1
        err = s.errors[0]
        assert err["type"] == "ValueError"
        assert err["message"] == "boom"
        assert err["hint"] == "check the input"
        assert "ValueError" in err["stack"]

    def test_metric_round_trip(self):
        s = SessionState()
        s.metric("mAP50", 0.671)
        s.metric("inference_ms", 8.7)
        assert s.metrics["mAP50"] == 0.671
        assert s.metrics["inference_ms"] == 8.7

    def test_to_dict_serializable(self):
        s = SessionState()
        s.metric("mAP50", 0.671)
        s.log("c1", action="install")
        d = s.to_dict()
        # Should be JSON-encodable
        s_json = json.dumps(d, default=str)
        assert "mAP50" in s_json
        assert "install" in s_json

    def test_to_json_includes_session_id(self):
        s = SessionState()
        s.session_id = "test-run-123"
        assert '"session_id": "test-run-123"' in s.to_json()

    def test_summary_table_contains_counts(self):
        s = SessionState()
        s.log("c1", action="install")
        s.error("c1", ValueError("x"))
        s.metric("mAP50", 0.671)
        text = s.summary_table()
        assert "Logs: 1" in text
        assert "Errors: 1" in text
        assert "Metrics: 1" in text
        assert "0.671" in text


class TestSingleton:
    def test_get_state_returns_same_instance(self):
        a = get_state()
        b = get_state()
        assert a is b

    def test_reset_state_clears(self):
        a = get_state()
        a.metric("test", 1)
        assert "test" in a.metrics

        b = reset_state()
        assert b is not a
        assert "test" not in b.metrics


class TestCellContext:
    def test_success_path_logs_ok(self):
        state = get_state()
        with cell("c1", action="install"):
            result = 1 + 1
        # Two log entries: start and ok
        assert len(state.logs) == 2
        assert state.logs[0]["status"] == "start"
        assert state.logs[1]["status"] == "ok"
        assert "elapsed_ms" in state.logs[1]

    def test_error_path_captures_and_reraises(self):
        state = get_state()
        with pytest.raises(ValueError, match="boom"):
            with cell("c1", action="install"):
                raise ValueError("boom")
        # Logs: start, error; Errors: 1
        assert len(state.logs) == 2
        assert state.logs[0]["status"] == "start"
        assert state.logs[1]["status"] == "error"
        assert len(state.errors) == 1
        assert state.errors[0]["type"] == "ValueError"

    def test_timing_recorded_on_success(self):
        state = get_state()
        with cell("c1", action="install"):
            pass
        elapsed = state.logs[1]["elapsed_ms"]
        assert isinstance(elapsed, (int, float))
        assert elapsed >= 0


class TestRunCell:
    def test_returns_fn_result(self):
        result = run_cell("c1", "test", lambda: 42)
        assert result == 42

    def test_logs_fn_result(self):
        state = get_state()
        run_cell("c1", "test", lambda: {"key": "value"})
        # start (cell), result (run_cell), ok (cell) — 3 entries
        assert len(state.logs) == 3
        # The middle one is the result log from run_cell
        result_logs = [log for log in state.logs if log.get("status") == "result"]
        assert len(result_logs) == 1
        assert result_logs[0]["result"] == {"key": "value"}

    def test_captures_fn_exception(self):
        state = get_state()
        with pytest.raises(RuntimeError, match="oops"):
            run_cell("c1", "test", lambda: (_ for _ in ()).throw(RuntimeError("oops")))
        assert len(state.errors) == 1


class TestRemediationHints:
    def test_all_hints_are_nonempty_strings(self):
        for k, v in REMEDIATION_HINTS.items():
            assert isinstance(v, str)
            assert len(v) > 10, f"Hint for {k!r} is too short"

    def test_hint_for_finds_by_message(self):
        exc = ValueError("Roboflow dataset not found at URL")
        hint = hint_for(exc)
        assert "Roboflow" in hint or "workspace" in hint

    def test_hint_for_finds_by_class_name(self):
        class CustomError(FileNotFoundError):
            pass

        exc = CustomError("whatever")
        hint = hint_for(exc)
        # Falls back to FileNotFoundError hint
        assert "Colab cwd" in hint or "path" in hint.lower()

    def test_hint_for_unknown_returns_generic(self):
        class WeirdError(Exception):
            pass

        exc = WeirdError("???")
        hint = hint_for(exc)
        assert "Coach" in hint or "hint" in hint.lower()


class TestEnvCheck:
    def test_env_check_returns_dict(self):
        info = env_check()
        assert isinstance(info, dict)
        assert "gpu" in info
        assert "ram_gb" in info
        assert "python" in info
        assert "is_colab" in info

    def test_env_check_python_version(self):
        info = env_check()
        # Should look like "3.X.Y"
        assert info["python"].count(".") >= 2

    def test_env_check_is_colab_false_locally(self):
        # When running tests outside Colab, is_colab should be False
        info = env_check()
        assert info["is_colab"] is False

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_env_check_handles_missing_meminfo(self, _mock_open):
        info = env_check()
        # Should not raise; ram_gb stays at 0
        assert info["ram_gb"] == 0


def test_html_helpers_exist():
    """The colab_session module must expose HTML helpers for the rich-output demo."""
    from colab_session import (
        render_css, render_hero, render_section_divider,
        render_status_pill, render_comparison_table, render_error_card, render_flow_diagram,
    )
    # All helpers return non-empty strings
    assert len(render_css()) > 1000, "render_css must include the full theme stylesheet"
    assert "tinkr-card" in render_hero("T", "S", "P", [{"title": "X", "value": "1"}]), \
        "render_hero must produce a tinkr-card"
    assert "STEP 2 of 5" in render_section_divider(2, 5, "T"), \
        "render_section_divider must include the step indicator"
    assert "p-ok" in render_status_pill("OK", "ok"), "render_status_pill must set a pill class"
    assert "tinkr-table" in render_comparison_table(["A", "B"], [["1", "2"]]), \
        "render_comparison_table must produce a table"
    assert "tinkr-error" in render_error_card("KeyError", "x", "hint"), \
        "render_error_card must produce an error card"
    assert "tinkr-flow" in render_flow_diagram("A → B"), "render_flow_diagram must wrap in flow class"
