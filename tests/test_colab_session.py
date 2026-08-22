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
    pick_device,
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

    def test_systemexit_zero_logged_as_skipped_not_error(self):
        """SystemExit(0) is used as a 'skip the rest' signal by cells like
        cell-9-prod. cell() must log it as 'skipped' (not 'error') and
        re-raise so the rest of the cell is skipped cleanly.
        """
        state = get_state()
        state.logs.clear()  # isolate
        with pytest.raises(SystemExit):
            with cell("c1", action="test-skip"):
                raise SystemExit(0)
        # status should be 'skipped' (last log entry), not 'error'
        assert state.logs[-1]["status"] == "skipped"
        # No error should be recorded
        assert len(state.errors) == 0, f"errors should be empty, got {state.errors}"

    def test_systemexit_nonzero_logged_as_error(self):
        """SystemExit(code != 0) is a real error and should be logged as such."""
        state = get_state()
        state.logs.clear()
        with pytest.raises(SystemExit):
            with cell("c1", action="test-skip-error"):
                raise SystemExit(1)
        assert state.logs[-1]["status"] == "error"
        assert len(state.errors) == 1
        assert state.errors[0]["type"] == "SystemExit"


class TestPickDevice:
    def test_auto_returns_string(self):
        device = pick_device("auto")
        assert device in ("cuda:0", "cpu")

    def test_explicit_cuda0_passes_through(self):
        assert pick_device("cuda:0") == "cuda:0"

    def test_explicit_cuda1_passes_through(self):
        assert pick_device("cuda:1") == "cuda:1"

    def test_explicit_cpu_passes_through(self):
        assert pick_device("cpu") == "cpu"

    def test_falls_back_to_cpu_when_no_torch(self, monkeypatch):
        """If torch is not importable, pick_device must return 'cpu'."""
        import builtins
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # Re-pick_device shouldn't try to import torch; should fall through
        # to the "except ImportError" branch.
        # Note: the function uses `import torch` inside try; our fake_import
        # intercepts it. Should return 'cpu'.
        assert pick_device("auto") == "cpu"


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


class TestLogFile:
    """state.log() and state.error() must write to a local log file so
    the user can download it (files.download at the end of the demo)
    rather than copying errors by hand or saving the notebook to GitHub.
    """

    def test_log_file_written_on_state_log(self, tmp_path, monkeypatch):
        import os
        import json as _json
        import colab_session
        test_log = str(tmp_path / "test_log.jsonl")
        monkeypatch.setattr(colab_session, "LOG_FILE_PATH", test_log)
        colab_session.reset_state()
        state = colab_session.get_state()
        state.log("test-cell", action="test-action", key="value")
        assert os.path.exists(test_log), "log file should be created on state.log()"
        with open(test_log) as f:
            lines = f.readlines()
        assert len(lines) == 1, f"expected 1 line, got {len(lines)}"
        entry = _json.loads(lines[0])
        assert entry["cell_id"] == "test-cell"
        assert entry["action"] == "test-action"
        assert entry["key"] == "value"

    def test_log_file_written_on_state_error(self, tmp_path, monkeypatch):
        import os
        import json as _json
        import colab_session
        test_log = str(tmp_path / "test_log_err.jsonl")
        monkeypatch.setattr(colab_session, "LOG_FILE_PATH", test_log)
        colab_session.reset_state()
        state = colab_session.get_state()
        try:
            raise ValueError("test error message")
        except ValueError as e:
            state.error("test-cell-err", e, hint="test hint")
        assert os.path.exists(test_log)
        with open(test_log) as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = _json.loads(lines[0])
        assert entry["_kind"] == "error"
        assert entry["type"] == "ValueError"
        assert entry["message"] == "test error message"
        assert entry["hint"] == "test hint"
        assert "test-cell-err" in entry["cell_id"]


# --- Gemini model regression (Aug 2026) ------------------------------------

class TestGeminiModel:
    """The Gemini model name must be a current GA model. The 2.0-flash-lite
    model was retired 2026-06-01 and 2.5-flash-lite was retired 2026-07-22.
    The 3.5-flash-lite model is the current GA as of Aug 2026.
    """

    def test_coach_diagnose_default_model_is_current(self):
        """coach_diagnose() must default to a currently-available Gemini model."""
        import inspect
        import colab_session
        sig = inspect.signature(colab_session.coach_diagnose)
        default = sig.parameters["model"].default
        # The retired model that triggered the bug
        assert default != "gemini-2.0-flash-lite", (
            "gemini-2.0-flash-lite was retired 2026-06-01 — must use a current model"
        )
        assert default != "gemini-2.5-flash-lite", (
            "gemini-2.5-flash-lite was retired 2026-07-22 — must use a current model"
        )
        # Current GA (as of Aug 2026)
        assert "gemini-3" in default, (
            f"default model must be a Gemini 3.x variant (current GA), got '{default}'"
        )

    def test_coach_review_default_model_is_current(self):
        """coach_review() must default to a currently-available Gemini model."""
        import inspect
        import colab_session
        sig = inspect.signature(colab_session.coach_review)
        default = sig.parameters["model"].default
        assert default != "gemini-2.0-flash-lite"
        assert default != "gemini-2.5-flash-lite"
        assert "gemini-3" in default, (
            f"default model must be a Gemini 3.x variant, got '{default}'"
        )


# --- Val split helper (Aug 2026) -------------------------------------------


class TestEnsureValSplit:
    """_ensure_val_split is the v1.5 fix for the recycling_v3 0-image val
    problem. Tests run against a temp dir with synthetic train images —
    no real network or dataset download required.
    """

    def _make_dataset(self, tmp: Path, n_train: int = 100) -> None:
        """Create a minimal YOLO dataset layout under tmp/ds."""
        ds = tmp / "ds"
        (ds / "train" / "images").mkdir(parents=True)
        (ds / "train" / "labels").mkdir(parents=True)
        (ds / "valid" / "images").mkdir(parents=True)
        (ds / "valid" / "labels").mkdir(parents=True)
        for i in range(n_train):
            (ds / "train" / "images" / f"img_{i:03d}.jpg").write_bytes(b"\xff\xd8\xff")
            (ds / "train" / "labels" / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    def test_noop_when_val_already_has_enough_images(self, tmp_path):
        """If valid/ already has >= min_val_images, no files are moved."""
        from colab_session import _ensure_val_split

        self._make_dataset(tmp_path, n_train=100)
        # Pre-populate valid/ with 60 images
        for i in range(60):
            (tmp_path / "ds" / "valid" / "images" / f"pre_{i}.jpg").write_bytes(b"\xff")
        n_train_before = len(list((tmp_path / "ds" / "train" / "images").glob("*.jpg")))
        n = _ensure_val_split(tmp_path / "ds", val_fraction=0.1, min_val_images=50, seed=42)
        n_train_after = len(list((tmp_path / "ds" / "train" / "images").glob("*.jpg")))
        assert n == 60, "should report the pre-existing val count (60)"
        assert n_train_after == n_train_before, "no train files should be moved"

    def test_moves_fraction_when_val_is_empty(self, tmp_path):
        """If valid/ has 0 images, move val_fraction of train → valid."""
        from colab_session import _ensure_val_split

        self._make_dataset(tmp_path, n_train=200)
        n = _ensure_val_split(tmp_path / "ds", val_fraction=0.1, min_val_images=50, seed=42)
        # 10% of 200 = 20, but min_val_images=50 so 50 are moved
        assert n == 50, f"expected 50 val images, got {n}"
        assert len(list((tmp_path / "ds" / "valid" / "images").glob("*.jpg"))) == 50
        assert len(list((tmp_path / "ds" / "train" / "images").glob("*.jpg"))) == 150

    def test_moves_matching_labels(self, tmp_path):
        """Each image move must also move its matching .txt label."""
        from colab_session import _ensure_val_split

        self._make_dataset(tmp_path, n_train=100)
        _ensure_val_split(tmp_path / "ds", val_fraction=0.1, min_val_images=50, seed=42)
        # Every val image must have a matching label
        n_val_imgs = len(list((tmp_path / "ds" / "valid" / "images").glob("*.jpg")))
        n_val_lbls = len(list((tmp_path / "ds" / "valid" / "labels").glob("*.txt")))
        assert n_val_imgs == n_val_lbls, (
            f"label count {n_val_lbls} must match image count {n_val_imgs}"
        )

    def test_is_idempotent(self, tmp_path):
        """Re-running with the same args should be a no-op (val count stays stable)."""
        from colab_session import _ensure_val_split

        self._make_dataset(tmp_path, n_train=200)
        n1 = _ensure_val_split(tmp_path / "ds", val_fraction=0.1, min_val_images=50, seed=42)
        n2 = _ensure_val_split(tmp_path / "ds", val_fraction=0.1, min_val_images=50, seed=42)
        assert n1 == n2, "second run should return the same count"

    def test_handles_missing_train_dir(self, tmp_path):
        """If the dataset layout doesn't exist, return 0 gracefully."""
        from colab_session import _ensure_val_split

        # No ds/ subdir at all
        n = _ensure_val_split(tmp_path / "does_not_exist", val_fraction=0.1, min_val_images=50)
        assert n == 0
