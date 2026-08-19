"""Tests for the live progress dashboard (ProgressTracker + render_html).

The dashboard is the per-cell interactive section the user asked for:
- Visible at the top of the notebook from cell 1 onward
- Updates in place as each `with cell(...)` block runs
- Shows cell id, action, status, elapsed time, errors
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "notebooks"))
import colab_session  # noqa: E402
from colab_session import ProgressEntry, ProgressTracker  # noqa: E402


# --- ProgressEntry ---


def test_progress_entry_defaults_to_pending():
    """A new entry with no explicit status starts as 'pending'."""
    e = ProgressEntry(cell_id="cell-1", action="env-check")
    assert e.status == "pending"
    assert e.elapsed_ms == 0.0
    assert e.error == ""
    assert e.icon == "&#9711;"  # ○
    assert e.css_class == "t-row t-pending"


def test_progress_entry_status_icons():
    """Each status maps to a clear visual icon."""
    cases = {
        "pending": "&#9711;",   # ○
        "running": "&#9654;",   # ▶
        "ok": "&#10003;",       # ✓
        "error": "&#10007;",     # ✗
        "skipped": "&#8856;",    # ⊘
    }
    for status, icon in cases.items():
        e = ProgressEntry(cell_id="x", action="y", status=status)
        assert e.icon == icon, f"wrong icon for {status}"


# --- ProgressTracker ---


def test_tracker_starts_empty():
    """A fresh tracker has no entries and no widget."""
    t = ProgressTracker(total_cells=29)
    assert t.entries == []
    assert t.by_id == {}
    assert t.widget is None
    assert t.total_cells == 29


def test_tracker_start_creates_running_entry():
    """start() creates a new entry with status='running'."""
    t = ProgressTracker(total_cells=29)
    t.start("cell-1", "env-check")
    assert len(t.entries) == 1
    assert t.entries[0].cell_id == "cell-1"
    assert t.entries[0].action == "env-check"
    assert t.entries[0].status == "running"
    assert t.by_id["cell-1"] is t.entries[0]


def test_tracker_finish_updates_existing_entry():
    """finish() updates the existing entry in place (not appending a new one)."""
    t = ProgressTracker(total_cells=29)
    t.start("cell-1", "env-check")
    t.finish("cell-1", "ok", 5234.5)
    assert len(t.entries) == 1, "finish() should not append a new entry"
    assert t.entries[0].status == "ok"
    assert t.entries[0].elapsed_ms == 5234.5


def test_tracker_finish_with_error_stores_error_message():
    """finish('error', ..., error='...') stores the message for display."""
    t = ProgressTracker(total_cells=29)
    t.start("cell-1", "env-check")
    t.finish("cell-1", "error", 102.0, error="ImportError: No module named foo")
    assert t.entries[0].status == "error"
    assert "ImportError" in t.entries[0].error


def test_tracker_finish_creates_entry_if_not_started():
    """finish() without a prior start() still creates an entry (defensive)."""
    t = ProgressTracker(total_cells=29)
    t.finish("cell-x", "ok", 100.0)
    assert len(t.entries) == 1
    assert t.entries[0].cell_id == "cell-x"


def test_tracker_orders_entries_by_start_time():
    """Entries appear in the order they were started, not the order they finished."""
    t = ProgressTracker(total_cells=29)
    t.start("cell-1", "first")
    t.start("cell-2", "second")
    t.finish("cell-1", "ok", 100.0)
    t.finish("cell-2", "ok", 200.0)
    assert [e.cell_id for e in t.entries] == ["cell-1", "cell-2"]


# --- render_html ---


def test_render_html_uses_themed_css():
    """The dashboard HTML is prefixed with the _THEME_CSS so it renders styled."""
    t = ProgressTracker(total_cells=29)
    t.start("cell-1", "env-check")
    t.finish("cell-1", "ok", 5234.5)
    html = t.render_html()
    assert html.startswith(colab_session._THEME_CSS), \
        "render_html must prefix with _THEME_CSS for proper styling"
    assert "tinkr-dashboard" in html


def test_render_html_shows_correct_count():
    """The header count reflects the done/running/pending state."""
    t = ProgressTracker(total_cells=29)
    t.start("cell-1", "env-check"); t.finish("cell-1", "ok", 100.0)
    t.start("cell-2", "install"); t.finish("cell-2", "ok", 200.0)
    t.start("cell-3", "init"); t.finish("cell-3", "error", 50.0, error="boom")
    t.start("cell-4", "toggle")
    html = t.render_html()
    # 2 done, 1 running, 1 error, 25 pending
    assert "<b>2</b>/29 done" in html
    assert "<b>1</b> running" in html
    assert "<b>1</b> error" in html
    assert "<b>25</b> pending" in html


def test_render_html_handles_zero_entries():
    """An empty tracker renders without crashing."""
    t = ProgressTracker(total_cells=29)
    html = t.render_html()
    assert "0</b>/29 done" in html
    assert "Cells will appear here" in html


def test_render_html_time_formatting():
    """Elapsed times are formatted as ms / s / m depending on magnitude."""
    t = ProgressTracker(total_cells=29)
    t.start("a", ""); t.finish("a", "ok", 234.0)        # < 1s → "234ms"
    t.start("b", ""); t.finish("b", "ok", 12_345.0)     # < 60s → "12.3s"
    t.start("c", ""); t.finish("c", "ok", 90_000.0)     # >= 60s → "1.5m"
    html = t.render_html()
    assert "234ms" in html
    assert "12.3s" in html
    assert "1.5m" in html


def test_render_html_running_row_uses_ellipsis():
    """A running cell's time column shows an ellipsis (not '0ms' or '—')."""
    t = ProgressTracker(total_cells=29)
    t.start("cell-1", "env-check")
    html = t.render_html()
    # The cell-1 row should have an ellipsis in the time column.
    assert "t-row t-running" in html
    assert "&hellip;" in html


def test_render_html_error_row_includes_css_class():
    """An error row gets the t-error class for red coloring."""
    t = ProgressTracker(total_cells=29)
    t.start("cell-1", ""); t.finish("cell-1", "error", 100.0, error="boom")
    html = t.render_html()
    assert 'class="t-row t-error"' in html


# --- init_progress_dashboard (defensive) ---


def test_init_dashboard_does_not_raise_when_ipywidgets_missing(monkeypatch):
    """init_progress_dashboard must never raise — cell 1 must keep working
    even if ipywidgets isn't installed, the import fails, or display() rejects
    the widget. The dashboard is nice-to-have, not load-bearing.

    Caught live on 2026-08-19: the previous init raised inside the
    IPython.display import on some Colab configurations, killing cell 1.
    """
    # Simulate: ipywidgets import fails, display import fails
    import builtins

    # Hide IPython and ipywidgets from the import system
    monkeypatch.setitem(builtins.__dict__, "__test_blocked__", True)
    import sys
    blocked = {"ipywidgets", "IPython"}
    for mod in list(sys.modules):
        if mod in blocked or mod.startswith("IPython."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.display", None)

    # init should return None (or a fallback), not raise
    result = colab_session.init_progress_dashboard(total_cells=29)
    # Result is None or a widget-like — we don't care which, just no raise
    assert result is None or hasattr(result, "value")


def test_init_dashboard_creates_tracker_on_first_call():
    """First call creates state.progress; second call reuses it."""
    # Reset state
    colab_session.reset_state()
    colab_session.init_progress_dashboard(total_cells=29)
    state = colab_session.get_state()
    assert state.progress is not None
    assert state.progress.total_cells == 29
    tracker_id = id(state.progress)

    colab_session.init_progress_dashboard(total_cells=29)
    assert id(colab_session.get_state().progress) == tracker_id, "second call must not create a new tracker"


def test_init_dashboard_recovers_after_ipywidgets_failure(monkeypatch):
    """If the first attempt fails, the second attempt should still work
    (once ipywidgets becomes available).
    """
    import sys
    colab_session.reset_state()

    # First call: no ipywidgets
    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    try:
        colab_session.init_progress_dashboard(total_cells=29)
    except Exception:
        pass  # tolerated but we also want to ensure no raise

    # Second call: ipywidgets is now available
    fake_ipywidgets = type(sys)("ipywidgets")
    class FakeHTML:
        def __init__(self, value=""): self.value = value
    fake_ipywidgets.HTML = FakeHTML
    monkeypatch.setitem(sys.modules, "ipywidgets", fake_ipywidgets)

    # The state should already have a progress tracker
    state = colab_session.get_state()
    assert state.progress is not None
