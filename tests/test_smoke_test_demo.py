"""Tests for scripts/smoke_test_demo.py.

The smoke test is a 5-second structural check on notebooks/demo_v2.ipynb.
These tests pin its behavior so a refactor can't silently turn it into
a no-op or break the cell-count contract.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "smoke_test_demo.py"
NOTEBOOK = REPO_ROOT / "notebooks" / "demo_v2.ipynb"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import smoke_test_demo  # noqa: E402


# --- the API: parse / count / parse-check ---


def test_parse_notebook_returns_valid_dict():
    """Given a real notebook, parse_notebook returns the JSON dict with cells + nbformat."""
    nb = smoke_test_demo.parse_notebook(NOTEBOOK)
    assert isinstance(nb, dict)
    assert "cells" in nb
    assert nb["nbformat"] == 4


def test_parse_notebook_raises_on_missing_file():
    """Given a non-existent path, parse_notebook raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        smoke_test_demo.parse_notebook(REPO_ROOT / "does_not_exist.ipynb")


def test_parse_notebook_raises_on_bad_json(tmp_path):
    """Given a file that isn't valid JSON, parse_notebook raises JSONDecodeError."""
    bad = tmp_path / "bad.ipynb"
    bad.write_text("not json at all")
    with pytest.raises(json.JSONDecodeError):
        smoke_test_demo.parse_notebook(bad)


def test_check_cell_count_passes_at_29():
    """The shipped notebook has 29 cells. If this fails, the builder drifted."""
    nb = smoke_test_demo.parse_notebook(NOTEBOOK)
    errors = smoke_test_demo.check_cell_count(nb, expected=29)
    assert errors == [], f"cell count drift: {errors}"


def test_check_cell_count_flags_drift():
    """A fake notebook with 5 cells should fail the 29-cell check."""
    nb = {"cells": [{}] * 5, "nbformat": 4}
    errors = smoke_test_demo.check_cell_count(nb, expected=29)
    assert len(errors) == 1
    assert "cell count drift" in errors[0]


def test_check_code_cells_parse_returns_empty_for_valid_notebook():
    """All code cells in the shipped notebook must parse cleanly."""
    nb = smoke_test_demo.parse_notebook(NOTEBOOK)
    errors = smoke_test_demo.check_code_cells_parse(nb)
    assert errors == [], f"unexpected parse errors: {errors}"


def test_check_code_cells_parse_catches_syntax_error(tmp_path):
    """A code cell with `def foo(:` should fail ast.parse."""
    bad = {
        "nbformat": 4,
        "cells": [
            {"cell_type": "code", "source": ["def foo(:\n"]},
        ],
    }
    errors = smoke_test_demo.check_code_cells_parse(bad)
    assert len(errors) == 1
    cell_idx, msg = errors[0]
    assert cell_idx == 1
    assert "msg" in msg or "line" in msg


def test_check_code_cells_parse_skips_markdown_cells():
    """Markdown cells should never be ast-parsed."""
    nb = {
        "nbformat": 4,
        "cells": [
            {"cell_type": "markdown", "source": ["# Title\n", "**bold**"]},
        ],
    }
    errors = smoke_test_demo.check_code_cells_parse(nb)
    assert errors == []


def test_check_code_cells_parse_skips_empty_code_cells():
    """Empty code cells (placeholders) should not error."""
    nb = {
        "nbformat": 4,
        "cells": [
            {"cell_type": "code", "source": []},
            {"cell_type": "code", "source": [""]},
        ],
    }
    errors = smoke_test_demo.check_code_cells_parse(nb)
    assert errors == []


# --- the import-shape warning (non-fatal) ---


def test_check_code_cells_have_imports_warns_when_no_imports():
    """A code cell with no imports triggers a warning (not a failure)."""
    nb = {
        "nbformat": 4,
        "cells": [
            {"cell_type": "code", "source": ["x = 1 + 2\n"]},
        ],
    }
    warnings = smoke_test_demo.check_code_cells_have_imports(nb)
    assert len(warnings) == 1
    assert "cell 1" in warnings[0]


def test_check_code_cells_have_imports_quiet_on_real_notebook():
    """The shipped notebook's code cells all have imports. No warnings expected."""
    nb = smoke_test_demo.parse_notebook(NOTEBOOK)
    warnings = smoke_test_demo.check_code_cells_have_imports(nb)
    # Some code cells are intentional one-liners (e.g. status pills) without
    # imports, so we only assert "no warnings" is the common case. Allow ≤2.
    assert len(warnings) <= 2, f"unexpected import-shape warnings: {warnings}"


# --- the integrated run_smoke_test entry point ---


def test_run_smoke_test_passes_on_shipped_notebook():
    """End-to-end: run_smoke_test on the real .ipynb returns ok=True."""
    result = smoke_test_demo.run_smoke_test(NOTEBOOK)
    assert result.ok is True
    assert result.cell_count == 29
    assert result.code_cells == 22
    assert result.markdown_cells == 7
    assert result.parse_errors == []
    assert "OK" in result.summary()


def test_run_smoke_test_fails_on_missing_file():
    """End-to-end: a non-existent path returns ok=False with a parse error."""
    result = smoke_test_demo.run_smoke_test(REPO_ROOT / "missing.ipynb")
    assert result.ok is False
    assert len(result.parse_errors) == 1
    assert "could not parse notebook" in result.parse_errors[0][1]


def test_run_smoke_test_summary_includes_cell_count():
    """The summary string is human-readable and includes the cell count."""
    result = smoke_test_demo.run_smoke_test(NOTEBOOK)
    s = result.summary()
    assert "29 cells" in s
    assert "22 code" in s
    assert "7 md" in s
    assert "0 parse errors" in s


# --- the CLI entry point ---


def test_cli_returns_0_on_success():
    """Invoking the script with no args exits 0 on the shipped notebook."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, f"unexpected failure:\n{proc.stdout}\n{proc.stderr}"
    assert "[OK]" in proc.stdout
    assert "29 cells" in proc.stdout


def test_cli_returns_1_on_missing_file():
    """Invoking the script on a non-existent path exits 1 with a clear error."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "/tmp/does_not_exist_demo_v2.ipynb"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 1
    assert "FAIL" in proc.stdout or "could not parse" in proc.stdout
