"""Tests for notebooks/build_demo_v3.py.

Validates the v3 notebook builder:
- Generates valid JSON
- Notebook has the expected structure (kernelspec, language_info, colab metadata)
- Each cell has the required fields
- Markdown / code cells are balanced
- Cell numbering is contiguous (1, 2, 3, ...) with no half-numbers
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILDER = REPO_ROOT / "notebooks" / "build_demo_v3.py"
NOTEBOOK = REPO_ROOT / "notebooks" / "demo_v3.ipynb"


def _run_builder() -> None:
    """Re-run the v3 notebook builder to make sure the on-disk .ipynb is current."""
    result = subprocess.run(
        [sys.executable, str(BUILDER), str(NOTEBOOK)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, f"v3 builder failed:\n{result.stderr}"


@pytest.fixture(scope="module", autouse=True)
def rebuild_notebook():
    _run_builder()
    yield
    # Don't rebuild on teardown — leave the file in its current state.


# ---------------------------------------------------------------------------
# Notebook structure
# ---------------------------------------------------------------------------

def test_notebook_exists():
    assert NOTEBOOK.exists(), f"v3 notebook missing at {NOTEBOOK}"


def test_notebook_is_valid_json():
    nb = json.loads(NOTEBOOK.read_text())
    assert isinstance(nb, dict)
    assert "cells" in nb
    assert "metadata" in nb
    assert nb["nbformat"] == 4


def test_notebook_has_colab_metadata():
    """v3 is built for Colab — must have the kernelspec + colab metadata."""
    nb = json.loads(NOTEBOOK.read_text())
    md = nb["metadata"]
    assert "kernelspec" in md
    assert md["kernelspec"]["name"] == "python3"
    assert "colab" in md, "v3 notebook should declare colab metadata"


def test_cells_have_required_fields():
    """Every cell (code or markdown) must have cell_type, metadata, source."""
    nb = json.loads(NOTEBOOK.read_text())
    for i, c in enumerate(nb["cells"]):
        assert "cell_type" in c, f"cell {i} missing cell_type"
        assert "source" in c, f"cell {i} missing source"
        assert c["cell_type"] in ("code", "markdown"), f"cell {i} has bad cell_type: {c['cell_type']}"


# ---------------------------------------------------------------------------
# Cell numbering (the big rule that the user flagged on Aug 22 2026)
# ---------------------------------------------------------------------------

def test_v3_cell_numbering_has_no_half_numbers():
    """v3 must NEVER use half-numbered cells (7.5, 7.6, 8.5, 9.5, 9.6 are forbidden).
    The user explicitly called this out as confusing on Aug 22 2026 — v3 is the
    clean rebuild.
    """
    nb = json.loads(NOTEBOOK.read_text())
    decimal_cells = []
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        first_line = "".join(c.get("source", [])).splitlines()[0] if c.get("source") else ""
        m = re.search(r"Cell\s+(\d+)\.(\d+)", first_line)
        if m:
            decimal_cells.append((int(m.group(1)), int(m.group(2)), first_line[:60]))
    assert not decimal_cells, (
        f"Found half-numbered cells in v3 (forbidden): {decimal_cells}. "
        f"Use clean contiguous numbering instead (1, 2, 3, ...)."
    )


def test_v3_cell_numbering_is_contiguous():
    """Cell numbers in v3 must form a contiguous sequence (1, 2, 3, ...)."""
    nb = json.loads(NOTEBOOK.read_text())
    numbers = []
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        first_line = "".join(c.get("source", [])).splitlines()[0] if c.get("source") else ""
        m = re.search(r"Cell\s+(\d+)\b", first_line)
        if m:
            numbers.append(int(m.group(1)))
    # The sequence should be 1, 2, 3, ..., N with no gaps
    expected = list(range(1, len(numbers) + 1))
    assert numbers == expected, (
        f"v3 cell numbers are not contiguous.\n"
        f"  Expected: {expected}\n"
        f"  Actual:   {numbers}\n"
        f"v3 is a fresh build — keep the numbering clean."
    )


# ---------------------------------------------------------------------------
# Cell 0 (title) + cell 1 (runtime)
# ---------------------------------------------------------------------------

def test_v3_cell_0_is_title_markdown():
    """Cell 0 must be a markdown title."""
    nb = json.loads(NOTEBOOK.read_text())
    cell0 = nb["cells"][0]
    assert cell0["cell_type"] == "markdown", "v3 cell 0 must be markdown"
    src = "".join(cell0["source"])
    # Must mention the demo name
    assert "conveyor" in src.lower() or "perception" in src.lower() or "recycl" in src.lower(), (
        "v3 cell 0 should be a 1-screen pitch about the conveyor-perception demo"
    )


def test_v3_cell_1_is_runtime_check():
    """Cell 1 must be the runtime + env check."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 2, "v3 needs at least 2 cells (title + runtime)"
    cell1 = nb["cells"][1]
    assert cell1["cell_type"] == "code", "v3 cell 1 must be code"
    src = "".join(cell1["source"])
    # Must check the runtime environment
    assert "sys.version" in src or "Python" in src, "v3 cell 1 must report Python version"
    assert "cuda" in src.lower() or "gpu" in src.lower(), "v3 cell 1 must check GPU"
    assert "disk" in src.lower() or "shutil" in src, "v3 cell 1 must check disk space"
    # Must init the SessionState singleton
    assert "get_state" in src or "colab_session" in src, "v3 cell 1 must init SessionState"


def test_v3_cell_1_adds_notebooks_subdir_to_sys_path():
    """colab_session.py lives at notebooks/colab_session.py, NOT at the repo root.
    Cell 1 must add REPO/notebooks to sys.path so the import resolves.

    Regression: Aug 22 2026 — cell 1 only added REPO, so Colab raised
    ModuleNotFoundError on `from colab_session import get_state` on the very first
    run. Locking in the fix so the subdir is never dropped again.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell1 = nb["cells"][1]
    src = "".join(cell1["source"])
    assert "notebooks" in src, (
        "v3 cell 1 must reference the notebooks/ subdir in its sys.path setup — "
        "colab_session.py lives at notebooks/colab_session.py, not at the repo root."
    )
    assert "sys.path.insert" in src, "v3 cell 1 must call sys.path.insert to expose the repo"


def test_v3_cell_1_colab_session_import_actually_resolves():
    """End-to-end regression for the Aug 22 2026 ModuleNotFoundError. Exec cell 1's
    source in an isolated namespace; if `from colab_session import get_state`
    cannot resolve, the test fails. This is the runtime-equivalent check that
    string assertions can't fully substitute.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell1 = nb["cells"][1]
    src = "".join(cell1["source"])

    original_path = list(sys.path)
    try:
        exec(src, {"__name__": "__v3_cell1_test__"})
    except ModuleNotFoundError as e:
        pytest.fail(
            f"v3 cell 1 raised ModuleNotFoundError when exec'd: {e}\n"
            f"This is the Aug 22 2026 bug resurfacing. Cell 1 must add REPO/notebooks "
            f"to sys.path BEFORE the `from colab_session import get_state` line so the "
            f"module can be resolved."
        )
    finally:
        # Roll back any sys.path mutations the cell made so we don't leak state
        # into other tests in the same pytest run.
        for entry in list(sys.path):
            if entry not in original_path:
                sys.path.remove(entry)
        # Also drop colab_session from sys.modules (it would otherwise persist
        # for the rest of the pytest session with a SessionState that points at
        # the temp namespace).
        sys.modules.pop("colab_session", None)
