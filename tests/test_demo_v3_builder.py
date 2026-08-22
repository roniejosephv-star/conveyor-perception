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
    # Must report the repo path (where cell 2 will clone to)
    assert "REPO" in src, "v3 cell 1 must compute + print the REPO path"


def test_v3_cell_1_does_not_import_colab_session():
    """REGRESSION GUARD for the Aug 22 2026 ModuleNotFoundError crash.

    Cell 1 runs BEFORE the clone (cell 2). The repo doesn't exist yet on a
    fresh Colab session, so any `import colab_session` here crashes. Cell 1
    must be purely an env check — SessionState init is deferred to cell 3,
    which runs after the clone.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell1 = nb["cells"][1]
    src = "".join(cell1["source"])
    # Use regex with MULTILINE so we only match ACTUAL import statements at
    # line start, not the words "import colab_session" appearing inside a
    # comment that explains why we don't import it.
    import re
    bad_import = re.search(r"^\s*import\s+colab_session\b", src, re.MULTILINE)
    bad_from = re.search(r"^\s*from\s+colab_session\b", src, re.MULTILINE)
    assert not bad_import, (
        "v3 cell 1 must NOT import colab_session — the repo isn't cloned yet. "
        "Aug 22 2026: this exact line caused ModuleNotFoundError on Colab."
    )
    assert not bad_from, (
        "v3 cell 1 must NOT `from colab_session import ...` — same reason."
    )


def test_v3_cell_1_does_not_init_sessionstate():
    """SessionState is cell 3's job. Cell 1 must not call get_state() or
    state.metric() — those would require the import we just banned.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell1 = nb["cells"][1]
    src = "".join(cell1["source"])
    assert "get_state" not in src, (
        "v3 cell 1 must NOT call get_state() — that lives in cell 3 (after the clone)."
    )
    assert "state.metric" not in src, (
        "v3 cell 1 must NOT call state.metric() — there's no state singleton yet."
    )


# ---------------------------------------------------------------------------
# Cell 2 (install + clone)
# ---------------------------------------------------------------------------

def test_v3_cell_2_is_install_and_clone():
    """Cell 2 must be the install + clone step."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 3, "v3 needs at least 3 cells (title + runtime + install/clone)"
    cell2 = nb["cells"][2]
    assert cell2["cell_type"] == "code", "v3 cell 2 must be code"
    src = "".join(cell2["source"])
    # Must do a git clone. We accept either the literal "git clone" string (in a
    # comment) OR the Python-list form ['git', 'clone', ...] used in subprocess.run.
    has_clone = "git clone" in src or re.search(r"\[\s*'git'\s*,\s*'clone'", src) is not None
    assert has_clone, "v3 cell 2 must clone the repo (literal 'git clone' or subprocess list form)"
    assert "github.com" in src, "v3 cell 2 clone must reference the GitHub URL"
    # Must install the open-source deps we need for the pipeline
    for pkg in ["ultralytics", "supervision", "roboflow"]:
        assert pkg in src, f"v3 cell 2 must install {pkg}"
    # Must use pip
    assert "pip" in src, "v3 cell 2 must use pip to install packages"


def test_v3_cell_2_clone_is_idempotent():
    """Cell 2 must skip the clone if the repo is already on disk — running the
    cell twice shouldn't re-clone and shouldn't fail.

    Regression guard: a non-idempotent clone would mean a Colab re-run crashes
    because git refuses to clone into a non-empty dir.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell2 = nb["cells"][2]
    src = "".join(cell2["source"])
    # Must check for pyproject.toml before cloning
    assert "pyproject.toml" in src, (
        "v3 cell 2 must check for pyproject.toml before cloning — this is how "
        "it knows the repo is already on disk and the clone should be skipped."
    )
    # Must have an exists() check on the repo path
    assert ".exists()" in src, "v3 cell 2 must call .exists() to make the clone idempotent"


def test_v3_cell_2_adds_repo_and_notebooks_to_sys_path():
    """After the clone, cell 2 must add both REPO and REPO/notebooks to sys.path
    so cell 3 can `from colab_session import get_state` (colab_session.py lives
    in notebooks/, not at the repo root).
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell2 = nb["cells"][2]
    src = "".join(cell2["source"])
    assert "sys.path" in src, "v3 cell 2 must mutate sys.path"
    assert "notebooks" in src, "v3 cell 2 must add REPO/notebooks to sys.path (colab_session.py lives there)"
    assert "sys.path.insert" in src, "v3 cell 2 must use sys.path.insert"


def test_v3_cell_2_verifies_colab_session_import_works():
    """After the clone + path setup, cell 2 must smoke-test that
    `from colab_session import get_state` actually works. If it doesn't, the
    user finds out HERE (cell 2) instead of crashing cell 3 with a confusing
    ModuleNotFoundError.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell2 = nb["cells"][2]
    src = "".join(cell2["source"])
    assert "from colab_session import get_state" in src, (
        "v3 cell 2 must smoke-test `from colab_session import get_state` — "
        "this catches path/clone problems before cell 3 tries to use the state."
    )


# ---------------------------------------------------------------------------
# Cell 3 (state + toggles)
# ---------------------------------------------------------------------------

def test_v3_cell_3_is_state_and_toggles():
    """Cell 3 must be the state + toggle UI step."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 4, "v3 needs at least 4 cells (title + runtime + install + state/toggles)"
    cell3 = nb["cells"][3]
    assert cell3["cell_type"] == "code", "v3 cell 3 must be code"
    src = "".join(cell3["source"])
    # Must create the state singleton
    assert "from colab_session import" in src, "v3 cell 3 must import from colab_session"
    assert "get_state" in src, "v3 cell 3 must call get_state() to create the singleton"
    # Must show the toggle UI
    assert "toggle_ui" in src, "v3 cell 3 must call toggle_ui() to build the form"
    assert "display" in src, "v3 cell 3 must display the toggle UI"


def test_v3_cell_3_handles_ipywidgets_defensively():
    """Cell 3 is the first cell that uses ipywidgets (the toggle UI depends on
    it). It must install ipywidgets if it's not already available, so the cell
    works whether or not the user re-ran cell 2 with the new INSTALL list.

    Pattern: defensive install at the boundary of any new dependency — fail
    fast in this cell, not in the middle of the toggle UI render.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell3 = nb["cells"][3]
    src = "".join(cell3["source"])
    assert "ipywidgets" in src, "v3 cell 3 must reference ipywidgets (the toggle UI's dep)"
    # Must have either a try/import for ipywidgets OR a pip install in the cell
    has_defensive_install = (
        "import ipywidgets" in src and "pip" in src
    )
    assert has_defensive_install, (
        "v3 cell 3 must defensively install ipywidgets — either via a try/import "
        "followed by pip install, or by including it in the cell directly. "
        "Don't assume cell 2's INSTALL list is current."
    )


def test_v3_cell_3_sys_path_setup_is_idempotent():
    """Cell 3 must redo the sys.path setup (in case cell 2 was skipped) — the
    toggle UI import needs REPO/notebooks on path. The setup must use the
    'already in sys.path' guard so re-runs don't duplicate entries.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell3 = nb["cells"][3]
    src = "".join(cell3["source"])
    assert "sys.path" in src, "v3 cell 3 must mutate sys.path so colab_session is importable"
    assert "notebooks" in src, "v3 cell 3 must add REPO/notebooks to sys.path"
    # The "if sp not in sys.path" guard is what makes it idempotent
    assert "not in sys.path" in src, (
        "v3 cell 3 sys.path setup must be idempotent — use `if sp not in sys.path` "
        "before sys.path.insert so re-runs don't duplicate entries."
    )


def test_v3_cell_3_groups_toggles_in_summary():
    """The current-toggles summary print must group abstractions and modules
    separately (mirroring the visual grouping in the toggle UI). This is a
    small UX detail but it makes the output scannable.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell3 = nb["cells"][3]
    src = "".join(cell3["source"])
    # Must mention both "abstraction" and "module" in some kind of grouping context
    assert "abstraction" in src.lower(), "v3 cell 3 summary must mention abstractions"
    assert "module" in src.lower(), "v3 cell 3 summary must mention modules"
    # Must iterate over state.toggles
    assert "state.toggles" in src, "v3 cell 3 must read state.toggles for the summary"
