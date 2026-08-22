"""Tests for notebooks/build_demo.py.

Validates the final notebook builder:
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
BUILDER = REPO_ROOT / "notebooks" / "build_demo.py"
NOTEBOOK = REPO_ROOT / "notebooks" / "demo.ipynb"


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
def test_v3_cell_2_installs_trackers_for_bytetrack():
    """REGRESSION GUARD for the Aug 22 2026 ByteTrack fallback warning.

    Without `trackers>=2.6.0` in the INSTALL list, TrackingPipeline falls back
    to simple IoU tracking at runtime. The warning is annoying and the IoU
    tracker is less robust to occlusion. Keep trackers in the install list
    so fresh runs get ByteTrack from the start.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell2 = nb["cells"][2]
    src = "".join(cell2["source"])
    assert "trackers" in src, (
        "v3 cell 2 must install `trackers` (>=2.6.0) — the ByteTrack backend "
        "for TrackingPipeline. Without it, cell 10 falls back to IoU tracking."
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


# ---------------------------------------------------------------------------
# Cell 4 (load abstractions)
# ---------------------------------------------------------------------------

def test_v3_cell_4_is_load_abstractions():
    """Cell 4 must be the load-abstractions step."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 5, "v3 needs at least 5 cells (title + runtime + install + state + abstractions)"
    cell4 = nb["cells"][4]
    assert cell4["cell_type"] == "code", "v3 cell 4 must be code"
    src = "".join(cell4["source"])
    # Must import the 4 abstraction classes
    for cls in [
        "DetectionPipeline",
        "TrackingPipeline",
        "DriftMonitor",
        "MCPTriageSurface",
    ]:
        assert cls in src, f"v3 cell 4 must import {cls}"
    # Must read state.toggles to respect the toggle UI
    assert "state.toggles" in src, "v3 cell 4 must check state.toggles to respect the toggle UI"


def test_v3_cell_4_adds_src_to_sys_path():
    """conveyor_perception lives at src/conveyor_perception/. Cell 4 must add
    src/ to sys.path so the `from conveyor_perception.core.*` imports resolve.
    Without this, every import in cell 4 fails with ModuleNotFoundError.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell4 = nb["cells"][4]
    src = "".join(cell4["source"])
    assert "sys.path" in src, "v3 cell 4 must mutate sys.path"
    assert "src" in src.lower(), "v3 cell 4 must add the src/ subdir to sys.path (where the package lives)"


def test_v3_cell_4_respects_all_4_abstraction_toggles():
    """Each of the 4 abstractions has a toggle in state.toggles. Cell 4 must
    check each one (so an unticked abstraction is skipped, not just
    silently loaded). The 4 toggle keys:
      - abstraction:detector
      - abstraction:tracker
      - abstraction:triage
      - abstraction:drift_monitor
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell4 = nb["cells"][4]
    src = "".join(cell4["source"])
    for toggle_key in [
        "abstraction:detector",
        "abstraction:tracker",
        "abstraction:triage",
        "abstraction:drift_monitor",
    ]:
        assert toggle_key in src, (
            f"v3 cell 4 must check the `{toggle_key}` toggle from state.toggles "
            f"so the user can disable individual abstractions."
        )


def test_v3_cell_4_logs_to_state():
    """Cell 4 must call state.log() to record what was loaded. This is how
    the coach/summary cell at the end of the notebook knows which components
    ran and which were skipped.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell4 = nb["cells"][4]
    src = "".join(cell4["source"])
    assert "state.log" in src, "v3 cell 4 must call state.log() to record load results"


# ---------------------------------------------------------------------------
# Cell 5 (load modules)
# ---------------------------------------------------------------------------

def test_v3_cell_5_is_load_modules():
    """Cell 5 must be the load-modules step (the 8 JD modules)."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 6, "v3 needs at least 6 cells (title + runtime + install + state + abstractions + modules)"
    cell5 = nb["cells"][5]
    assert cell5["cell_type"] == "code", "v3 cell 5 must be code"
    src = "".join(cell5["source"])
    # Must reference all 8 module import paths
    for mod_path in [
        "conveyor_perception.perception",
        "conveyor_perception.triage",
        "conveyor_perception.predictive_maintenance",
        "conveyor_perception.multitask",
        "conveyor_perception.integration",
        "conveyor_perception.robustness",
        "conveyor_perception.monitoring",
        "conveyor_perception.optimization",
    ]:
        assert mod_path in src, f"v3 cell 5 must reference module path `{mod_path}`"


def test_v3_cell_5_uses_importlib_for_dynamic_import():
    """The 8 modules are loaded by path (not by static import). Cell 5 must use
    importlib.import_module so the toggle UI can skip individual modules at
    runtime without having to refactor the cell.

    This is the right pattern: static `from x import y` would require an
    `if/else` for each module; dynamic importlib is one loop over a metadata
    table.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell5 = nb["cells"][5]
    src = "".join(cell5["source"])
    assert "importlib" in src, "v3 cell 5 must import importlib"
    assert "importlib.import_module" in src, "v3 cell 5 must call importlib.import_module() for dynamic loading"


def test_v3_cell_5_respects_all_8_module_toggles():
    """Each of the 8 modules has a toggle in state.toggles. Cell 5 must check
    each one so an unticked module is skipped, not silently loaded.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell5 = nb["cells"][5]
    src = "".join(cell5["source"])
    for toggle_key in [
        "module:perception",
        "module:triage",
        "module:predictive_maintenance",
        "module:multitask",
        "module:integration",
        "module:robustness",
        "module:monitoring",
        "module:optimization",
    ]:
        assert toggle_key in src, (
            f"v3 cell 5 must check the `{toggle_key}` toggle from state.toggles."
        )


def test_v3_cell_5_handles_per_module_import_failures():
    """One bad module must not crash the whole cell. Cell 5 must wrap each
    importlib.import_module() in a try/except so a single failure is reported
    but the other modules still load.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell5 = nb["cells"][5]
    src = "".join(cell5["source"])
    # Must have a try/except around importlib.import_module
    has_try = "try:" in src
    has_except = "except" in src
    assert has_try and has_except, (
        "v3 cell 5 must wrap each importlib.import_module() in try/except so "
        "one bad module doesn't kill the whole cell."
    )


def test_v3_cell_5_logs_to_state():
    """Cell 5 must call state.log() to record what was loaded / skipped / failed.
    The summary cell at the end reads this to know which components ran.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell5 = nb["cells"][5]
    src = "".join(cell5["source"])
    assert "state.log" in src, "v3 cell 5 must call state.log() to record load results"


# ---------------------------------------------------------------------------
# Cell 6 (data registry)
# ---------------------------------------------------------------------------

def test_v3_cell_6_is_data_registry():
    """Cell 6 must be the data-registry scan step."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 7, "v3 needs at least 7 cells (title + runtime + install + state + abstractions + modules + data registry)"
    cell6 = nb["cells"][6]
    assert cell6["cell_type"] == "code", "v3 cell 6 must be code"
    src = "".join(cell6["source"])
    # Must scan the two data roots. Use the Path-syntax form because the cell
    # uses `REPO / 'data' / 'sample'` rather than the string "data/sample".
    assert re.search(r"['\"]data['\"]\s*/\s*['\"]sample['\"]", src), (
        "v3 cell 6 must scan data/sample/ (built via REPO / 'data' / 'sample')."
    )
    assert re.search(r"['\"]data['\"]\s*/\s*['\"]raw['\"]", src), (
        "v3 cell 6 must scan data/raw/ (built via REPO / 'data' / 'raw')."
    )
    # Must look for data.yaml (the YOLO dataset marker)
    assert "data.yaml" in src, "v3 cell 6 must look for data.yaml (YOLO dataset marker)"


def test_v3_cell_6_handles_yaml_defensively():
    """data.yaml is YAML, so cell 6 needs PyYAML. Defensively install it (same
    pattern as cell 3's ipywidgets) so the cell works without a re-run of cell 2.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell6 = nb["cells"][6]
    src = "".join(cell6["source"])
    # Must import yaml (or have a try/import + pip install for it)
    assert "yaml" in src, "v3 cell 6 must import yaml (the data.yaml parser)"
    # Must have a try/except OR include yaml in the cell's install
    has_defensive = "import yaml" in src and "pip" in src
    assert has_defensive, (
        "v3 cell 6 must defensively install PyYAML — try/import + pip install, "
        "so the cell works even if cell 2's INSTALL list doesn't include it."
    )


def test_v3_cell_6_caches_registry_to_state():
    """Cell 6 caches the registry on state.dataset_registry so downstream cells
    (cell 8 training, cell 9 compare) can read it without re-scanning. The
    summary cell at the end also reads it.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell6 = nb["cells"][6]
    src = "".join(cell6["source"])
    assert "state.dataset_registry" in src, (
        "v3 cell 6 must cache the registry on state.dataset_registry so "
        "downstream cells (training, compare, summary) can read it."
    )
    assert "state.log" in src, "v3 cell 6 must call state.log() to record the scan"


# ---------------------------------------------------------------------------
# Cell 7 (data download)
# ---------------------------------------------------------------------------

def test_v3_cell_7_is_data_download():
    """Cell 7 must be the data-download step (idempotent Roboflow pull)."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 8, "v3 needs at least 8 cells (...+ data download)"
    cell7 = nb["cells"][7]
    assert cell7["cell_type"] == "code", "v3 cell 7 must be code"
    src = "".join(cell7["source"])
    # Must reference the Roboflow SDK
    assert "roboflow" in src.lower(), "v3 cell 7 must use the Roboflow SDK"
    assert "Roboflow" in src, "v3 cell 7 must import the Roboflow class"
    # Must have a target selection
    assert "TARGET_NAME" in src, "v3 cell 7 must have a TARGET_NAME for the dataset to download"
    # Must reference the data/raw dir
    assert "data/raw" in src or "DATA_RAW" in src, "v3 cell 7 must target data/raw/ for downloads"


def test_v3_cell_7_is_idempotent():
    """Re-running cell 7 must NOT re-download if the dataset is already on disk.
    The idempotency check is: does data.yaml exist in the target dir?
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell7 = nb["cells"][7]
    src = "".join(cell7["source"])
    # Must check data.yaml existence before downloading
    assert "data.yaml" in src, "v3 cell 7 must check for data.yaml (idempotency guard)"
    assert ".exists()" in src, "v3 cell 7 must use .exists() to make the download idempotent"


def test_v3_cell_7_handles_download_failure():
    """Network failures, missing API key, missing deps — all should be caught.
    The cell must wrap the download in try/except and provide a manual fallback.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell7 = nb["cells"][7]
    src = "".join(cell7["source"])
    # Must have try/except around the download
    assert "try:" in src and "except" in src, (
        "v3 cell 7 must wrap the download in try/except so a network failure "
        "doesn't crash the cell."
    )
    # Must mention a manual fallback
    assert "Manual" in src or "manual" in src, (
        "v3 cell 7 must provide a manual-download fallback so the user can "
        "recover from a network failure without re-running the notebook."
    )


def test_v3_cell_7_refreshes_registry():
    """After a successful download (or even a skip), cell 7 must refresh
    state.dataset_registry so cell 8 (training) sees the new/updated dataset
    without re-scanning data/ itself.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell7 = nb["cells"][7]
    src = "".join(cell7["source"])
    assert "state.dataset_registry" in src, (
        "v3 cell 7 must refresh state.dataset_registry so cell 8 sees the "
        "post-download state without re-scanning."
    )


def test_v3_cell_7_ensures_val_split():
    """REGRESSION GUARD for the v1.5 '0 val images' fix.

    The recycling_v3 Roboflow download gives valid/ = 0 images, which makes
    mAP50 = 1.0 (vacuously). The v1.5 fix calls _ensure_val_split to move
    10% of train/ → valid/ so the mAP number is computed on real held-out
    images. Without this call, the headline mAP metric is meaningless.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell7 = nb["cells"][7]
    src = "".join(cell7["source"])
    assert "_ensure_val_split" in src, (
        "v3 cell 7 must call _ensure_val_split() to ensure a real val split. "
        "Without this, mAP50 is computed on 0 images and the headline number "
        "is meaningless."
    )
    # Must target the right datasets
    assert "recycling_demo" in src and "recycling_v3" in src, (
        "v3 cell 7 must call _ensure_val_split for both recycling_demo and "
        "recycling_v3 (the two YOLO datasets in the registry)."
    )
    assert "state.log" in src, "v3 cell 7 must call state.log() to record the download outcome"


# ---------------------------------------------------------------------------
# Cell 8 (train)
# ---------------------------------------------------------------------------

def test_v3_cell_8_is_train():
    """Cell 8 must be the train step (YOLO26s on a selected dataset, cached)."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 9, "v3 needs at least 9 cells (...+ train)"
    cell8 = nb["cells"][8]
    assert cell8["cell_type"] == "code", "v3 cell 8 must be code"
    src = "".join(cell8["source"])
    # Must use Ultralytics
    assert "ultralytics" in src.lower() or "YOLO" in src, "v3 cell 8 must use Ultralytics YOLO"
    # Must reference the yolo26s model
    assert "yolo26s" in src, "v3 cell 8 must use yolo26s (the model from the title)"
    # Must call model.train()
    assert "model.train" in src, "v3 cell 8 must call model.train()"


def test_v3_cell_8_is_cached_on_rerun():
    """Re-running cell 8 must NOT re-train if best.pt + results.csv already exist.
    The cache check is: both files present in models/{dataset_name}/weights/.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell8 = nb["cells"][8]
    src = "".join(cell8["source"])
    # Must check for best.pt
    assert "best.pt" in src, "v3 cell 8 must check for best.pt (cache marker)"
    # Must check for results.csv
    assert "results.csv" in src, "v3 cell 8 must check for results.csv (training metrics log)"
    # Must use .exists() to make the check work
    assert ".exists()" in src, "v3 cell 8 must use .exists() for the cache check"


def test_v3_cell_8_respects_module_perception_toggle():
    """If the user unticked module:perception in cell 3, cell 8 must NOT train.
    Skipping is graceful (print a message, log to state, continue) — not a crash.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell8 = nb["cells"][8]
    src = "".join(cell8["source"])
    # Must check the toggle
    assert "module:perception" in src, (
        "v3 cell 8 must check the module:perception toggle — if off, skip training "
        "rather than spending 1-15 min on a model the user doesn't want."
    )
    # Must have a skip path (not just raise)
    has_skip = "skipped" in src.lower() or "skip" in src.lower()
    assert has_skip, "v3 cell 8 must have a skip path (don't crash on toggle-off)"


def test_v3_cell_8_sets_active_model_state():
    """After a successful train (or cache hit), cell 8 must set
    state.active_model_path and state.active_dataset so downstream cells
    (cell 9 compare, cell 10 pipeline) know which model to use.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell8 = nb["cells"][8]
    src = "".join(cell8["source"])
    assert "state.active_model_path" in src, (
        "v3 cell 8 must set state.active_model_path so downstream cells "
        "know which model to use."
    )
    assert "state.active_dataset" in src, (
        "v3 cell 8 must set state.active_dataset so downstream cells know "
        "which dataset the model was trained on."
    )
    assert "state.log" in src, "v3 cell 8 must call state.log() to record the training outcome"


def test_v3_cell_8_patience_is_low_enough_to_fire():
    """REGRESSION GUARD for the Aug 22 2026 'training runs all epochs even
    when the model plateaued' issue.

    Ultralytics' default patience is 50 (and even the v3 v2's patience=15
    never fires on a 8-epoch run because the epoch cutoff happens first).
    For our 8-epoch demo on recycling_demo, patience must be small enough
    that early stopping can actually trigger — typically 3-5 epochs.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell8 = nb["cells"][8]
    src = "".join(cell8["source"])
    # Find the patience= line. Accept either a literal int (3, 5) or a
    # small computed expression.
    m = re.search(r"patience\s*=\s*([^,\n]+)", src)
    assert m, "v3 cell 8 must call model.train() with a patience= argument"
    pat = m.group(1).strip()
    # Strip any trailing comment for the comparison
    pat_value = pat.split("#")[0].strip()
    # Must be a small number — not 15, 50, or 'None'
    assert pat_value not in ("50", "15", "None", "100"), (
        f"v3 cell 8 patience={pat} is too high — early stopping won't fire "
        f"on a 8-epoch run. Use patience=3 (or similar) so plateaued runs "
        f"actually stop early."
    )


# ---------------------------------------------------------------------------
# Cell 9 (compare)
# ---------------------------------------------------------------------------

def test_v3_cell_9_is_compare():
    """Cell 9 must be the compare step (side-by-side metrics across trained models)."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 10, "v3 needs at least 10 cells (...+ compare)"
    cell9 = nb["cells"][9]
    assert cell9["cell_type"] == "code", "v3 cell 9 must be code"
    src = "".join(cell9["source"])
    # Must scan the models/ directory
    assert "models" in src, "v3 cell 9 must scan the models/ directory"
    # Must read results.csv (the Ultralytics training log)
    assert "results.csv" in src, "v3 cell 9 must read results.csv for final-epoch metrics"
    # Must compare mAP50
    assert "mAP50" in src, "v3 cell 9 must display mAP50 in the comparison"


def test_v3_cell_9_highlights_best_model():
    """When 2+ models are present, cell 9 must highlight the one with the best
    mAP50 and promote it to state.active_model_path so downstream cells use it.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell9 = nb["cells"][9]
    src = "".join(cell9["source"])
    # Must call max() to find the best
    assert "max(" in src, "v3 cell 9 must use max() to find the best mAP50"
    # Must promote the best to active_model_path
    assert "state.active_model_path" in src, (
        "v3 cell 9 must promote the best model to state.active_model_path "
        "so downstream cells (pipeline, triage) use it."
    )


def test_v3_cell_9_handles_no_models_gracefully():
    """If no models exist (cell 8 never ran), cell 9 must print a clean
    message and log to state — not crash.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell9 = nb["cells"][9]
    src = "".join(cell9["source"])
    # Must check for the rows being empty
    assert "No trained models" in src or "not rows" in src, (
        "v3 cell 9 must handle the empty-models case with a clear message."
    )


def test_v3_cell_9_logs_to_state():
    """Cell 9 must call state.log() to record the comparison result for the
    summary cell at the end of the notebook.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell9 = nb["cells"][9]
    src = "".join(cell9["source"])
    assert "state.log" in src, "v3 cell 9 must call state.log() to record the comparison"


# ---------------------------------------------------------------------------
# Cell 10 (pipeline)
# ---------------------------------------------------------------------------

def test_v3_cell_10_is_pipeline():
    """Cell 10 must assemble the Detector→Tracker→Drift→Triage pipeline."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 11, "v3 needs at least 11 cells (...+ pipeline)"
    cell10 = nb["cells"][10]
    assert cell10["cell_type"] == "code", "v3 cell 10 must be code"
    src = "".join(cell10["source"])
    # Must use the MultitaskPipeline
    assert "MultitaskPipeline" in src, "v3 cell 10 must use the MultitaskPipeline"
    # Must import the 5 core components
    for cls in [
        "UltralyticsDetector",
        "TrackingPipeline",
        "DriftMonitor",
        "L1TriageAgent",
        "MaintenanceAdvisor",
    ]:
        assert cls in src, f"v3 cell 10 must reference {cls}"


def test_v3_cell_10_uses_active_model_from_state():
    """Cell 10 must read state.active_model_path (set by cell 9's compare) —
    not hardcode a model path. This is the contract between cell 9 and cell 10.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell10 = nb["cells"][10]
    src = "".join(cell10["source"])
    assert "state.active_model_path" in src, (
        "v3 cell 10 must read state.active_model_path (set by cell 9) so the "
        "best model from compare flows into the pipeline. No hardcoded paths."
    )


def test_v3_cell_10_respects_module_multitask_toggle():
    """If the user unticked module:multitask in cell 3, cell 10 must skip the
    pipeline assembly. Skipping is graceful (print a message, log to state).
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell10 = nb["cells"][10]
    src = "".join(cell10["source"])
    assert "module:multitask" in src, (
        "v3 cell 10 must check the module:multitask toggle — if off, skip the "
        "pipeline assembly rather than running 5 components the user disabled."
    )


def test_v3_cell_10_runs_multiple_frames():
    """Drift signals need multiple frames to populate. Cell 10 must run
    more than one frame through the pipeline so drift has something to measure.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell10 = nb["cells"][10]
    src = "".join(cell10["source"])
    # Must have a loop over N_FRAMES
    assert "for _i in range(" in src or "for i in range(" in src, (
        "v3 cell 10 must loop over multiple frames to populate drift signals. "
        "A single-frame pipeline run gives drift nothing to measure."
    )


def test_v3_cell_10_logs_to_state():
    """Cell 10 must call state.log() to record the pipeline run — including
    timing + detection counts. The summary cell at the end reads this.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell10 = nb["cells"][10]
    src = "".join(cell10["source"])
    assert "state.log" in src, "v3 cell 10 must call state.log() to record the pipeline run"
    # Must record timing metric for the T4 perf pitch
    assert "t4_inference_ms" in src or "ms_per_frame" in src, (
        "v3 cell 10 must record inference timing — the T4 ms/frame number is "
        "the demo's headline perf metric."
    )


# ---------------------------------------------------------------------------
# Cell 11 (visual analytics)
# ---------------------------------------------------------------------------

def test_v3_cell_11_is_visual_analytics():
    """Cell 11 must be the visual analytics step (supervision annotators)."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 12, "v3 needs at least 12 cells (...+ visual analytics)"
    cell11 = nb["cells"][11]
    assert cell11["cell_type"] == "code", "v3 cell 11 must be code"
    src = "".join(cell11["source"])
    # Must import supervision
    assert "import supervision" in src or "from supervision" in src, (
        "v3 cell 11 must import supervision for the annotator layer."
    )
    # Must use the modern annotators
    assert "RoundBoxAnnotator" in src, "v3 cell 11 must use RoundBoxAnnotator"
    assert "RichLabelAnnotator" in src, "v3 cell 11 must use RichLabelAnnotator"
    assert "HeatMapAnnotator" in src, "v3 cell 11 must use HeatMapAnnotator"
    assert "PolygonZone" in src, "v3 cell 11 must use PolygonZone for spatial regions"
    assert "LineZone" in src, "v3 cell 11 must use LineZone for throughput counter"


def test_v3_cell_11_uses_supervision_030_api():
    """REGRESSION GUARD for the supervision 0.30.0 vs 0.31+ API mismatch.

    The cell must use the 0.30.0 kwargs (roundness=, opacity=, not border_radius=
    or alpha=). Earlier commits used 0.31+ kwargs and crashed at runtime.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell11 = nb["cells"][11]
    src = "".join(cell11["source"])
    # Must use roundness= (0.30.0), NOT border_radius= (0.31+) on RoundBoxAnnotator
    assert re.search(r"RoundBoxAnnotator\s*\([^)]*roundness\s*=", src), (
        "v3 cell 11 must call RoundBoxAnnotator(roundness=...) — the supervision "
        "0.30.0 API. Using border_radius= is the 0.31+ API and will crash."
    )
    # Must use opacity= (0.30.0), NOT alpha= (0.31+) on HeatMapAnnotator
    assert re.search(r"HeatMapAnnotator\s*\([^)]*opacity\s*=", src), (
        "v3 cell 11 must call HeatMapAnnotator(opacity=...) — the supervision "
        "0.30.0 API. Using alpha= is the 0.31+ API and will crash."
    )


def test_v3_cell_11_uses_active_model():
    """Cell 11 must read state.active_model_path (set by cell 9) — not hardcode.
    Same contract as cell 10.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell11 = nb["cells"][11]
    src = "".join(cell11["source"])
    assert "state.active_model_path" in src, (
        "v3 cell 11 must read state.active_model_path (set by cell 9) so the "
        "best model from compare flows into the visual layer."
    )


def test_v3_cell_11_saves_annotated_image():
    """Cell 11 must save the annotated image to disk so the summary cell at
    the end of the notebook can pick it up without re-running inference.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell11 = nb["cells"][11]
    src = "".join(cell11["source"])
    # Must save the image (cv2.imwrite or similar) + reference the path
    assert "imwrite" in src or ".save" in src, (
        "v3 cell 11 must save the annotated image to disk (imwrite/.save)."
    )


def test_v3_cell_11_logs_to_state():
    """Cell 11 must call state.log() to record detection count + FPS for the
    summary cell at the end of the notebook.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell11 = nb["cells"][11]
    src = "".join(cell11["source"])
    assert "state.log" in src, "v3 cell 11 must call state.log() to record visual analytics result"
    assert "t4_measured_fps" in src, "v3 cell 11 must record the measured FPS"


def test_v3_cell_11_fps_loop_actually_measures_inference():
    """REGRESSION GUARD for the Aug 22 2026 '1.9M FPS' bug.

    sv.FPSMonitor.tick() is a manual stopwatch — it just increments a
    counter. If the actual work isn't inside the tick loop, the FPS reads
    how fast the loop runs (essentially infinite), not how fast the model
    runs. The fix is to wrap the actual inference call (det.detect) inside
    the same for-loop as _fps.tick().
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell11 = nb["cells"][11]
    src = "".join(cell11["source"])
    # Find the for-loop that calls _fps.tick()
    m = re.search(r"for _ in range\(\d+\):\s*\n\s*_fps\.tick\(\)", src)
    assert m, "v3 cell 11 must have a `_fps.tick()` inside a for-loop"
    # The next non-blank statement inside the loop must be a real call
    # (not a comment, not a pass). Look for det.detect or pipeline.step.
    after = src[m.end():m.end() + 500]
    has_real_work = re.search(r"\b(?:det|pipeline)\.\w+\(", after) is not None
    assert has_real_work, (
        "v3 cell 11 FPS loop has no real work inside it — only tick() calls. "
        "Wrap the actual inference (det.detect(...) or pipeline.step(...)) "
        "inside the loop so the FPSMonitor measures real inference throughput, "
        "not loop speed."
    )


# ---------------------------------------------------------------------------
# Cell 12 (production path)
# ---------------------------------------------------------------------------

def test_v3_cell_12_is_production_path():
    """Cell 12 must be the production path (Roboflow Inference comparison)."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 13, "v3 needs at least 13 cells (...+ production path)"
    cell12 = nb["cells"][12]
    assert cell12["cell_type"] == "code", "v3 cell 12 must be code"
    src = "".join(cell12["source"])
    # Must reference the inference package
    assert "inference" in src, "v3 cell 12 must use the Roboflow Inference package"
    # Must mention the production deployment story
    assert "production" in src.lower() or "edge" in src.lower(), (
        "v3 cell 12 must frame the comparison as the production/edge deployment story."
    )


def test_v3_cell_12_handles_inference_missing():
    """The `inference` package has strict dep requirements that may conflict
    with our pinned versions. The cell must skip gracefully + print a clear
    install hint instead of crashing.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell12 = nb["cells"][12]
    src = "".join(cell12["source"])
    # Must wrap the import in try/except
    assert "try:" in src and "except ImportError" in src, (
        "v3 cell 12 must wrap the `inference` import in try/except so a "
        "dep conflict doesn't crash the demo."
    )
    # Must print an install hint on the skip path
    assert "pip install" in src, (
        "v3 cell 12 must print a 'pip install' hint on the skip path so the "
        "user knows how to enable the production-path comparison."
    )


def test_v3_cell_12_logs_to_state():
    """Cell 12 must call state.log() to record the production-path result
    (whether it ran or skipped) for the summary cell.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell12 = nb["cells"][12]
    src = "".join(cell12["source"])
    assert "state.log" in src, "v3 cell 12 must call state.log() to record the production-path result"


def test_v3_cell_12_reads_roboflow_model_id_from_userdata():
    """v1.5 fix: cell 12 must read ROBOFLOW_MODEL_ID from the user's setup
    (Colab userdata or env var) so the production-path comparison uses the
    ACTUAL recycling_v3 weights the user uploaded — not the yolov8n-640
    placeholder. Without this read, the demo can never do an apples-to-apples
    comparison automatically.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell12 = nb["cells"][12]
    src = "".join(cell12["source"])
    # Must read the env var or Colab userdata for ROBOFLOW_MODEL_ID
    assert "ROBOFLOW_MODEL_ID" in src, (
        "v3 cell 12 must read ROBOFLOW_MODEL_ID from userdata (Colab) or os.environ "
        "so the user's uploaded model ID is used automatically after the one-time setup."
    )
    # Must have BOTH paths (Colab userdata + env var) or at least one of them
    has_userdata = "userdata" in src
    has_env = "os.environ" in src or "environ" in src
    assert has_userdata or has_env, (
        "v3 cell 12 must read ROBOFLOW_MODEL_ID from Colab userdata and/or os.environ. "
        "Otherwise the user has to paste the ID into the cell on every run."
    )
    # Must have a fallback to the placeholder
    assert "yolov8n-640" in src, (
        "v3 cell 12 must fall back to the yolov8n-640 placeholder when "
        "ROBOFLOW_MODEL_ID isn't set, so the demo still works out of the box."
    )


# ---------------------------------------------------------------------------
# Cell 13 (triage + monitor)
# ---------------------------------------------------------------------------

def test_v3_cell_13_is_triage_and_monitor():
    """Cell 13 must be the operational layer (triage + robustness + dashboard)."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 14, "v3 needs at least 14 cells (...+ triage + monitor)"
    cell13 = nb["cells"][13]
    assert cell13["cell_type"] == "code", "v3 cell 13 must be code"
    src = "".join(cell13["source"])
    # Must include the 3 component classes
    for cls in ["L1TriageAgent", "RobustnessTestSuite", "MonitoringDashboard"]:
        assert cls in src, f"v3 cell 13 must use {cls}"


def test_v3_cell_13_respects_module_toggles():
    """The 3 sub-sections of cell 13 (triage, robustness, monitoring) each
    have a corresponding module:* toggle. Untick any of them and that
    sub-section should skip without breaking the others.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell13 = nb["cells"][13]
    src = "".join(cell13["source"])
    for toggle in ["module:triage", "module:robustness", "module:monitoring"]:
        assert toggle in src, (
            f"v3 cell 13 must check the `{toggle}` toggle so the sub-section "
            f"can be skipped independently of the others."
        )


def test_v3_cell_13_runs_robustness_suite():
    """The robustness sub-section must instantiate RobustnessTestSuite and
    call .run() on a real image. Reporting-only (no suite) would be a half-built cell.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell13 = nb["cells"][13]
    src = "".join(cell13["source"])
    # Must call .run() on the suite
    assert re.search(r"_suite\.run\(\)|suite\.run\(\)", src), (
        "v3 cell 13 must call .run() on the RobustnessTestSuite — running "
        "the suite is what produces the per-augmentation mAP report."
    )
    # Must call .to_markdown() (or similar) to print the report
    assert "to_markdown" in src or "to_dict" in src, (
        "v3 cell 13 must print the robustness report (via to_markdown or to_dict)."
    )


def test_v3_cell_13_logs_to_state():
    """Cell 13 must call state.log() to record the operational layer outcome
    for the summary cell at the end of the notebook.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell13 = nb["cells"][13]
    src = "".join(cell13["source"])
    assert "state.log" in src, "v3 cell 13 must call state.log() to record the triage + monitor result"
    # Must record the retrain-recommended verdict (a real operational signal)
    assert "retrain_recommended" in src or "robustness_verdict" in src, (
        "v3 cell 13 must record the retrain-recommended verdict so the summary "
        "cell can include it in the run report."
    )


def test_v3_cell_13_wires_robustness_into_retrain():
    """REGRESSION GUARD for the v1.5 'robustness/retrain contradiction' fix.

    The Coach (cell 14) flagged that the dashboard says retrain=False while
    robustness_verdict=BROKEN (5/13 augmentations failed). The v1.5 fix:
    cell 13 sub-section 3 must cross-reference state.metrics['robustness_verdict']
    and force retrain_recommended=True if BROKEN. Otherwise the dashboard tells
    the user 'don't retrain' on a model that's actually broken in production.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell13 = nb["cells"][13]
    src = "".join(cell13["source"])
    # Must reference both the dashboard's value AND the robustness verdict
    assert "robustness_verdict" in src, (
        "v3 cell 13 must read state.metrics['robustness_verdict'] to cross-reference "
        "the robustness suite's output into the retrain decision."
    )
    # Must have an override pattern: "if BROKEN and not _dash_retrain"
    has_override = re.search(
        r"['\"]BROKEN['\"].*?_final_retrain|rb_verdict.*?BROKEN.*?_final_retrain",
        src,
        re.DOTALL,
    )
    assert has_override, (
        "v3 cell 13 must override the dashboard's retrain_recommended when "
        "robustness_verdict == 'BROKEN'. The Coach caught the contradiction; "
        "this is the fix."
    )


def test_v3_cell_13_inference_unit_derived_from_key():
    """REGRESSION GUARD for the Aug 22 2026 '67.4 ms/frame' label bug.

    The dashboard's inference line was printing the value with a unit
    derived from the VALUE string ('67.4' has no 'fps' substring → 'ms/frame'),
    but the value 67.4 is actually FPS (from sv.FPSMonitor.fps). The fix is
    to derive the unit from the METRIC KEY ('t4_measured_fps' contains 'fps'
    → 'FPS', 't4_inference_ms' contains 'ms' → 'ms/frame'). If anyone reverts
    the fix, the test catches it.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell13 = nb["cells"][13]
    src = "".join(cell13["source"])
    # The bug pattern: unit detection on the value string
    bad_pattern = re.search(r"['\"]FPS['\"].*?fps['\"]\s*in\s+str\(_fps\)\.lower", src) or \
                  re.search(r"['\"]ms/frame['\"].*?fps['\"]\s*in\s+str\(_fps\)\.lower", src)
    assert not bad_pattern, (
        "v3 cell 13 derives the inference unit from the value string "
        "('fps' in str(_fps).lower). This is the Aug 22 2026 bug — the "
        "value 67.4 has no 'fps' substring, so the label was 'ms/frame' "
        "even though 67.4 is FPS. Derive the unit from the METRIC KEY instead."
    )
    # The fix pattern: unit derived from the key
    good_pattern = re.search(
        r"_fps_key.*?fps.*?_fps_unit|_fps_unit.*?fps_key|'t4_measured_fps'.*?FPS|'t4_inference_ms'.*?ms/frame",
        src,
    )
    assert good_pattern, (
        "v3 cell 13 must derive the inference unit from the metric KEY, "
        "not the value. Look for _fps_key/_fps_unit variables or explicit "
        "'t4_measured_fps' → FPS / 't4_inference_ms' → ms/frame mapping."
    )


# ---------------------------------------------------------------------------
# Cell 14 (coach + summary)
# ---------------------------------------------------------------------------

def test_v3_cell_14_is_coach_and_summary():
    """Cell 14 must be the closer (coach review + run summary + session log)."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 15, "v3 needs at least 15 cells (...+ coach + summary)"
    cell14 = nb["cells"][14]
    assert cell14["cell_type"] == "code", "v3 cell 14 must be code"
    src = "".join(cell14["source"])
    # Must call coach_review for the Gemini section
    assert "coach_review" in src, "v3 cell 14 must call coach_review() for the Gemini review"
    # Must read state for the run summary
    assert "state.metrics" in src or "summary_table" in src, (
        "v3 cell 14 must read state for the run summary."
    )
    # Must save the session log
    assert "session_log" in src or "to_dict" in src, (
        "v3 cell 14 must save a downloadable session log."
    )


def test_v3_cell_14_handles_missing_gemini_key():
    """coach_review returns a string even without a GEMINI_API_KEY (it just
    says 'No GEMINI_API_KEY configured. Skipping Coach review.'). The cell
    must catch any exception from the coach and continue — the summary +
    session log are the must-haves, the coach is the cherry on top.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell14 = nb["cells"][14]
    src = "".join(cell14["source"])
    # Must wrap coach_review in try/except
    assert "try:" in src and "except" in src, (
        "v3 cell 14 must wrap coach_review in try/except so a Coach failure "
        "(no API key, no SDK, network) doesn't kill the summary or session log."
    )


def test_v3_cell_14_logs_to_state():
    """Cell 14 must call state.log() to mark the run as complete — the
    notebook's audit trail ends here.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell14 = nb["cells"][14]
    src = "".join(cell14["source"])
    assert "state.log" in src, "v3 cell 14 must call state.log() to mark the run as complete"


# ---------------------------------------------------------------------------
# Cell 15 (T4 vs EverestLabs — the final comparison)
# ---------------------------------------------------------------------------

def test_v3_cell_15_is_t4_vs_everestlabs():
    """Cell 15 must be the T4 vs EverestLabs comparison (the final cell)."""
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) >= 16, "v3 needs at least 16 cells (title + 15 numbered)"
    cell15 = nb["cells"][15]
    assert cell15["cell_type"] == "code", "v3 cell 15 must be code"
    src = "".join(cell15["source"])
    # Must use the styled comparison table helper
    assert "render_comparison_table" in src, (
        "v3 cell 15 must use render_comparison_table() to produce the styled "
        "comparison HTML table."
    )
    # Must reference the EverestLabs published spec
    assert "Everest" in src or "EVEREST" in src, (
        "v3 cell 15 must reference the EverestLabs published spec as the reference."
    )
    # Must read T4 measured numbers from state.metrics
    assert "state.metrics" in src, "v3 cell 15 must read T4 numbers from state.metrics"


def test_v3_cell_15_computes_verdict():
    """Cell 15 must compute a verdict (meets_bar / below_bar) based on the
    T4 vs EverestLabs comparison. The verdict is the headline result of
    the whole demo.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell15 = nb["cells"][15]
    src = "".join(cell15["source"])
    # Must have a verdict string somewhere
    has_verdict = "meets_bar" in src or "below_bar" in src or "verdict" in src.lower()
    assert has_verdict, (
        "v3 cell 15 must compute a verdict (meets_bar / below_bar) — the "
        "headline result of the whole demo."
    )


def test_v3_cell_15_logs_to_state():
    """Cell 15 must call state.log() to record the verdict — this is the
    notebook's last cell, so the log is the final audit-trail entry.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell15 = nb["cells"][15]
    src = "".join(cell15["source"])
    assert "state.log" in src, "v3 cell 15 must call state.log() to record the final verdict"
