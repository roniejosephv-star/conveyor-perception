"""Tests for notebooks/build_demo_v2.py.

These tests verify the notebook builder:
- Generates valid JSON
- Produces the expected number of cells
- Each cell has the required fields
- Markdown / code cells are balanced
- Key sections (§1 §2 §3 §4) are present
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILDER = REPO_ROOT / "notebooks" / "build_demo_v2.py"
NOTEBOOK = REPO_ROOT / "notebooks" / "demo_v2.ipynb"


def _run_builder() -> None:
    """Re-run the notebook builder to make sure the on-disk .ipynb is current."""
    result = subprocess.run(
        [sys.executable, str(BUILDER)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, f"builder failed:\n{result.stderr}"


@pytest.fixture(scope="module", autouse=True)
def rebuild_notebook():
    _run_builder()
    yield
    # Don't rebuild on teardown — leave the file in its current state.


def test_notebook_exists_and_is_valid_json():
    assert NOTEBOOK.exists()
    nb = json.loads(NOTEBOOK.read_text())
    assert "cells" in nb
    assert "nbformat" in nb
    assert nb["nbformat"] == 4


def test_cell_count_is_19():
    nb = json.loads(NOTEBOOK.read_text())
    # 1 markdown intro + 1 markdown §1 header + 4 code (§1) + 1 markdown §2 header + 6 code (§2)
    # + 1 markdown §3 header + 1 code (§3) + 1 markdown §4 header + 3 code (§4, +publish cell)
    # = 5 markdown + 14 code = 19 total
    assert len(nb["cells"]) == 19


def test_cells_have_required_fields():
    nb = json.loads(NOTEBOOK.read_text())
    for i, cell in enumerate(nb["cells"]):
        assert "cell_type" in cell, f"cell {i} missing cell_type"
        assert cell["cell_type"] in ("markdown", "code"), f"cell {i} bad cell_type"
        assert "source" in cell, f"cell {i} missing source"
        assert isinstance(cell["source"], list), f"cell {i} source must be list"


def test_markdown_code_balance():
    nb = json.loads(NOTEBOOK.read_text())
    md_count = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    code_count = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    # 5 markdown (intro + 4 section headers) + 14 code cells
    assert md_count == 5, f"expected 5 markdown cells, got {md_count}"
    assert code_count == 14, f"expected 14 code cells, got {code_count}"


def test_publish_cell_uses_pat_and_pyg_github():
    """The publish cell should authenticate with GITHUB_TOKEN via PyGithub."""
    nb = json.loads(NOTEBOOK.read_text())
    publish_cell = _find_cell_by_comment(nb, "Publish to GitHub Release")
    assert publish_cell is not None, "could not find the publish cell"
    src = "".join(publish_cell["source"])
    assert "GITHUB_TOKEN" in src
    assert "PyGithub" in src or "from github import Github" in src
    assert "create_git_release" in src
    assert "upload_asset_from_path" in src


def test_all_four_sections_present():
    nb = json.loads(NOTEBOOK.read_text())
    full_text = "\n".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "markdown"
    )
    for section in ["§1 SETUP", "§2 WALKTHROUGH", "§3 COMPARISON", "§4 COACH"]:
        assert section in full_text, f"missing section header: {section}"


def test_state_helper_imports_in_setup():
    """Cells 1-5 (§1) should import + use the colab_session helpers."""
    nb = json.loads(NOTEBOOK.read_text())
    setup_code = "\n".join(
        "".join(c["source"])
        for c in nb["cells"][:5]
        if c["cell_type"] == "code"
    )
    assert "colab_session" in setup_code
    assert "get_state" in setup_code
    assert "env_check" in setup_code


def test_coach_cell_uses_gemini():
    """The error-diagnosis cell should call coach_diagnose."""
    nb = json.loads(NOTEBOOK.read_text())
    coach_cell = _find_cell_by_comment(nb, "Error log + Coach diagnosis")
    assert coach_cell is not None, "could not find Coach diagnosis cell"
    src = "".join(coach_cell["source"])
    assert "coach_diagnose" in src
    assert "state.errors" in src


def test_summary_cell_offers_download():
    """The final cell should call download_session_log."""
    nb = json.loads(NOTEBOOK.read_text())
    summary_cell = _find_cell_by_comment(nb, "Summary + downloadable")
    assert summary_cell is not None, "could not find Summary cell"
    src = "".join(summary_cell["source"])
    assert "download_session_log" in src
    assert "state.summary_table" in src


def _find_cell_by_comment(nb: dict, comment_substring: str) -> dict | None:
    """Find a code cell whose first comment line contains the substring."""
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        first_lines = "\n".join(src.split("\n")[:3])
        if comment_substring.lower() in first_lines.lower():
            return cell
    return None


def test_kernel_spec_python3():
    nb = json.loads(NOTEBOOK.read_text())
    spec = nb["metadata"]["kernelspec"]
    assert spec["language"] == "python"
    assert spec["name"] == "python3"


def test_no_banned_pii_in_cells():
    """No competitor names, no leaked secrets, no internal jargon.

    Note: "EverestLabs" IS allowed in this notebook because the user is
    interviewing AT EverestLabs (target company), and the comparison cell
    is meant to show their stack vs the prototype's. Real competitors
    (AMP Robotics, etc.) and the user's other projects (Tinkr, Argus) are
    banned. M4 is also banned — the user explicitly removed it because
    it isn't relevant to the EverestLabs narrative.
    """
    nb = json.loads(NOTEBOOK.read_text())
    full_text = "\n".join(
        "".join(c.get("source", [])) for c in nb["cells"]
    ).lower()
    banned = [
        "amp robotics",   # real competitor
        "tinkr",          # user's other project — never name in job-hunt contexts
        "argus",          # user's other project — never name in job-hunt contexts
        "blink",          # old Tinkr codename
        "m4 mps",         # not relevant to EverestLabs
        " m4 (this mac)",
        "(mac m4",
    ]
    for token in banned:
        assert token not in full_text, f"banned token {token!r} in notebook"


def test_comparison_cell_has_no_m4():
    """The §3 comparison cell should have only EverestLabs and T4 columns."""
    nb = json.loads(NOTEBOOK.read_text())
    cmp_cell = _find_cell_by_comment(nb, "T4 vs EverestLabs")
    assert cmp_cell is not None, "could not find the comparison cell"
    src = "".join(cmp_cell["source"])
    assert "M4_MEASURED" not in src
    assert "M4 (this Mac)" not in src
    # Should still have both EverestLabs + T4 columns
    assert "EverestLabs" in src
    assert "T4 (this run)" in src


def test_toggle_cell_calls_toggle_ui():
    """The toggle UI cell should call toggle_ui() and display it."""
    nb = json.loads(NOTEBOOK.read_text())
    toggle_cell = _find_cell_by_comment(nb, "Module toggle UI")
    assert toggle_cell is not None, "could not find toggle UI cell"
    src = "".join(toggle_cell["source"])
    assert "toggle_ui" in src
    assert "display" in src


def test_pipeline_cell_reads_toggles():
    """The module-load cell (cell 8) should read state.toggles for each component."""
    nb = json.loads(NOTEBOOK.read_text())
    # Find the cell whose comment is "Cell 7: The 7+1 JD modules"
    modules_cell = None
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "load 7+1 JD modules" in src.lower() or "load-7-jd-modules" in src.lower() or "7+1 JD modules" in src:
            modules_cell = cell
            break
    assert modules_cell is not None, "could not find the modules-load cell"
    src = "".join(modules_cell["source"])
    assert "state.toggles" in src
    # Each module's toggle is checked
    assert "module:perception" in src
    assert "module:triage" in src
