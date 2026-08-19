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


def test_cell_count_is_29():
    nb = json.loads(NOTEBOOK.read_text())
    # 1 markdown intro + 1 code (hero) + 1 markdown (how to use) + 1 markdown §1 header + 4 code (§1)
    # + 1 markdown §2 header + 8 code (§2 + visual analytics + production path) + 1 markdown §3 header + 1 code (§3)
    # + 1 markdown §4 header + 3 code (§4, +publish cell) + 1 markdown §5 header
    # + 5 code (§5 stage cells + widget dashboard) + 1 markdown §5 close
    # = 8 markdown + 21 code = 29 total
    assert len(nb["cells"]) == 29


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
    # 7 markdown (how-to + 5 section headers + §5 close) + 22 code (hero + cells 1-15 + visual + production + dashboard + §5 stages)
    assert md_count == 7, f"expected 7 markdown cells, got {md_count}"
    assert code_count == 22, f"expected 22 code cells, got {code_count}"


def test_publish_cell_uses_pat_and_pyg_github():
    """The publish cell should authenticate with GITHUB_TOKEN via PyGithub."""
    nb = json.loads(NOTEBOOK.read_text())
    publish_cell = _find_cell_by_comment(nb, "Publish to GitHub Release")
    assert publish_cell is not None, "could not find the publish cell"
    src = "".join(publish_cell["source"])
    assert "GITHUB_TOKEN" in src
    assert "PyGithub" in src or "from github import Github" in src
    assert "create_git_release" in src
    # PyGithub 2.x method is 'upload_asset' (NOT 'upload_asset_from_path')
    assert "upload_asset_from_path" not in src, \
        "PyGithub 2.x uses 'upload_asset', not 'upload_asset_from_path'"
    assert "release.upload_asset(" in src, "must call release.upload_asset(...) to attach the asset"


def test_publish_cell_self_heals_pyg_github_install():
    """The publish cell must self-install PyGithub if missing.

    Colab doesn't ship PyGithub by default. The earlier `!pip install`
    can be masked by resolver warnings, so the cell needs a try/except
    fallback that runs `pip install --no-deps PyGithub` and re-imports.
    """
    nb = json.loads(NOTEBOOK.read_text())
    publish_cell = _find_cell_by_comment(nb, "Publish to GitHub Release")
    assert publish_cell is not None, "could not find the publish cell"
    src = "".join(publish_cell["source"])
    # The self-healing block must come BEFORE the first GITHUB_TOKEN use
    token_pos = src.find("GITHUB_TOKEN")
    heal_pos = src.find("PyGithub not found")  # the install-on-miss message
    assert heal_pos != -1, "publish cell must self-heal PyGithub install"
    assert heal_pos < token_pos, "self-heal must run before the GITHUB_TOKEN lookup"
    # The fallback must be --no-deps to avoid numpy 1.26/2.x conflict
    assert "--no-deps" in src, "must use --no-deps to avoid numpy conflict"
    assert "subprocess.check_call" in src or "pip install" in src, "must invoke pip install"


def test_cell1_self_heals_colab_session_import():
    """Cell 1 (env check) must self-heal the repo clone in 3 cases.

    Colab's /content persists across runtime restarts, so the repo may
    exist in 3 different states:
      (a) Not cloned yet           → git clone
      (b) Cloned (valid git repo)  → git pull --rebase
      (c) Dir exists but not git   → rm -rf, then git clone

    The self-heal handles all 3. It also catches CalledProcessError and
    prints a clear error with next steps instead of crashing the cell.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell1 = _find_cell_by_comment(nb, "Runtime + env check")
    assert cell1 is not None, "could not find cell 1 (Runtime + env check)"
    src = "".join(cell1["source"])
    # The self-heal block must exist BEFORE the post-heal import statement
    import_pos = src.rfind("from colab_session import env_check, get_state")
    heal_pos = src.find("is_valid_git")
    assert heal_pos != -1, "cell 1 must self-heal colab_session by handling the repo state"
    assert heal_pos < import_pos, "self-heal must come before the final import statement"
    # All 3 cases must be handled
    assert "shutil.rmtree" in src, "case (c): must handle non-git dir by removing it"
    assert "'clone'" in src and "'git'" in src, "must invoke git clone via subprocess"
    assert "git pull" in src or "'pull'" in src, "case (b): must git pull when repo is valid"
    assert "roniejosephv-star/conveyor-perception" in src, "must use the right repo URL"
    # Must catch errors and print troubleshooting steps
    assert "Self-heal failed" in src or "Possible causes" in src, \
        "must print clear error if self-heal fails"


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
    assert "T4 (this Colab run)" in src or "T4 (this run)" in src


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


def test_cell9_resolves_model_path_via_yolo():
    """Cell 9 must resolve the model path via YOLO() before passing to UltralyticsDetector.

    UltralyticsDetector does a Path.exists() check, but YOLO() downloads the
    model to ~/.cache/ultralytics/, not CWD. Calling YOLO(model_path) first
    triggers the auto-download AND returns the resolved ckpt_path.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell9 = _find_cell_by_comment(nb, "End-to-end pipeline")
    assert cell9 is not None, "could not find cell 9"
    src = "".join(cell9["source"])
    # Must call YOLO() to trigger auto-download
    assert "YOLO(raw_model)" in src or "YOLO(" in src, "must call YOLO() to trigger auto-download"
    # Must use the resolved ckpt_path
    assert "ckpt_path" in src, "must use the resolved .ckpt_path (not the raw model name)"


def test_cell10_is_self_sufficient_when_cell9_fails():
    """Cell 10 must re-create 'triage' and 'dashboard' if cell 9 didn't run.

    If cell 9 fails (e.g., model download error), cell 10 was crashing with
    NameError on `triage` or `dashboard` because those vars are defined in
    cell 9. The fix: cell 10 re-creates them with safe defaults.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell10 = _find_cell_by_comment(nb, "Triage queue, robustness suite")
    assert cell10 is not None, "could not find cell 10"
    src = "".join(cell10["source"])
    # Must import L1TriageAgent + MonitoringDashboard (so it can re-create them)
    assert "L1TriageAgent" in src, "cell 10 must import L1TriageAgent"
    assert "MonitoringDashboard" in src, "cell 10 must import MonitoringDashboard"
    # Must re-create 'triage' and 'dashboard' if missing
    assert "'triage' not in dir()" in src, "cell 10 must re-create 'triage' if missing"
    assert "'dashboard' not in dir()" in src, "cell 10 must re-create 'dashboard' if missing"
    # Must guard the robustness suite on 'det' availability
    assert "'det' in dir()" in src, "cell 10 must check if 'det' is available before robustness suite"


def test_cell14_does_not_import_summary_table_as_function():
    """Cell 14 must not import summary_table from colab_session (it's a method).

    summary_table is a method on SessionState, not a module-level function.
    Importing it raises ImportError. The cell calls state.summary_table()
    which works because the method is on the state instance, not the import.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell14 = _find_cell_by_comment(nb, "Summary + downloadable")
    assert cell14 is not None, "could not find cell 14 (summary)"
    src = "".join(cell14["source"])
    assert "import summary_table" not in src, \
        "summary_table is a method on SessionState, not a module function"


# --- §5 OPTIMIZATION LOOP tests ------------------------------------------


def test_publish_cell_uses_correct_pyg_github_method():
    """The publish cell must use upload_asset (PyGithub 2.x).

    PyGithub 2.x has `release.upload_asset(path, name=...)`. Using a
    non-existent variant raises AttributeError, the cell() context manager
    catches it as 1 error, and the optimization loop breaks because the
    release has no assets for the Action to download.
    """
    nb = json.loads(NOTEBOOK.read_text())
    publish_cell = _find_cell_by_comment(nb, "Publish to GitHub Release")
    assert publish_cell is not None, "could not find the publish cell"
    src = "".join(publish_cell["source"])
    # The fix for the non-existent method variant (broken version is in history)
    assert ".upload_asset_from_path(" not in src, \
        "PyGithub 2.x doesn't have upload_asset_from_path — use upload_asset(path, name=...)"
    assert "release.upload_asset(" in src, "must call release.upload_asset(...) to attach the session.json"


def test_cell6_uses_correct_class_names_and_constructors():
    """Cell 6 must use the right class names + constructors.

    Two bugs were fixed in commit d6e9b31 (this):
      1. `Detector` doesn't exist — it's `DetectionPipeline`, aliased.
      2. `MCPTriageSurface()` with no args fails — needs (name, alert_source).
    Also: cell 6 must add /content/conveyor-perception/src to sys.path
    so the `conveyor_perception` package is importable.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell6 = _find_cell_by_comment(nb, "The 4 framework abstractions")
    assert cell6 is not None, "could not find cell 6 (4 framework abstractions)"
    src = "".join(cell6["source"])
    # Must alias DetectionPipeline as Detector (Detector itself doesn't exist)
    assert "DetectionPipeline as Detector" in src, \
        "must alias DetectionPipeline as Detector (Detector class doesn't exist)"
    # Must pass name + InMemoryAlertQueue to MCPTriageSurface
    assert "MCPTriageSurface('l1-triage', InMemoryAlertQueue())" in src, \
        "MCPTriageSurface needs (name, alert_source)"
    assert "InMemoryAlertQueue" in src, "must import InMemoryAlertQueue"
    # Must add the src/ dir to sys.path so the package imports
    assert "/content/conveyor-perception/src" in src, \
        "must add /content/conveyor-perception/src to sys.path"


def test_no_broken_state_cell_usage():
    """The notebook must use `cell(...)` (module-level fn), not `state.cell(...)`.

    `cell()` is a module-level contextmanager in colab_session.py, not a
    method on SessionState. Using `state.cell(...)` raises AttributeError
    on every cell that uses it. The fix: import `cell` and use it bare.
    """
    nb = json.loads(NOTEBOOK.read_text())
    bad = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "state.cell(" in src:
            bad.append(i)
    assert bad == [], (
        f"Cells {bad} use state.cell(...) which is broken — use the module-level "
        f"`cell(...)` from colab_session instead. Add 'cell' to the import in cell 1."
    )


def test_section_5_header_present():
    """The notebook must have the §5 OPTIMIZATION LOOP section header."""
    nb = json.loads(NOTEBOOK.read_text())
    full_text = "\n".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "markdown"
    )
    assert "§5 OPTIMIZATION LOOP" in full_text, "missing §5 OPTIMIZATION LOOP header"
    assert "PUBLISH" in full_text
    assert "TRIGGER" in full_text
    assert "ANALYZE" in full_text
    assert "PROPOSE" in full_text
    # The closing statement
    assert "framework improves itself" in full_text or "What you just saw" in full_text


def test_all_four_stage_cells_present():
    """Each of the 4 stages must have a dedicated code cell with a status indicator."""
    nb = json.loads(NOTEBOOK.read_text())
    for stage in ["§5 STAGE 1 — PUBLISH", "§5 STAGE 2 — TRIGGER", "§5 STAGE 3 — ANALYZE", "§5 STAGE 4 — PROPOSE"]:
        cell = _find_cell_by_comment(nb, stage)
        assert cell is not None, f"missing stage cell: {stage}"
        src = "".join(cell["source"])
        # Each stage must have a status emoji and a hint
        assert "✅" in src or "⏳" in src or "❌" in src or "🔄" in src, \
            f"{stage} must have a status emoji"
        assert "💡 Audience hint" in src, f"{stage} must have an audience hint"
        # Each stage must call the GitHub REST API (or have a no-token fallback)
        assert "requests.get" in src or "_no_token_msg" in src, \
            f"{stage} must call the GitHub API or have a no-token fallback"


def test_stage_cells_use_github_rest_api():
    """The stage cells should use the GitHub REST API directly (no PyGithub dep)."""
    nb = json.loads(NOTEBOOK.read_text())
    for stage in ["§5 STAGE 1", "§5 STAGE 2", "§5 STAGE 3", "§5 STAGE 4"]:
        cell = _find_cell_by_comment(nb, stage)
        assert cell is not None, f"missing {stage}"
        src = "".join(cell["source"])
        assert "api.github.com" in src, f"{stage} must hit api.github.com"
        assert "Authorization" in src, f"{stage} must send auth header"
        assert "per_page" in src, f"{stage} must bound the response size"


def test_stage_cells_self_heal_no_token():
    """If GITHUB_TOKEN is missing, each stage must print a hint, not crash."""
    nb = json.loads(NOTEBOOK.read_text())
    for stage in ["§5 STAGE 1", "§5 STAGE 2", "§5 STAGE 3", "§5 STAGE 4"]:
        cell = _find_cell_by_comment(nb, stage)
        assert cell is not None, f"missing {stage}"
        src = "".join(cell["source"])
        assert "_no_token_msg" in src, f"{stage} must use the no-token helper"
        assert "no GITHUB_TOKEN" in src, f"{stage} must tell the user what is missing"


# --- Interactive (hybrid) chrome tests -----------------------------------


def test_production_path_cell_uses_roboflow_inference():
    """Cell 9.7 (production path) must use Roboflow Inference, with graceful fallback."""
    nb = json.loads(NOTEBOOK.read_text())
    prod = _find_cell_by_comment(nb, "Roboflow Inference")
    assert prod is not None, "could not find the production path cell"
    src = "".join(prod["source"])
    # Must use Roboflow Inference
    assert "inference.models.utils" in src, "must use inference.models.utils.get_model"
    # Must be self-healing: try/except ImportError
    assert "except ImportError" in src, "must handle ImportError gracefully"
    # Must explain the production story
    assert "production" in src.lower(), "must explain the production deployment story"


def test_visual_analytics_cell_uses_modern_annotators():
    """Cell 9.5 (visual analytics) must use the modern supervision annotators."""
    nb = json.loads(NOTEBOOK.read_text())
    visual = _find_cell_by_comment(nb, "Visual Analytics")
    assert visual is not None, "could not find the visual analytics cell"
    src = "".join(visual["source"])
    # Must use the modern annotators
    assert "RoundBoxAnnotator" in src, "must use sv.RoundBoxAnnotator"
    assert "RichLabelAnnotator" in src, "must use sv.RichLabelAnnotator (pill labels)"
    assert "HeatMapAnnotator" in src, "must use sv.HeatMapAnnotator (density)"
    assert "PolygonZone" in src, "must use sv.PolygonZone (spatial)"
    assert "LineZone" in src, "must use sv.LineZone (throughput counter)"
    assert "FPSMonitor" in src, "must use sv.FPSMonitor (real FPS)"


def test_hero_cell_uses_render_hero_helper():
    """The first cell (hero) must use the render_hero helper for a rich front door."""
    nb = json.loads(NOTEBOOK.read_text())
    first = nb["cells"][0]
    src = "".join(first["source"])
    assert first["cell_type"] == "code", "hero must be a code cell (HTML renders in output)"
    assert "render_hero" in src, "hero must use the render_hero helper"
    assert "tinkr-hero" not in src, "hero should CALL the helper, not embed raw HTML"


def test_widget_dashboard_uses_ipywidgets_tab():
    """The §5 dashboard must use ipywidgets.Tab with 4 tabs."""
    nb = json.loads(NOTEBOOK.read_text())
    dashboard = _find_cell_by_comment(nb, "Interactive Dashboard")
    assert dashboard is not None, "could not find the §5 dashboard cell"
    src = "".join(dashboard["source"])
    assert "ipywidgets" in src, "must import ipywidgets"
    assert "widgets.Tab" in src, "must use widgets.Tab"
    assert "Pipeline Flow" in src, "must have a Pipeline Flow tab"
    assert "Live Stats" in src, "must have a Live Stats tab"
    assert "Coach Log" in src, "must have a Coach Log tab"
    assert "Releases" in src, "must have a Releases tab"
