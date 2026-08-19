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
    import_pos = src.rfind("from colab_session import env_check, get_state, cell, init_progress_dashboard")
    heal_pos = src.find("_colab_session_ready")
    assert heal_pos != -1, "cell 1 must self-heal colab_session by handling the repo state"
    assert heal_pos < import_pos, "self-heal must come before the final import statement"
    # All 3 cases must be handled
    assert "shutil.rmtree" in src, "case (c): must handle non-git dir by removing it"
    assert "'clone'" in src and "'git'" in src, "must invoke git clone via subprocess"
    assert "'pull'" in src, "case (b): must git pull when repo is valid"
    assert "roniejosephv-star/conveyor-perception" in src, "must use the right repo URL"
    # Must catch errors and print troubleshooting steps
    # (post-bulletproof: error message is "import still failed / re-cloning fresh")
    assert any(s in src for s in [
        "Self-heal failed",
        "Possible causes",
        "Pull + import still failed",
        "import still failed",
        "re-cloning fresh",
    ]), "must print clear error if self-heal fails"


def test_cell1_is_bulletproof():
    """Cell 1 must have a fallback for every operation. No naked os.chdir,
    no naked imports, no unhandled exceptions. The cell can ONLY die if
    colab_session cannot be imported even after a fresh clone (and that
    path raises SystemExit with a clear message).
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell1 = _find_cell_by_comment(nb, "Runtime + env check")
    assert cell1 is not None
    src = "".join(cell1["source"])
    # Step 1 (chdir) is wrapped in try/except OSError
    assert "try:" in src and "os.chdir(REPO)" in src
    assert "except OSError" in src, "chdir must be wrapped in try/except OSError"
    # The self-heal has a _colab_session_ready flag (not just a bare import)
    assert "_colab_session_ready" in src, "must use a flag to track import success"
    # If the flag is False, the cell raises SystemExit with a clear message
    assert "raise SystemExit" in src, "must raise SystemExit with clear msg if import fails"
    assert "CRITICAL:" in src, "SystemExit must say CRITICAL"
    assert "Disconnect and delete runtime" in src, "must tell user how to recover"
    # The dashboard init is wrapped in try/except (belt+suspenders)
    assert "init_progress_dashboard" in src
    # The state.log is wrapped in try/except
    assert "state.log" in src
    # The cell always ends with the success line
    assert "✓ Cell 1 done" in src or "Cell 1 done" in src


def test_cell1_handles_file_at_repo_path():
    """Cell 1 must handle a botched state where REPO is a file (not a dir).
    The os.path.isdir check guards os.chdir, and the self-heal nukes + re-clones.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell1 = _find_cell_by_comment(nb, "Runtime + env check")
    src = "".join(cell1["source"])
    assert "os.path.isdir(REPO)" in src, "must check isdir before chdir"
    assert "elif os.path.exists(REPO)" in src, "must handle file-at-REPO case"


def test_cell1_no_repeating_loop():
    """Cell 1 must not loop. The self-heal runs at most 3 times (initial +
    pull-retry + nuke-retry). After the 3rd attempt fails, the cell raises
    SystemExit with a clear message — no silent infinite retries.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell1 = _find_cell_by_comment(nb, "Runtime + env check")
    src = "".join(cell1["source"])
    import_count = src.count("from colab_session import env_check, get_state, cell, init_progress_dashboard")
    # 3 imports is correct: initial + pull-retry + nuke-retry (last-resort)
    assert import_count == 3, (
        f"cell 1 must import colab_session exactly 3 times "
        f"(initial + pull-retry + nuke-retry). Found {import_count} — "
        f"a higher count indicates a retry loop, a lower count means we lost a fallback."
    )


def test_cell2_installs_trackers_for_bytetrack():
    """Cell 2 (install + clone) must install the `trackers` package —
    it's the new home for ByteTrack (Roboflow, Apache 2.0) and replaces
    the deprecated supervision.ByteTrack. Without it, the tracking
    pipeline falls back to the simple IoU tracker (works, but worse IDs).
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell2 = _find_cell_by_comment(nb, "Install + clone + Roboflow key")
    assert cell2 is not None
    src = "".join(cell2['source'])
    assert "trackers" in src, (
        "cell 2 must install the `trackers` package (>=2.6.0) so ByteTrack "
        "is available. Otherwise the tracking pipeline falls back to the "
        "simple IoU tracker and emits the warning storm in cell 9."
    )
    # The pin should be specific (>=2.6.0)
    import re
    pin_match = re.search(r"trackers[=<>]+([\d.]+)", src)
    assert pin_match is not None, "trackers package must be pinned (e.g. >=2.6.0)"
    pin_version = pin_match.group(1)
    pin_major_minor = float(".".join(pin_version.split(".")[:2]))
    assert pin_major_minor >= 2.6, (
        f"trackers must be >=2.6.0 (the ByteTrack version), got {pin_match.group(0)}"
    )


def test_cell2_splits_install_into_critical_and_optional():
    """Cell 2 must install in TWO PASSES — critical deps (ultralytics,
    supervision, trackers) MUST succeed or the cell raises, while optional
    deps (roboflow, gemini) can fail without breaking the demo.

    History of the bulletproof pattern (Aug 2026):
    - v1: subprocess.run([sys.executable, '-m', 'pip', 'install', ...]) —
      returned 0 but didn't update IPython module registry.
    - v2: get_ipython().run_line_magic('pip', 'install -q ...') — bash split
      the comma in 'numpy>=1.26,<2.0', error was swallowed, both passes
      reported success but nothing installed.
    - v3: literal `%pip install ...` line — IPython parsed it natively, but
      pip raised ResolutionImpossible (numpy<2.0 vs ultralytics>=8.4 which
      needs numpy>=2.0). The %pip line silently failed (no exception,
      no exit code surfaced) and the cell printed '✓ installed' even
      though ultralytics was not on disk.
    - v4 (current): subprocess.run([sys.executable, '-m', 'pip', 'install',
      ...], check=True) — RAISES on non-zero exit. No numpy pin (ultralytics
      8.4 needs numpy>=2.0; Colab has 2.5.1 pre-installed).

    The two-pass split guarantees the critical packages are present.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell2 = _find_cell_by_comment(nb, "Install + clone + Roboflow key")
    assert cell2 is not None
    src = "".join(cell2['source'])
    # CRITICAL pass must use subprocess.run with check=True (raises on failure)
    # The literal %pip line silently swallows dependency conflicts.
    crit_block = src.split("CRITICAL_PKGS")[1].split("OPTIONAL_PKGS")[0] if "CRITICAL_PKGS" in src else src
    # Look for the install command and verification
    assert "subprocess.run" in crit_block or "%pip install" in crit_block, (
        "must use subprocess.run or %pip for the critical pass"
    )
    # If subprocess.run, must NOT be check=False (silent failure)
    if "subprocess.run" in crit_block:
        # Either check=True (raises) OR capture_output + returncode check
        assert "check=True" in crit_block or "returncode" in crit_block, (
            "subprocess.run for critical install must either use check=True (raises) "
            "or check returncode + raise (silent failure otherwise)"
        )
    # Critical must include the demo's hard dependencies
    assert "ultralytics" in crit_block, "ultralytics must be in the critical install"
    assert "supervision" in crit_block, "supervision must be in the critical install"
    assert "trackers" in crit_block, "trackers must be in the critical install (for ByteTrack)"
    # MUST NOT pin numpy<2.0 — ultralytics 8.4 needs numpy>=2.0
    assert "numpy<2.0" not in src and "numpy<2" not in src, (
        "numpy<2.0 pin is FORBIDDEN — ultralytics 8.4 requires numpy>=2.0, the "
        "pin causes a ResolutionImpossible that %pip silently swallows"
    )
    # OPTIONAL pass must use a list (no commas in those specs)
    assert "OPTIONAL" in src.upper(), "must have an optional pass"
    assert "fastmcp" in src, "fastmcp must be in optional pass"
    assert "google-generativeai" in src, "google-generativeai must be in optional pass"
    # Must verify imports after install
    assert "__import__" in src, "must verify critical imports after install"


def test_cell2_uses_pip_magic_not_subprocess():
    """Cell 2 must use the IPython %pip magic, NOT subprocess.run([pip, ...]).

    This is a CRITICAL Colab platform fact (Aug 2026): on Colab, a notebook cell
    that calls `subprocess.run([sys.executable, '-m', 'pip', 'install', ...])`
    can return exit code 0 without the new package becoming importable in
    subsequent cells OR in subprocess Python. The Colab IPython kernel keeps
    a separate module registry that subprocess pip doesn't update.

    The platform-correct fix: use a LITERAL `%pip install ...` line (or
    `get_ipython().run_line_magic('pip', ...)`) so Colab's kernel handles
    the install specially and updates the registry.

    Symptom of regression: run-1787150113.json — cell 2 reported 'ok' in 9.4s,
    but cell 8's subprocess `from ultralytics import YOLO` raised
    ModuleNotFoundError, and cell 9's `from ultralytics import YOLO` also
    raised ModuleNotFoundError.

    See: https://github.com/googlecolab/colabtools/issues/1481
         "Many apparently installed modules listed with !pip list cannot be imported"
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell2 = _find_cell_by_comment(nb, "Install + clone + Roboflow key")
    assert cell2 is not None
    src = "".join(cell2['source'])
    # Must use literal %pip line or run_line_magic (NOT subprocess pip)
    assert "run_line_magic('pip'" in src or "%pip install" in src, (
        "cell 2 must use the IPython %pip magic — subprocess.run([pip, ...]) "
        "silently fails to update the Colab kernel's module registry"
    )
    # Must NOT use subprocess for the install (only for clone/pull).
    # Skip comment lines (the explanatory text mentions subprocess.run for context).
    install_lines = [
        line for line in src.split('\n')
        if "subprocess.run" in line and "pip" in line
        and not line.lstrip().startswith("#")
    ]
    assert not install_lines, (
        f"cell 2 must not use subprocess.run for pip install; found: {install_lines}"
    )


def test_cell2_critical_uses_literal_pip_line_not_run_line_magic():
    """The CRITICAL pass must NOT pin numpy<2.0 — ultralytics 8.4 needs numpy>=2.0.

    Why: the user's re-test on commit 2fdff03 showed pip raising
    ResolutionImpossible because numpy<2.0 conflicts with ultralytics 8.4
    which needs numpy>=2.0. The %pip line SILENTLY swallowed the error
    (no exception, no exit code surfaced) and the cell printed '✓ installed'
    even though ultralytics was not on disk.

    The fix: drop the numpy pin entirely. Colab has numpy 2.5.1 pre-installed;
    ultralytics will use that.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell2 = _find_cell_by_comment(nb, "Install + clone + Roboflow key")
    assert cell2 is not None
    src = "".join(cell2['source'])
    # The numpy<2.0 pin is FORBIDDEN — causes ResolutionImpossible
    for forbidden in ("numpy<2.0", "numpy<2", "'numpy>=1.26,<2.0'", '"numpy>=1.26,<2.0"'):
        assert forbidden not in src, (
            f"the numpy<2.0 pin is forbidden ({forbidden!r} found) — ultralytics 8.4 "
            f"requires numpy>=2.0; the pin causes ResolutionImpossible that "
            f"%pip silently swallows"
        )
    # Must explicitly note that numpy is unpinned (so future readers know why)
    assert "unpinned" in src.lower() or "let ultralytics" in src.lower() or "no numpy pin" in src.lower(), (
        "the build script must document why numpy is unpinned (ultralytics needs >=2.0, "
        "Colab has 2.5.1 pre-installed)"
    )


def test_cell2_verifies_imports_with_fresh_load():
    """Cell 2 must force a fresh import (not just check sys.modules cache).

    Why: subprocess.run([sys.executable, ...]) in cell 8 spawns a fresh Python
    process that reads site-packages from disk. If %pip install wrote files
    correctly, subprocess WILL see them. But if a stale sys.modules entry
    masks the failure, the cell claims success. We force fresh import by
    removing the module from sys.modules first.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell2 = _find_cell_by_comment(nb, "Install + clone + Roboflow key")
    src = "".join(cell2['source'])
    # Must check __file__ or __path__ (proves real install, not stub)
    assert "__file__" in src or "__path__" in src, (
        "cell 2 must verify the imported module has __file__ or __path__ — "
        "a module with neither is a stub or namespace package, not a real install"
    )
    # Must remove from sys.modules before re-import (or other freshness mechanism)
    assert "sys.modules" in src, (
        "cell 2 must manipulate sys.modules to force a fresh import (not use cache)"
    )


def test_cell1_validates_cwd_before_subprocess():
    """Cell 1 must ensure the CWD is a real directory BEFORE any subprocess
    call. If the user opens the notebook with a stale CWD (a directory that
    was deleted between sessions — common in Colab), every `subprocess.run`
    will fail with `fatal: Unable to read current working directory` and the
    self-heal will report false negatives.

    The fix: Step 0 chdir to /content (which always exists in Colab) and
    every subprocess call has an explicit `cwd='/content'`.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell1 = _find_cell_by_comment(nb, "Runtime + env check")
    src = "".join(cell1["source"])
    # Step 0 must call os.getcwd() and fall back to /content if it raises
    assert "os.getcwd()" in src, "must probe CWD with os.getcwd()"
    assert "os.chdir('/content')" in src, "must fall back to /content if CWD is invalid"
    # All subprocess.run calls must have explicit cwd='/content'
    import re
    subprocess_calls = re.findall(r'subprocess\.run\([^)]*\)', src, re.DOTALL)
    assert len(subprocess_calls) >= 2, "must have at least 2 subprocess calls (clone + pull)"
    for i, call in enumerate(subprocess_calls):
        assert "cwd='/content'" in call or "cwd=\"/content\"" in call, (
            f"subprocess call #{i+1} must have explicit cwd='/content' — "
            f"otherwise a stale CWD will crash the call with "
            f"'Unable to read current working directory'. Call: {call[:120]}"
        )


def test_no_hardcoded_cuda_device():
    """No cell may hardcode `device='cuda:0'` or `device='cuda:N'` —
    hardcoding crashes on CPU-only Colab runtimes with
    `ValueError: Invalid CUDA 'device=0' requested`.
    The fix: use `pick_device()` which auto-detects cuda:0 vs cpu.
    """
    nb = json.loads(NOTEBOOK.read_text())
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        src = "".join(cell.get('source', []))
        # Block any literal "cuda:0" or "cuda:N" device argument
        for pattern in ["device='cuda:0'", "device=\"cuda:0\"", "device='cuda:1'", "device='cuda:2'"]:
            assert pattern not in src, (
                f"cell #{i} hardcodes {pattern!r} — must use pick_device() instead. "
                f"Otherwise CPU-only Colab runtimes crash with "
                f"'Invalid CUDA device=0 requested'."
            )


def test_pipeline_uses_pick_device():
    """The pipeline cell (cell 9) and visual analytics cell (cell 9-visual)
    must use `device=pick_device()` instead of hardcoded cuda:0.
    """
    nb = json.loads(NOTEBOOK.read_text())
    pipeline = _find_cell_by_comment(nb, "End-to-end pipeline")
    assert pipeline is not None, "could not find the pipeline cell"
    src = "".join(pipeline['source'])
    assert "device=pick_device()" in src, (
        "pipeline cell must use device=pick_device() (auto: cuda:0 or cpu)"
    )


def test_visual_uses_detect_not_infer():
    """The visual analytics cell must call `det.detect(...)`, not `det.infer(...)`.
    UltralyticsDetector exposes `detect`, not `infer`. The `infer` typo
    surfaced as `AttributeError: 'UltralyticsDetector' object has no attribute 'infer'`.
    """
    nb = json.loads(NOTEBOOK.read_text())
    visual = _find_cell_by_comment(nb, "Visual analytics")
    assert visual is not None, "could not find the visual analytics cell"
    src = "".join(visual['source'])
    assert "det.infer(" not in src, (
        "visual cell calls det.infer() but the method is named 'detect'. "
        "Fix: use det.detect(image_bgr) instead."
    )
    assert "det.detect(" in src, "visual cell must call det.detect(...)"


def test_visual_skips_cleanly_when_supervision_missing():
    """When supervision import fails (sv=None), the visual cell must skip
    cleanly (no AttributeError on sv.Detections.empty()). The contract:
    check _sv_ok first; if False, raise SystemExit(0) (graceful skip).
    """
    nb = json.loads(NOTEBOOK.read_text())
    visual = _find_cell_by_comment(nb, "Visual analytics")
    src = "".join(visual['source'])
    # Must check _sv_ok BEFORE any sv. usage in the with cell() block
    if_in_sv_ok = src.find("if not _sv_ok:")
    sv_detections_pos = src.find("sv.Detections")
    assert if_in_sv_ok != -1, (
        "visual cell must check `if not _sv_ok:` before using sv. "
        "Otherwise AttributeError on sv.Detections.empty() when supervision is missing."
    )
    assert if_in_sv_ok < sv_detections_pos, (
        f"`if not _sv_ok:` check (pos {if_in_sv_ok}) must come BEFORE "
        f"first sv. usage (pos {sv_detections_pos})"
    )
    # The skip path must raise SystemExit(0) for graceful handling
    assert "raise SystemExit(0)" in src, (
        "visual cell must `raise SystemExit(0)` to skip cleanly when supervision is missing"
    )


def test_cell8_detects_coco_fallback():
    """Cell 8 trains in the kernel using the bundled recycling data.

    Note: the COCO-fallback path was a feature of the old subprocess-based
    cell 8 (which called download_dataset.py). In the new in-kernel flow,
    cell 8 always uses the bundled data from data/sample/recycling_demo/ —
    no COCO fallback needed (and no network call).

    This test now verifies the in-kernel training uses the bundled data.yaml.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell8 = _find_cell_by_comment(nb, "Train YOLO26s")
    assert cell8 is not None
    src = "".join(cell8['source'])
    # Must reference the bundled data location
    assert "recycling_v3" in src or "data/raw" in src or "data.yaml" in src, (
        "cell 8 must reference the bundled data (data/raw/recycling_v3/ or data.yaml)"
    )
    # Must use YOLO() and model.train() in the kernel
    assert "YOLO(" in src and "model.train" in src, (
        "cell 8 must use YOLO() and model.train() in the kernel"
    )


def test_cell8_uses_in_kernel_training():
    """Cell 8 must train in the KERNEL, NOT via subprocess.run(scripts/train_yolo26.py).

    Why: subprocess.run spawns a fresh Python that may not have the kernel's
    installed packages on sys.path (Colab module registry quirk). In-kernel
    training is simpler, easier to debug, and uses the same T4.

    Pattern: YOLO() and model.train() called directly in the cell.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell8 = _find_cell_by_comment(nb, "Train YOLO26s")
    assert cell8 is not None
    src = "".join(cell8['source'])
    # Must use the in-kernel YOLO API directly
    assert "from ultralytics import YOLO" in src, (
        "cell 8 must import YOLO in the kernel (not via subprocess)"
    )
    assert "model.train(" in src, (
        "cell 8 must call model.train(...) in the kernel"
    )
    # Must NOT use the subprocess path (it's been buggy on Colab)
    install_lines = [
        line for line in src.split('\n')
        if "subprocess" in line and "scripts/train" in line
        and not line.lstrip().startswith("#")
    ]
    assert not install_lines, (
        f"cell 8 must not use subprocess for training; found: {install_lines}"
    )


def test_cell8_self_heals_missing_data():
    """Cell 8 must self-heal: if data.yaml is missing at the expected path,
    copy the bundled data automatically (don't just error out).

    Why: the user may run cells out of order, or the OLD download script
    may have left data in a stale format. Cell 8 should be robust — copy
    the bundled data itself if needed.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell8 = _find_cell_by_comment(nb, "Train YOLO26s")
    assert cell8 is not None
    src = "".join(cell8['source'])
    # Must check for data.yaml existence
    assert "data.yaml" in src and ".exists()" in src, (
        "cell 8 must check for data.yaml existence"
    )
    # Must reference the bundled data fallback path
    assert "data/sample/recycling_demo" in src or "BUNDLED" in src, (
        "cell 8 must know the bundled data path for self-heal"
    )
    # Must use copytree for the fallback
    assert "copytree" in src or "shutil.copy" in src, (
        "cell 8 must copy the bundled data when self-healing"
    )


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


def test_cell15_publish_uses_pip_magic_not_subprocess():
    """Cell 15 (publish) self-heals PyGithub. It MUST use %pip magic, NOT
    subprocess.check_call, for the same Colab-registry reason as cell 2.

    Symptom of regression: `subprocess.check_call([sys.executable, '-m', 'pip',
    'install', 'PyGithub', ...])` returns 0 but the subsequent
    `from github import Github` in the same try/except raises ImportError
    because Colab's IPython module registry wasn't updated.
    """
    nb = json.loads(NOTEBOOK.read_text())
    cell15 = _find_cell_by_comment(nb, "Publish to GitHub Release")
    assert cell15 is not None, "could not find Cell 15 (Publish)"
    src = "".join(cell15["source"])
    # Must use %pip magic for the self-heal install
    assert "run_line_magic('pip'" in src, (
        "cell 15 self-heal must use %pip magic — subprocess.check_call will "
        "silently fail to update Colab's kernel module registry"
    )
    # Must NOT use subprocess for the install (only legitimate use is git ops)
    install_lines = [
        line for line in src.split('\n')
        if "subprocess" in line and "pip" in line
        and not line.lstrip().startswith("#")
    ]
    assert not install_lines, (
        f"cell 15 must not use subprocess for pip install; found: {install_lines}"
    )


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
    """Cell 9.6 (production path) must use Roboflow Inference, with graceful fallback."""
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


def test_production_path_cell_no_systemexit_inside_except():
    """Regression (Aug 19 2026): cell 9.6 raised `SystemExit(0)` inside the
    `except ImportError` block. IPython's traceback formatter (ultratb.py) has
    a bug that crashes with
        TypeError: object of type 'NoneType' has no len()
    when SystemExit is raised inside an except block (during
    `find_recursion()`). The fix: use a flag (`_inference_ok = True/False`)
    and gate the rest of the cell body with `if not _inference_ok: pass / else: ...`.
    No `raise SystemExit` may appear inside the except ImportError block.
    """
    nb = json.loads(NOTEBOOK.read_text())
    prod = _find_cell_by_comment(nb, "Roboflow Inference")
    assert prod is not None, "could not find the production path cell"
    src = "".join(prod["source"])

    # Find the except ImportError block boundaries.
    except_idx = src.find("except ImportError")
    assert except_idx != -1, "no `except ImportError` block — pre-fix contract broken"

    # Find the end of the except block: the first `if not _inference_ok:` line
    # (which is the very next line after the except block ends) — or any line
    # whose stripped form is at the same indent as the except line. We use the
    # `if not _inference_ok:` heuristic since that immediately follows the
    # except block in the contract.
    after_except_idx = src.find("if not _inference_ok:", except_idx)
    assert after_except_idx != -1, (
        "no `if not _inference_ok:` gate after the except block — "
        "the flag-pattern contract is broken"
    )

    except_block = src[except_idx:after_except_idx]
    # The except block must NOT contain any `raise SystemExit` (any code) —
    # this is the exact bug we are guarding against.
    assert "raise SystemExit" not in except_block, (
        "`raise SystemExit` is forbidden inside the `except ImportError` block "
        "of cell 9.6 — IPython's ultratb.py crashes with "
        "`TypeError: object of type 'NoneType' has no len()` when SystemExit "
        "is raised during traceback formatting inside an except block. "
        "Use the `_inference_ok` flag pattern instead."
    )

    # The flag pattern must be present (try sets _inference_ok = True,
    # the else branch contains the rest of the cell).
    assert "_inference_ok = True" in src, (
        "must set `_inference_ok = True` in the try block on successful import"
    )
    assert "_inference_ok = False" in src, (
        "must initialize `_inference_ok = False` before the try (defensive default)"
    )
    assert "else:" in src, "must gate the production-path body with `if/else`"

    # The model-loading code must live under the `else:` branch (i.e., only
    # runs when inference import succeeded). This is the structural guarantee
    # that we don't try to call `get_model(...)` when the import failed.
    else_idx = src.find("else:")
    get_model_idx = src.find("get_model(model_id=model_id)")
    assert else_idx != -1 and get_model_idx != -1, "sanity: missing else / get_model"
    assert get_model_idx > else_idx, (
        "`get_model(...)` must appear AFTER the `else:` gate so the model is "
        "only loaded when the import succeeded."
    )


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
    """The hero cell must use the render_hero helper for a rich front door.

    The hero is NOT pinned to position 1 — it must come AFTER the self-heal
    cell (which clones the repo and makes colab_session importable). See
    test_first_local_import_after_self_heal for the ordering invariant.
    """
    nb = json.loads(NOTEBOOK.read_text())
    hero = None
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        if "render_hero" in "".join(cell.get("source", [])):
            hero = cell
            break
    assert hero is not None, "no cell uses render_hero — the hero is missing"
    src = "".join(hero["source"])
    assert hero["cell_type"] == "code", "hero must be a code cell (HTML renders in output)"
    assert "tinkr-hero" not in src, "hero should CALL the helper, not embed raw HTML"


def test_hero_cell_comes_after_self_heal():
    """Regression: the cell that uses `render_hero` (the hero) must come
    AFTER the cell that does the self-heal (git clone + sys.path.insert).

    The original bug: the hero cell was at position 1, importing
    colab_session before the self-heal at position 3. On a fresh Colab
    open the hero errored immediately and every downstream cell cascaded.
    Caught live on 2026-08-19 from a user run; this test pins the
    ordering invariant.

    Note: the self-heal cell itself imports colab_session (wrapped in
    try/except), so the invariant is specifically about the cell that
    uses `render_hero` — that's the cell that needs colab_session to
    already be importable.
    """
    nb = json.loads(NOTEBOOK.read_text())
    heal_idx: int | None = None
    hero_idx: int | None = None
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if heal_idx is None and "subprocess.run" in src and "git clone" in src:
            heal_idx = i
        if hero_idx is None and "render_hero" in src:
            hero_idx = i
    assert heal_idx is not None, (
        "no self-heal cell found (no code cell has both subprocess.run and "
        "git clone). The self-heal must clone the repo so colab_session "
        "is importable in subsequent cells."
    )
    assert hero_idx is not None, (
        "no cell uses render_hero — the demo is missing its visual front door"
    )
    assert heal_idx < hero_idx, (
        f"hero cell (UI position {hero_idx + 1}) comes BEFORE the self-heal "
        f"(UI position {heal_idx + 1}). On a fresh Colab open the hero will "
        f"fail with ModuleNotFoundError because colab_session isn't "
        f"importable yet, and every downstream cell will cascade. Move the "
        f"self-heal (the cell with both subprocess.run and git clone) to "
        f"BEFORE the cell that calls render_hero."
    )


def test_self_heal_recovers_stale_repo():
    """Regression: the self-heal must recover when the repo exists with
    a STALE colab_session.py (from a previous Colab session that was
    clobbered by a "Save to GitHub" round-trip).

    The original bug (2026-08-19, second clobber round): the self-heal
    branched on whether `colab_session.py` EXISTS in the repo, and only
    did `git pull --rebase` when the file was MISSING. When the file
    was present (because a prior clobbered session left a stale copy),
    the self-heal just printed "import failed" and re-raised — it never
    pulled, so the user was stuck on the old version. Downstream cell 13
    (visual analytics) then couldn't import `supervision` because the
    install cell never ran.

    The fix: the self-heal's `if has_file:` branch does `git pull --rebase`
    AND the post-elif re-try has a "nuke + re-clone" last-resort fallback.

    This test pins BOTH invariants:
    1. The self-heal source has `git pull --rebase` in the helpers
       (called when the file exists but import failed, OR when the repo
       is valid but colab_session.py is missing).
    2. The self-heal has a final nuke + re-clone path (in `_do_clone()`)
       for when the pull doesn't fix the import.
    """
    nb = json.loads(NOTEBOOK.read_text())
    heal_src: str | None = None
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "subprocess.run" in src and "git clone" in src and "has_file" in src:
            heal_src = src
            break
    assert heal_src is not None, (
        "no self-heal cell found (must contain subprocess.run, git clone, "
        "and has_file — the fingerprint of the self-heal)"
    )
    # Invariant 1: at least one git pull --rebase in the source
    pull_count = heal_src.count("'git', '-C', REPO, 'pull', '--rebase'")
    assert pull_count >= 1, (
        f"self-heal must do `git pull --rebase` at least once to recover "
        f"from a stale repo, found {pull_count} occurrences"
    )
    # Invariant 2: at least one shutil.rmtree (the nuke fallback)
    rmtree_count = heal_src.count("shutil.rmtree")
    assert rmtree_count >= 1, (
        "self-heal must have a `shutil.rmtree` nuke + re-clone "
        "fallback for when the pull doesn't fix the import"
    )
    # Invariant 3: at least one `git clone` (for the nuke + re-clone path)
    clone_count = heal_src.count("'git', 'clone', REPO_URL, REPO")
    assert clone_count >= 1, (
        "self-heal must do at least one `git clone` (for the fresh-clone path)"
    )


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


def test_cell_numbering_is_contiguous():
    """Cell numbering in the §2 walkthrough must be contiguous — no gaps
    like 9.5 → 9.7 (the original 9.6 slot was RF-DETR-S, deferred to v2.0).
    Users seeing '9.5, 9.7' in the Colab UI think a cell is missing.
    """
    nb = json.loads(NOTEBOOK.read_text())
    # Find all Cell N or Cell N.M comments in code cells
    import re
    cell_numbers = []
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        first_line = ''.join(cell.get('source', [])).splitlines()[0] if cell.get('source') else ''
        # Match patterns like "Cell 9", "Cell 9.5", "Cell 1b"
        m = re.search(r'Cell\s+(\d+)(?:\.(\d+))?[a-z]?', first_line)
        if m:
            major = int(m.group(1))
            minor = int(m.group(2)) if m.group(2) else 0
            cell_numbers.append((major, minor, first_line))
    # We expect 9.5 → 9.6 (no 9.7). Specifically: no (9, 7, ...) in cell_numbers.
    nine_seven = [c for c in cell_numbers if c[0] == 9 and c[1] == 7]
    assert not nine_seven, (
        f"Cell 9.7 still present — should be renumbered to 9.6 to keep "
        f"the §2 walkthrough contiguous. Found: {nine_seven[0][2]!r}"
    )
    # We expect 9.5 AND 9.6 to both be present
    nine_five = [c for c in cell_numbers if c[0] == 9 and c[1] == 5]
    nine_six = [c for c in cell_numbers if c[0] == 9 and c[1] == 6]
    assert nine_five, "Cell 9.5 (visual analytics) must exist"
    assert nine_six, "Cell 9.6 (production path) must exist after renumbering"


def test_bundled_recycling_data_exists():
    """The bundled recycling demo data must be present in the repo so
    the demo can train on REAL recycling data even when Roboflow S3 is
    broken. The data is at data/sample/recycling_demo/.
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    bundled = repo_root / "data" / "sample" / "recycling_demo"
    assert bundled.exists(), (
        f"Bundled recycling demo data missing at {bundled}. "
        f"This is the offline fallback that lets the demo train on "
        f"real recycling data even when Roboflow S3 is broken."
    )
    assert (bundled / "data.yaml").exists(), "data.yaml missing in bundled data"
    assert (bundled / "train" / "images").exists(), "train/images missing"
    assert (bundled / "val" / "images").exists(), "val/images missing"
    # Sanity: must have actual images, not just empty dirs
    train_imgs = list((bundled / "train" / "images").glob("*.jpg"))
    val_imgs = list((bundled / "val" / "images").glob("*.jpg"))
    assert len(train_imgs) >= 50, f"need >=50 train images, got {len(train_imgs)}"
    assert len(val_imgs) >= 10, f"need >=10 val images, got {len(val_imgs)}"


def test_download_script_uses_bundled_data_first():
    """download_dataset.py must use the bundled data as the FIRST source
    (always works) and only fall back to Roboflow if the user has an API
    key. NEVER fall back to COCO pretrained — that was the silent-failure
    bug fixed at f48afe9.
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    script = (repo_root / "scripts" / "download_dataset.py").read_text()
    assert "BUNDLED_DEMO" in script, "must reference BUNDLED_DEMO path"
    assert "use_bundled_demo" in script, "must call use_bundled_demo first"
    # Must NOT have any COCO pretrained fallback
    assert "fallback_to_coco_pretrained" not in script, (
        "must NOT fall back to COCO pretrained — the user explicitly "
        "asked for real recycling training with no COCO fallback."
    )
    assert "COCO pretrained" not in script or script.count("COCO") <= 2, (
        "COCO pretrained should not be referenced in the download path"
    )


def test_train_script_accepts_bundled_demo_source():
    """The train script must accept 'bundled_demo' as a valid source
    (in addition to 'roboflow' and 'ultralytics_coco_pretrained'). The
    bundled data path is the new offline-fallback that lets the demo
    train on real recycling data without depending on Roboflow.
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    script = (repo_root / "scripts" / "train_yolo26.py").read_text()
    assert "bundled_demo" in script, (
        "train_yolo26.py must recognize 'bundled_demo' as a valid source"
    )
    # The source-not-recognized error must allow bundled_demo
    assert 'meta.get("source") not in ("roboflow", "bundled_demo")' in script, (
        "the source check must accept both 'roboflow' and 'bundled_demo'"
    )


# --- Subprocess self-heal (Aug 2026) ---------------------------------------

class TestSubprocessColabSelfHeal:
    """Scripts invoked as subprocesses from the notebook (cell 8 used to
    train + download) must self-heal Colab's site-packages path. The kernel
    (where `%pip install` writes) DOES have it on sys.path, but a fresh
    subprocess Python often does NOT — and `from ultralytics import YOLO`
    raises ModuleNotFoundError.

    Note (Aug 2026): cell 8 NOW trains in the KERNEL (no subprocess). The
    self-heal is still needed for `train_yolo26.py` and `download_dataset.py`
    in case they're called externally (e.g. from a CI script or `make train`).
    """

    def test_train_script_self_heal_present(self):
        from pathlib import Path
        script = (REPO_ROOT / "scripts" / "train_yolo26.py").read_text()
        # Must add /usr/local/lib/python3.12/dist-packages to sys.path BEFORE
        # any other import. The literal Python check: the string must appear
        # before the first `import` (or `from`) line that imports anything
        # other than stdlib self-heal stuff.
        assert "/usr/local/lib/python3.12/dist-packages" in script, (
            "train_yolo26.py must explicitly add Colab's site-packages to "
            "sys.path at the top — kernel has it, fresh subprocess Python does not"
        )
        # Must also run site.main() to re-process .pth files
        assert "site.main()" in script, (
            "train_yolo26.py must call site.main() to re-process .pth files"
        )

    def test_download_script_self_heal_present(self):
        from pathlib import Path
        script = (REPO_ROOT / "scripts" / "download_dataset.py").read_text()
        assert "/usr/local/lib/python3.12/dist-packages" in script, (
            "download_dataset.py must self-heal Colab's site-packages path"
        )
        assert "site.main()" in script, (
            "download_dataset.py must call site.main() to re-process .pth files"
        )
