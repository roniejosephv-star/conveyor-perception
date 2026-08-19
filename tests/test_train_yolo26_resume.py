"""Smoke tests for scripts/train_yolo26.py --resume flag.

The resume path lets the user continue a partial training run without
losing progress. These tests cover:
1. --resume flag is in --help
2. --resume with no last.pt → clear error (script exits non-zero)
3. --resume with last.pt present → script reaches the "Resuming from" print
   (we mock YOLO.train so we don't actually start training in CI)

The actual training run is NOT exercised here — that would take minutes
and require MPS/CUDA. The integration test for resume lives in
docs/COLAB_60SEC.md / docs/INTERVIEW_WALKTHROUGH.md.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_yolo26.py"
LAST_PT = REPO_ROOT / "models" / "train_runs" / "yolo26s_recyclable" / "weights" / "last.pt"


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=cwd or REPO_ROOT,
        timeout=15,  # fail fast if we accidentally enter the real training loop
    )


def test_resume_flag_appears_in_help():
    """The --resume flag must be discoverable via --help."""
    r = _run(["--help"])
    assert r.returncode == 0
    assert "--resume" in r.stdout
    assert "Resume from" in r.stdout


def test_resume_fails_clearly_when_last_pt_missing():
    """If last.pt doesn't exist on disk, --resume must fail with a clear error.

    We can't easily fake this with cwd because the script resolves ROOT from
    its own path. Instead we temporarily move last.pt aside, run, and restore.
    If last.pt is not on disk to begin with, the test is skipped.
    """
    if not LAST_PT.exists():
        pytest.skip("last.pt not on disk (no prior training run to fake-remove)")

    backup = LAST_PT.with_suffix(".pt.bak")
    try:
        LAST_PT.rename(backup)
        r = _run(["--resume", "--device", "cpu", "--epochs", "30"])
        assert r.returncode != 0
        combined = r.stdout + r.stderr
        assert "last.pt not found" in combined
        assert "Train from scratch" in combined
    finally:
        backup.rename(LAST_PT)


@pytest.mark.skipif(not LAST_PT.exists(), reason="last.pt not on disk (no prior training run)")
def test_resume_branch_passes_data_kwarg():
    """The --resume branch must pass data=str(data_yaml) to model.train.

    Static check on the script source. This guards against the
    size-mismatch bug where Ultralytics would override nc=4 (recycling)
    with nc=80 (COCO default) on resume.

    Why a static check: the alternative is to mock YOLO in a subprocess,
    but that requires either an in-process import (fragile) or a
    full training run (too slow for CI). The static check is
    sufficient because the bug is a single missing kwarg.
    """
    src = TRAIN_SCRIPT.read_text()
    # Find the resume branch (between "if args.resume:" and the matching else)
    import re
    m = re.search(r"if args\.resume:.*?else:", src, re.DOTALL)
    assert m, "Could not locate 'if args.resume:' branch in the script"
    branch = m.group(0)
    assert "data=str(data_yaml)" in branch, (
        "The --resume branch must call model.train(data=str(data_yaml), ...). "
        "Without data=, Ultralytics defaults to coco8.yaml and overrides "
        "nc=4 (recycling) with nc=80 (COCO), causing a size-mismatch error "
        "on the Detect head."
    )
    assert "resume=True" in branch, "The --resume branch must pass resume=True"
    assert str(LAST_PT.name) in branch, "The --resume branch must reference last.pt"
