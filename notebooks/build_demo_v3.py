#!/usr/bin/env python3
"""Build script for notebooks/demo_v3.ipynb.

This is the v3 notebook — a clean, focused rebuild of the conveyor-perception
demo for the EverestLabs AI Engineer interview. Demo v1 and v2 are kept in
the repo as knowledge sources / reference; v3 is the active, fresh build.

v3 design goals (locked Aug 22 2026, with Ronie Joseph):
  - 12-15 cells max (no half-numbers, no §-prefix nonsense)
  - Each cell has one job; cells do what they say
  - v2's colab_session.py is the SessionState singleton (re-used as-is)
  - Cells are tested one-at-a-time: paste → run → screenshot → next

Cell narrative (planned):
  0.  Title             (markdown, 1-screen pitch)
  1.  Runtime + env     (GPU, disk, RAM, Python)
  2.  Install + clone   (minimal open-source deps)
  3.  State + toggles   (SessionState + module toggle UI)
  4.  Load abstractions (4 framework abstractions)
  5.  Load modules      (7+1 JD modules)
  6.  Data registry     (what's on disk?)
  7.  Data download     (idempotent)
  8.  Train             (cached on re-run)
  9.  Compare           (side-by-side metrics)
  10. Pipeline          (Detector→Tracker→Drift→Triage→Maintenance)
  11. Visual Analytics  (the impressive part)
  12. Production path   (Roboflow Inference, library mode)
  13. Triage + monitor  (queue + robustness + dashboard)
  14. Coach + summary   (Gemini review + downloadable session log)
  15. Comparison        (T4 vs EverestLabs target)
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Cell builder helpers (same pattern as v2 — keep it simple)
# ---------------------------------------------------------------------------

def code(*lines: str) -> dict[str, Any]:
    """Build a code cell from individual source lines (1 string per line)."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


def md(*lines: str) -> dict[str, Any]:
    """Build a markdown cell from individual source lines."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in lines],
    }


CELLS: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Cell 0 (markdown): Title — 1-screen pitch
# ---------------------------------------------------------------------------
CELLS.append(md(
    "# Conveyor perception",
    "",
    "**An end-to-end industrial CV pipeline on a free T4:** real recycling data → trained model → live detection → drift monitoring → triage decisions.",
    "",
    "- **Runtime:** Google Colab T4 (free tier, ~12h cap).  ",
    "- **Data:** bundled 4-class recycling set (CC BY 4.0).  ",
    "- **Model:** YOLO26s, trained in-kernel, cached on re-run.  ",
    "- **Goal:** show the loop — train → infer → drift → triage → maintain — on real data, in <5 minutes.",
))


# ---------------------------------------------------------------------------
# Cell 1 (code): Runtime + env check
# ---------------------------------------------------------------------------
CELLS.append(code(
    "# --- Cell 1: Runtime + env check ---",
    "import os, sys, json, platform",
    "from pathlib import Path",
    "",
    "# --- 1. Colab or local? ---",
    "IN_COLAB = 'google.colab' in sys.modules",
    "print(f'  Runtime: {\"Google Colab\" if IN_COLAB else \"Local (\" + platform.node() + \")\"}')",
    "",
    "# --- 2. Python + key libs (skip import if missing) ---",
    "print(f'  Python: {sys.version.split()[0]}  ({sys.executable.split(\"/\")[-1]})')",
    "for mod in ['numpy', 'torch', 'ultralytics', 'supervision', 'roboflow']:",
    "    try:",
    "        m = __import__(mod)",
    "        v = getattr(m, '__version__', '?')",
    "        print(f'  {mod:14s} {v}')",
    "    except ImportError:",
    "        print(f'  {mod:14s} — not installed yet (cell 2 will install)')",
    "",
    "# --- 3. GPU (or warn if CPU-only) ---",
    "_gpu = 'unknown'",
    "try:",
    "    import torch",
    "    _gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'",
    "except Exception:",
    "    pass",
    "print(f'  GPU:    {_gpu}')",
    "",
    "# --- 4. Disk + RAM ---",
    "_disk_free = '?'",
    "try:",
    "    import shutil",
    "    _u = shutil.disk_usage('/')",
    "    _disk_free = f'{_u.free / 1e9:.1f} GB free of {_u.total / 1e9:.1f} GB'",
    "except Exception:",
    "    pass",
    "print(f'  Disk:   {_disk_free}')",
    "",
    "# --- 5. Locate the repo (for local runs) — Colab gets cloned by cell 2 ---",
    "REPO = Path('/content/conveyor-perception' if IN_COLAB else '.').resolve()",
    "if not IN_COLAB:",
    "    # Walk up until we find the repo root (contains pyproject.toml)",
    "    while not (REPO / 'pyproject.toml').exists() and REPO != REPO.parent:",
    "        REPO = REPO.parent",
    "print(f'  Repo:   {REPO}{\" (will be cloned here by cell 2)\" if IN_COLAB else \"\"}')",
    "",
    "# NOTE: colab_session + the state singleton are intentionally NOT used here.",
    "# Cell 1 runs BEFORE the clone (cell 2), so the repo isn't on disk yet — any",
    "# `import colab_session` would crash with ModuleNotFoundError. State init is",
    "# deferred to cell 3, which runs after the clone + install are done. (Aug 22 2026)",
    "print()",
    "print('  ✓ env check done.  Next: cell 2 (clone + install).')",
))


# ---------------------------------------------------------------------------
# Build the notebook
# ---------------------------------------------------------------------------

def build(out_path: Path) -> int:
    """Write demo_v3.ipynb. Returns the number of cells written."""
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": __import__("sys").version.split()[0],
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    return len(CELLS)


if __name__ == "__main__":
    import sys as _sys
    out = Path(_sys.argv[1] if len(_sys.argv) > 1 else "demo_v3.ipynb")
    n = build(out)
    print(f"Wrote {out} with {n} cells")
