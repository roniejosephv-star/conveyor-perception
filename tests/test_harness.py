"""Tests for the HARNESS.md handoff document.

The HARNESS.md is the lifeline for any new agent that picks up this project.
It must always exist, must always have the current state, and must always
include the project-context sections (rules, architecture, what-to-do-next).

If this test ever fails, the HARNESS.md has been deleted or gutted — restore
it before any new work begins.
"""
from pathlib import Path

HARNESS = Path(__file__).parent.parent / "HARNESS.md"


def test_harness_exists():
    """HARNESS.md must exist at the repo root."""
    assert HARNESS.exists(), f"HARNESS.md missing at {HARNESS} — restore it before any new work"


def test_harness_has_tldr():
    """HARNESS.md must have a TL;DR section at the top so a new agent can read it first."""
    text = HARNESS.read_text()
    assert "## TL;DR" in text, "HARNESS.md must have a '## TL;DR' section at the top"


def test_harness_has_current_state():
    """HARNESS.md must have a 'Current State' section with the current commit count + test count."""
    text = HARNESS.read_text()
    assert "## Current State" in text, "HARNESS.md must have a '## Current State' section"
    # Must mention the current test count (245 as of 2026-08-19, post doc-sync)
    assert "245" in text, "HARNESS.md must mention the current test count"
    # Must mention the current cell count (29 as of 2026-08-19)
    assert "29" in text, "HARNESS.md must mention the current notebook cell count"


def test_harness_has_what_to_do_next():
    """HARNESS.md must have a 'What To Do Next' section with concrete options."""
    text = HARNESS.read_text()
    assert "## What To Do Next" in text, "HARNESS.md must have a '## What To Do Next' section"


def test_harness_has_project_context():
    """HARNESS.md must include the project context (user rules, target job, constraints)."""
    text = HARNESS.read_text()
    # The user rules that must never be violated
    assert "Tinkr" in text, "HARNESS.md must include the 'never name Tinkr' rule"
    assert "EverestLabs" in text, "HARNESS.md must mention the target company"
    assert "M4" in text, "HARNESS.md must include the M4-from-interview rule"
    assert "AMP Robotics" in text, "HARNESS.md must include the no-competitor-names rule"


def test_harness_has_architecture_map():
    """HARNESS.md must include the directory tree so a new agent knows the layout."""
    text = HARNESS.read_text()
    assert "## Project Architecture" in text, "HARNESS.md must have a '## Project Architecture' section"
    # Must show the key directories
    assert "src/conveyor_perception/" in text, "HARNESS.md must show the src layout"
    assert "notebooks/build_demo_v2.py" in text, "HARNESS.md must highlight the builder as source of truth"
    assert ".github/workflows/optimize.yml" in text, "HARNESS.md must highlight the optimization loop workflow"


def test_harness_has_handoff_notes():
    """HARNESS.md must have a 'Handoff Notes' section with concrete dos and don'ts."""
    text = HARNESS.read_text()
    assert "## Handoff Notes" in text, "HARNESS.md must have a '## Handoff Notes' section"
    # The single most important rule: never hand-edit the .ipynb
    assert "build_demo_v2.py" in text and "never" in text.lower(), \
        "HARNESS.md must warn against hand-editing the .ipynb"
