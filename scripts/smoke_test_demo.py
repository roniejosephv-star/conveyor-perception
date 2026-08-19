"""Smoke test for notebooks/demo_v2.ipynb.

A 5-second pre-call check that the builder produced a structurally
valid notebook. Catches:
- JSON corruption
- Cell-count drift
- Python syntax errors in any code cell
- Obvious import shape issues

This is NOT a runtime test. It does not execute the cells, does not
import the heavy libraries (ultralytics, torch, fastmcp, inference),
and does not need a GPU. For an end-to-end runtime check, use
docs/LIVE_DEMO_CHECKLIST.md §2 (the full Colab run-all).

Usage:
    python scripts/smoke_test_demo.py
    python scripts/smoke_test_demo.py path/to/other.ipynb
    python -c "from scripts.smoke_test_demo import run_smoke_test; print(run_smoke_test())"

Exit code 0 on success, 1 on failure.
"""
from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK = REPO_ROOT / "notebooks" / "demo_v2.ipynb"
EXPECTED_CELL_COUNT = 29


@dataclass
class SmokeResult:
    """Outcome of a smoke test run.

    Attributes:
        path: The notebook that was checked.
        cell_count: Actual cell count found.
        expected_cell_count: What we expected.
        code_cells: Number of code cells.
        markdown_cells: Number of markdown cells.
        parse_errors: List of (cell_index_1based, error_message) for cells that
            failed `ast.parse`. Empty if all good.
        warnings: Non-fatal issues (e.g., code cell with no imports).
        ok: True if and only if there are no parse errors and the cell count
            matches.
    """

    path: Path
    cell_count: int
    expected_cell_count: int
    code_cells: int
    markdown_cells: int
    parse_errors: list[tuple[int, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ok: bool = False

    def summary(self) -> str:
        """Human-readable one-line summary."""
        status = "OK" if self.ok else "FAIL"
        return (
            f"[{status}] {self.path.name}: "
            f"{self.cell_count} cells "
            f"({self.code_cells} code + {self.markdown_cells} md), "
            f"{len(self.parse_errors)} parse errors, "
            f"{len(self.warnings)} warnings"
        )


def parse_notebook(path: Path) -> dict[str, Any]:
    """Load and structurally validate the notebook JSON.

    Raises FileNotFoundError or json.JSONDecodeError on bad input.
    """
    if not path.exists():
        raise FileNotFoundError(f"notebook not found: {path}")
    data = json.loads(path.read_text())
    if "cells" not in data:
        raise ValueError(f"notebook has no 'cells' key: {path}")
    if "nbformat" not in data:
        raise ValueError(f"notebook has no 'nbformat' key: {path}")
    return data


def check_cell_count(nb: dict[str, Any], expected: int) -> list[str]:
    """Return a list of error messages (empty if cell count matches)."""
    actual = len(nb["cells"])
    if actual != expected:
        return [
            f"cell count drift: expected {expected}, got {actual} "
            f"(if you added/removed a cell, update build_demo_v2.py's "
            f"assertion and test_cell_count_is_29)"
        ]
    return []


def _extract_source(cell: dict[str, Any]) -> str:
    """Concatenate the cell's source lines into a single string."""
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def check_code_cells_parse(nb: dict[str, Any]) -> list[tuple[int, str]]:
    """ast.parse every code cell. Return (cell_index_1based, error) for failures."""
    errors: list[tuple[int, str]] = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = _extract_source(cell)
        if not src.strip():
            # Empty code cell is allowed (e.g., placeholder for a future run).
            continue
        try:
            ast.parse(src)
        except SyntaxError as exc:
            errors.append((i + 1, f"line {exc.lineno}: {exc.msg}"))
    return errors


def _cell_has_imports(cell: dict[str, Any]) -> bool:
    """True if the cell's source contains at least one import statement."""
    src = _extract_source(cell)
    if not src.strip():
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True
    return False


def check_code_cells_have_imports(nb: dict[str, Any]) -> list[str]:
    """Warn (not fail) on code cells that have no imports. Often a sign the
    builder emitted a cell that lost its imports during a refactor.
    """
    warnings: list[str] = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = _extract_source(cell)
        if not src.strip():
            continue
        if not _cell_has_imports(cell):
            warnings.append(f"cell {i + 1}: code cell has no import statements")
    return warnings


def run_smoke_test(
    path: Path = DEFAULT_NOTEBOOK,
    expected_cells: int = EXPECTED_CELL_COUNT,
) -> SmokeResult:
    """Run the full smoke test and return a structured result.

    The result is always returned (does not raise). Inspect `.ok` to
    determine pass/fail, or call `.summary()` for a one-liner.
    """
    try:
        nb = parse_notebook(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return SmokeResult(
            path=path,
            cell_count=0,
            expected_cell_count=expected_cells,
            code_cells=0,
            markdown_cells=0,
            parse_errors=[(0, f"could not parse notebook: {exc}")],
            ok=False,
        )

    code_cells = sum(1 for c in nb["cells"] if c.get("cell_type") == "code")
    markdown_cells = sum(1 for c in nb["cells"] if c.get("cell_type") == "markdown")

    count_errors = check_cell_count(nb, expected_cells)
    parse_errors = check_code_cells_parse(nb)
    import_warnings = check_code_cells_have_imports(nb)

    # Cell-count errors are reported as parse_errors (1-based cell 0 = notebook-level)
    all_errors: list[tuple[int, str]] = [(0, e) for e in count_errors] + parse_errors

    return SmokeResult(
        path=path,
        cell_count=len(nb["cells"]),
        expected_cell_count=expected_cells,
        code_cells=code_cells,
        markdown_cells=markdown_cells,
        parse_errors=all_errors,
        warnings=import_warnings,
        ok=not all_errors,
    )


def _format_report(result: SmokeResult) -> str:
    """Build the human-readable report printed to stdout."""
    lines = [
        f"Smoke test: {result.path}",
        f"  Cells: {result.cell_count} (expected {result.expected_cell_count})",
        f"  Code cells: {result.code_cells}",
        f"  Markdown cells: {result.markdown_cells}",
    ]
    if result.parse_errors:
        lines.append(f"  Parse errors ({len(result.parse_errors)}):")
        for cell_idx, msg in result.parse_errors:
            lines.append(f"    cell {cell_idx}: {msg}")
    if result.warnings:
        lines.append(f"  Warnings ({len(result.warnings)}):")
        for w in result.warnings:
            lines.append(f"    {w}")
    lines.append("")
    lines.append(result.summary())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    args = list(sys.argv[1:] if argv is None else argv)
    path = Path(args[0]) if args else DEFAULT_NOTEBOOK
    result = run_smoke_test(path=path)
    print(_format_report(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
