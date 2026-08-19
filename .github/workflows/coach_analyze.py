"""Coach analysis — runs in the GitHub Action on each v0.0.* release.

Reads the new session log + the previous one (if any), asks Gemini to
suggest ONE concrete improvement, and writes the result to
/tmp/coach_suggestion.json. The Action then opens a PR with the change.

Exit codes:
  0 — success, suggestion written
  1 — no GEMINI_API_KEY configured
  2 — Gemini call failed
  3 — Coach decided no action is needed
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

RUNS_DIR = Path("/tmp/runs")
SUGGESTION_PATH = Path("/tmp/coach_suggestion.json")


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"⚠ failed to read {path}: {e}")
        return None


def _diff_summary(prev: dict | None, curr: dict) -> str:
    """One-paragraph diff between two session logs. Used in the prompt."""
    if prev is None:
        return "No previous run. This is the first session."

    def _metrics(d: dict) -> dict:
        return d.get("metrics", {})

    def _errors(d: dict) -> int:
        return len(d.get("errors", []))

    prev_m = _metrics(prev)
    curr_m = _metrics(curr)

    parts = []
    for key in sorted(set(prev_m) | set(curr_m)):
        p, c = prev_m.get(key, "n/a"), curr_m.get(key, "n/a")
        if p != c:
            parts.append(f"  {key}: {p} → {c}")
    if not parts:
        parts.append("  (no metric changes)")

    diff = textwrap.dedent(
        f"""
        Session diff:
        - errors: {_errors(prev)} → {_errors(curr)}
        - metrics:
        {chr(10).join(parts)}
        - toggles:
            prev: {prev.get('toggles', {})}
            curr: {curr.get('toggles', {})}
        """
    )
    return diff


def _build_prompt(curr: dict, prev: dict | None) -> str:
    return textwrap.dedent(
        f"""
        You are the Conveyor Perception Coach. Review the latest session
        log of a recycling-line perception stack and propose ONE concrete,
        actionable code change to improve the next run.

        The current session:
        {json.dumps(curr, indent=2)[:4000]}

        {_diff_summary(prev, curr)}

        Rules:
        1. Propose at most ONE change. Quality over quantity.
        2. The change must be a real, specific code edit. Vague
           "investigate X" suggestions are not acceptable.
        3. Pick from: a config tweak (batch size, imgsz, threshold),
           a doc fix (typo in a docstring), a missing test, a
           refactor (extract a magic number into a constant), or a
           new metric to track.
        4. The change must NOT alter public API surface. Don't rename
           `Detector` or change `MultitaskPipeline.step()`'s signature.
        5. If the previous run had no errors and the current run had
           no errors, AND the metrics are unchanged, respond with
           exactly the string "NO_ACTION".

        Output format (strict JSON, no prose):
        {{
          "reason": "<one-sentence why this change matters>",
          "file_path": "<relative path, e.g. src/conveyor_perception/core/drift_monitor.py>",
          "old_snippet": "<3-10 lines of code from the current file>",
          "new_snippet": "<the replacement code>",
          "test_change": "<optional: same shape but for the test file>"
        }}

        Respond with JSON only. No markdown. No preamble.
        """
    ).strip()


def _parse_gemini_json(text: str) -> dict | None:
    """Gemini sometimes wraps JSON in ```json ... ```. Strip + parse."""
    text = text.strip()
    if text.startswith("```"):
        # Find the JSON block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        text = text[start:end]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("⚠ GEMINI_API_KEY not set — skipping Coach analysis")
        return 1

    curr = _load_json(RUNS_DIR / "current.json")
    if curr is None:
        print(f"✗ current.json not found in {RUNS_DIR}")
        return 2

    prev = _load_json(RUNS_DIR / "previous.json")

    import google.generativeai as genai  # type: ignore[import-not-found]

    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel("gemini-2.0-flash")
    prompt = _build_prompt(curr, prev)
    print(f"Calling Gemini with {len(prompt)} chars of prompt...")

    try:
        resp = gm.generate_content(prompt)
        text = (resp.text or "").strip()
    except Exception as e:
        print(f"✗ Gemini call failed: {e}")
        return 2

    if text == "NO_ACTION" or '"NO_ACTION"' in text:
        print("✓ Coach decided: NO_ACTION (no improvement to suggest)")
        return 3

    suggestion = _parse_gemini_json(text)
    if suggestion is None:
        print("✗ Could not parse Gemini response as JSON:")
        print(text[:500])
        return 2

    # Validate required keys
    required = {"reason", "file_path", "old_snippet", "new_snippet"}
    missing = required - set(suggestion)
    if missing:
        print(f"✗ Gemini response missing required keys: {missing}")
        print(json.dumps(suggestion, indent=2))
        return 2

    # Persist
    SUGGESTION_PATH.write_text(json.dumps(suggestion, indent=2))
    print(f"✓ Coach suggestion written to {SUGGESTION_PATH}")
    print(f"  file: {suggestion['file_path']}")
    print(f"  reason: {suggestion['reason']}")

    # Also export env vars for the create-pull-request action
    github_env = Path(os.environ.get("GITHUB_ENV", "/dev/null"))
    if github_env.exists() and github_env.write_text:
        with github_env.open("a") as f:
            f.write(f"COACH_BODY={suggestion['reason']}\n")
            f.write(f"COACH_PATHS={suggestion['file_path']}\n")
            if suggestion.get("test_change"):
                # crude: extract file_path from the test change
                # not always present; OK to skip
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
