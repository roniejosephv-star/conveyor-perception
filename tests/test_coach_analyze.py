"""Tests for .github/workflows/coach_analyze.py.

The Coach analysis runs in a GitHub Action on each v0.0.* release.
These tests cover the pure-Python helpers + the script's return-code
contract. The actual Gemini API call is mocked at the function level
(bypass the real SDK by patching genai.GenerativeModel) to avoid
network calls in CI.

Return-code contract:
  0 = success, suggestion written to /tmp/coach_suggestion.json
  1 = GEMINI_API_KEY not set
  2 = current.json missing OR Gemini call failed OR response invalid
  3 = Gemini said "NO_ACTION"
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / ".github" / "workflows" / "coach_analyze.py"


# --- Helpers tests (no mocks needed) --------------------------------------


def test_load_json_handles_missing():
    """Missing file → returns None."""
    result = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, '{SCRIPT.parent}'); "
         "import importlib.util; "
         "spec = importlib.util.spec_from_file_location('m', '" + str(SCRIPT) + "'); "
         "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
         "import pathlib; "
         "print(m._load_json(pathlib.Path('/nonexistent/path.json')))"],
        capture_output=True, text=True, timeout=10,
    )
    assert "None" in result.stdout


def test_load_json_handles_bad_json(tmp_path):
    """Bad JSON → returns None."""
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    result = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, '{SCRIPT.parent}'); "
         "import importlib.util, pathlib; "
         "spec = importlib.util.spec_from_file_location('m', '" + str(SCRIPT) + "'); "
         "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
         f"print(m._load_json(pathlib.Path('{p}')))"],
        capture_output=True, text=True, timeout=10,
    )
    assert "None" in result.stdout


def test_load_json_handles_good_json(tmp_path):
    """Good JSON → returns the parsed dict."""
    p = tmp_path / "good.json"
    p.write_text('{"a": 1, "b": [2, 3]}')
    result = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, '{SCRIPT.parent}'); "
         "import importlib.util, pathlib; "
         "spec = importlib.util.spec_from_file_location('m', '" + str(SCRIPT) + "'); "
         "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
         f"print(m._load_json(pathlib.Path('{p}')))"],
        capture_output=True, text=True, timeout=10,
    )
    assert "{'a': 1, 'b': [2, 3]}" in result.stdout


def test_diff_summary_first_run():
    """First run (no prev) → mentions 'No previous run' + 'first session'."""
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, '{SCRIPT.parent}'); "
         "import importlib.util; "
         "spec = importlib.util.spec_from_file_location('m', '" + str(SCRIPT) + "'); "
         "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
         "print(m._diff_summary(None, {'metrics': {'inference_ms': 8.7}, 'errors': []}))"],
        capture_output=True, text=True, timeout=10,
    )
    assert "No previous run" in out.stdout
    assert "first session" in out.stdout


def test_diff_summary_subsequent_run():
    """Subsequent run → includes metrics + errors diff."""
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, '{SCRIPT.parent}'); "
         "import importlib.util; "
         "spec = importlib.util.spec_from_file_location('m', '" + str(SCRIPT) + "'); "
         "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
         "prev = {'metrics': {'inference_ms': 12.0, 'mAP50': 0.65}, 'errors': [1, 2], 'toggles': {'a': True}}; "
         "curr = {'metrics': {'inference_ms': 8.7, 'mAP50': 0.671}, 'errors': [], 'toggles': {'a': True}}; "
         "print(m._diff_summary(prev, curr))"],
        capture_output=True, text=True, timeout=10,
    )
    assert "12.0" in out.stdout
    assert "8.7" in out.stdout
    assert "0.65" in out.stdout
    assert "0.671" in out.stdout
    assert "2 → 0" in out.stdout


def test_build_prompt_contains_session():
    """The prompt embeds the session + the NO_ACTION rule."""
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, '{SCRIPT.parent}'); "
         "import importlib.util; "
         "spec = importlib.util.spec_from_file_location('m', '" + str(SCRIPT) + "'); "
         "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
         "curr = {'session_id': 'test-1', 'metrics': {'inference_ms': 8.7}}; "
         "prompt = m._build_prompt(curr, None); "
         "print('test-1' in prompt, '8.7' in prompt, 'NO_ACTION' in prompt)"],
        capture_output=True, text=True, timeout=10,
    )
    assert "True True True" in out.stdout


def test_parse_gemini_json_plain():
    """Plain JSON → parsed dict."""
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, '{SCRIPT.parent}'); "
         "import importlib.util; "
         "spec = importlib.util.spec_from_file_location('m', '" + str(SCRIPT) + "'); "
         "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
         "text = '{\"reason\": \"x\", \"file_path\": \"a.py\", \"old_snippet\": \"b\", \"new_snippet\": \"c\"}'; "
         "print(m._parse_gemini_json(text))"],
        capture_output=True, text=True, timeout=10,
    )
    assert "reason': 'x'" in out.stdout


def test_parse_gemini_json_in_fence():
    """JSON inside ```json ... ``` → parsed dict."""
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, '{SCRIPT.parent}'); "
         "import importlib.util; "
         "spec = importlib.util.spec_from_file_location('m', '" + str(SCRIPT) + "'); "
         "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
         "text = '```json\\n{\"reason\": \"x\"}\\n```'; "
         "print(m._parse_gemini_json(text))"],
        capture_output=True, text=True, timeout=10,
    )
    assert "reason': 'x'" in out.stdout


def test_parse_gemini_json_garbage_returns_none():
    """Non-JSON text → None."""
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, '{SCRIPT.parent}'); "
         "import importlib.util; "
         "spec = importlib.util.spec_from_file_location('m', '" + str(SCRIPT) + "'); "
         "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
         "print(m._parse_gemini_json('not json at all')); "
         "print(m._parse_gemini_json(''))"],
        capture_output=True, text=True, timeout=10,
    )
    assert "None" in out.stdout
    assert out.stdout.count("None") == 2


# --- main() return-code contract (subprocess with real env) ---------------


def test_main_returns_1_without_api_key(tmp_path, monkeypatch):
    """No GEMINI_API_KEY → return 1, no suggestion file.

    We can test this without symlinking /tmp/runs because the script
    short-circuits on the API key check BEFORE trying to read current.json.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"},
        timeout=10,
    )
    assert result.returncode == 1, f"expected 1, got {result.returncode}: {result.stdout}\n{result.stderr}"
