"""Colab session helpers — logging, error capture, module toggles, Gemini diagnosis.

This file is `!run` from the Colab notebook `notebooks/demo_v2.ipynb`. It
provides the runtime machinery the notebook needs to be self-documenting
and self-debugging:

- **SessionState**: a singleton that lives in `globals()` and persists
  across notebook cells. Holds logs, errors, toggles, metrics, and the
  Gemini diagnoses. Serializable to JSON for download.
- **log / capture**: decorators that wrap a cell's main work and
  automatically capture inputs, outputs, timing, and exceptions.
- **toggle_ui**: an ipywidgets form that lets the user enable / disable
  each of the 4 abstractions and 7 JD modules before running the
  pipeline.
- **coach_diagnose**: asks Gemini to read the captured error log and
  suggest a root cause + fix. Falls back to static hints if no API key
  is configured.
- **env_check**: confirms T4 GPU, enough RAM, and enough disk before the
  pipeline starts.

Design constraint: this file must be importable in vanilla Python (not
Colab) so it can be unit-tested locally. The Colab-specific bits
(ipywidgets, google.colab.userdata, google.generativeai) are imported
lazily inside the functions that need them.

Usage in the notebook:

    # Cell N:
    %run /content/conveyor-perception/notebooks/colab_session.py
    state = get_state()
    with state.cell("cell-1", action="install"):
        # ... do work, return output
        pass
    state.log("cell-1", output=output_dict)
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# --- SessionState --------------------------------------------------------


def _default_toggles() -> dict[str, bool]:
    """All 4 abstractions + 7 modules default to enabled."""
    return {
        # 4 abstractions
        "abstraction:detector": True,
        "abstraction:tracker": True,
        "abstraction:triage": True,
        "abstraction:drift_monitor": True,
        # 7 JD modules
        "module:perception": True,
        "module:triage": True,
        "module:predictive_maintenance": True,
        "module:multitask": True,
        "module:integration": True,
        "module:robustness": True,
        "module:monitoring": True,
        "module:optimization": True,
    }


@dataclass
class SessionState:
    """The single source of truth for a Colab run.

    Persists in `globals()` so every cell sees the same instance. Every
    cell wraps its work in `state.cell(...)` which logs inputs, captures
    exceptions, and stores outputs.

    Attributes:
        env: The Colab runtime environment (GPU type, RAM, disk).
        logs: Every cell's action + input + output + timing.
        errors: Every captured exception with stack trace + hint.
        toggles: The 4 abstractions + 7 modules, each True/False.
        metrics: Free-form numerical results (inference_ms, mAP50, ...).
        gemini_diagnoses: Per-error Gemini responses, if a key is set.
        session_id: Unique ID for this run (timestamp-based).
    """

    env: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    toggles: dict[str, bool] = field(default_factory=_default_toggles)
    metrics: dict[str, Any] = field(default_factory=dict)
    gemini_diagnoses: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: f"run-{int(time.time())}")

    def log(self, cell_id: str, action: str = "", **fields: Any) -> None:
        """Append a log entry. `fields` becomes the entry's payload."""
        self.logs.append(
            {
                "ts": time.time(),
                "cell_id": cell_id,
                "action": action,
                **fields,
            }
        )

    def error(
        self,
        cell_id: str,
        exc: BaseException,
        hint: str = "",
    ) -> None:
        """Capture an exception with stack + optional hint."""
        self.errors.append(
            {
                "ts": time.time(),
                "cell_id": cell_id,
                "type": type(exc).__name__,
                "message": str(exc),
                "stack": traceback.format_exc(),
                "hint": hint,
            }
        )

    def metric(self, key: str, value: Any) -> None:
        """Store a single metric value."""
        self.metrics[key] = value

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "env": self.env,
            "toggles": self.toggles,
            "metrics": self.metrics,
            "logs": self.logs,
            "errors": self.errors,
            "gemini_diagnoses": self.gemini_diagnoses,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def summary_table(self) -> str:
        """Return a printable summary suitable for a notebook cell."""
        lines = [
            f"### Session {self.session_id}",
            f"- Toggles: {sum(self.toggles.values())}/{len(self.toggles)} enabled",
            f"- Logs: {len(self.logs)}",
            f"- Errors: {len(self.errors)}",
            f"- Metrics: {len(self.metrics)} keys",
        ]
        if self.env:
            gpu = self.env.get("gpu", "unknown")
            ram_gb = self.env.get("ram_gb", "?")
            lines.append(f"- Env: {gpu} · {ram_gb} GB RAM")
        if self.metrics:
            lines.append("")
            lines.append("**Metrics:**")
            for k, v in self.metrics.items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.3f}")
                else:
                    lines.append(f"- {k}: {v}")
        return "\n".join(lines)


_STATE_KEY = "_conveyor_perception_state"


def get_state() -> SessionState:
    """Return the singleton SessionState, creating it if needed.

    Stored in `globals()` so every cell in the same notebook session
    sees the same instance. Re-running a cell does NOT reset state —
    the user has to call `reset_state()` explicitly.
    """
    import builtins

    g = builtins.__dict__
    if _STATE_KEY not in g:
        g[_STATE_KEY] = SessionState()
    return g[_STATE_KEY]


def reset_state() -> SessionState:
    """Drop the existing state and start a fresh session."""
    import builtins

    g = builtins.__dict__
    g[_STATE_KEY] = SessionState()
    return g[_STATE_KEY]


@contextmanager
def cell(cell_id: str, action: str = ""):
    """Context manager that wraps a cell's main work.

    Logs the start, captures any exception, and logs the end with timing.
    The wrapped block's return value is stored as `result` in the log
    entry. Exceptions are recorded in `state.errors` with the cell id
    and re-raised (so the cell still shows the traceback).

    Usage:
        with state.cell("cell-1", action="install"):
            !pip install -q ...
    """
    state = get_state()
    t0 = time.perf_counter()
    state.log(cell_id, action=action, status="start")
    try:
        yield
    except BaseException as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        state.error(cell_id, exc)
        state.log(
            cell_id,
            action=action,
            status="error",
            elapsed_ms=elapsed_ms,
            error_type=type(exc).__name__,
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        state.log(
            cell_id,
            action=action,
            status="ok",
            elapsed_ms=elapsed_ms,
        )


# --- env_check -----------------------------------------------------------


def env_check() -> dict[str, Any]:
    """Detect the runtime environment. Best-effort outside Colab.

    Returns a dict with:
        gpu: "T4" / "A100" / "CPU" / "unknown"
        ram_gb: integer, approximate
        disk_gb_free: integer, approximate
        python: version string
        is_colab: True if running in Google Colab
    """
    info: dict[str, Any] = {
        "gpu": "unknown",
        "ram_gb": 0,
        "disk_gb_free": 0,
        "python": "",
        "is_colab": False,
    }

    # Python version
    import sys

    info["python"] = sys.version.split()[0]

    # Colab detection
    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401

        info["is_colab"] = True
    except ImportError:
        info["is_colab"] = False

    # GPU detection (nvidia-smi or torch)
    try:
        import shutil
        import subprocess

        smi = shutil.which("nvidia-smi")
        if smi is not None:
            out = subprocess.run(
                [smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            name = (out.stdout or "").strip().split("\n")[0]
            if "T4" in name:
                info["gpu"] = "T4"
            elif "A100" in name:
                info["gpu"] = "A100"
            elif "A10" in name:
                info["gpu"] = "A10G"
            elif "L4" in name:
                info["gpu"] = "L4"
            elif name:
                info["gpu"] = name
    except Exception:
        pass

    if info["gpu"] == "unknown":
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0) or "CUDA"
                info["gpu"] = name
            else:
                info["gpu"] = "CPU"
        except Exception:
            info["gpu"] = "CPU"

    # RAM (Linux only — Colab runs Linux)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / 1024 / 1024)
                    break
    except Exception:
        pass

    # Disk free
    try:
        import shutil

        total, used, free = shutil.disk_usage("/")
        info["disk_gb_free"] = round(free / 1024 / 1024 / 1024)
    except Exception:
        pass

    return info


# --- toggle_ui -----------------------------------------------------------


def toggle_ui() -> Any:
    """Build the ipywidgets toggle form. Returns the VBox widget.

    Each checkbox maps to a key in `state.toggles`. The user checks /
    unchecks modules before running the pipeline. Toggling triggers
    a write to the live `state.toggles` dict so the pipeline cell
    can read the current values.
    """
    state = get_state()

    try:
        import ipywidgets as widgets
    except ImportError as e:
        raise ImportError(
            "ipywidgets is required for toggle_ui(). Install with: pip install ipywidgets"
        ) from e

    boxes: list[Any] = []
    boxes.append(widgets.HTML("<b>4 framework abstractions</b>"))
    for key in [
        "abstraction:detector",
        "abstraction:tracker",
        "abstraction:triage",
        "abstraction:drift_monitor",
    ]:
        cb = widgets.Checkbox(
            value=state.toggles[key],
            description=key.split(":", 1)[1],
            indent=False,
        )

        def _on_change(change: Any, k: str = key) -> None:
            state.toggles[k] = change["new"]
            state.log("toggle-ui", action="toggle", key=k, value=change["new"])

        cb.observe(_on_change, names="value")
        boxes.append(cb)

    boxes.append(widgets.HTML("<br><b>7 JD modules</b>"))
    for key in [
        "module:perception",
        "module:triage",
        "module:predictive_maintenance",
        "module:multitask",
        "module:integration",
        "module:robustness",
        "module:monitoring",
        "module:optimization",
    ]:
        cb = widgets.Checkbox(
            value=state.toggles[key],
            description=key.split(":", 1)[1],
            indent=False,
        )

        def _on_change(change: Any, k: str = key) -> None:
            state.toggles[k] = change["new"]
            state.log("toggle-ui", action="toggle", key=k, value=change["new"])

        cb.observe(_on_change, names="value")
        boxes.append(cb)

    return widgets.VBox(boxes)


# --- coach_diagnose (Gemini) ---------------------------------------------


def _get_gemini_key() -> str | None:
    """Look up the Gemini API key from Colab's userdata or env.

    Colab path: `from google.colab import userdata; userdata.get('GEMINI_API_KEY')`.
    Local path: `os.environ.get('GEMINI_API_KEY')`. Returns None if not set.
    """
    # Colab userdata
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        key = userdata.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    # Env var
    return os.environ.get("GEMINI_API_KEY")


def coach_diagnose(
    error: dict[str, Any],
    *,
    extra_context: str = "",
    model: str = "gemini-2.0-flash",
) -> str:
    """Ask Gemini to diagnose a captured error.

    Args:
        error: One entry from `state.errors` (has `type`, `message`, `stack`, `hint`).
        extra_context: Optional extra context to include in the prompt
            (e.g. "the user is running on T4 with 12 GB RAM").
        model: The Gemini model to use. `gemini-2.0-flash` is the
            current free-tier default as of 2026.

    Returns:
        The Gemini response as a string, or a static fallback message
        if the API key isn't set or the call fails.
    """
    api_key = _get_gemini_key()
    if not api_key:
        return (
            "No GEMINI_API_KEY configured. Set it in Colab via:\n"
            "  from google.colab import userdata\n"
            "  userdata.set('GEMINI_API_KEY', 'your-key')\n\n"
            f"Static hint for {error.get('type', 'Unknown')}: {error.get('hint', '(no hint)')}"
        )

    try:
        import google.generativeai as genai  # type: ignore[import-not-found]
    except ImportError:
        return (
            "google-generativeai SDK not installed. Run:\n"
            "  !pip install -q google-generativeai\n\n"
            f"Static hint for {error.get('type', 'Unknown')}: {error.get('hint', '(no hint)')}"
        )

    genai.configure(api_key=api_key)
    prompt = _build_diagnosis_prompt(error, extra_context)
    try:
        gm = genai.GenerativeModel(model)
        resp = gm.generate_content(prompt)
        return resp.text or "(empty response from Gemini)"
    except Exception as e:
        return f"Gemini call failed: {e}\n\nStatic hint: {error.get('hint', '(no hint)')}"


def _build_diagnosis_prompt(error: dict[str, Any], extra_context: str) -> str:
    """Build the diagnosis prompt for Gemini."""
    return f"""You are the Conveyor Perception Coach, an expert at debugging
industrial computer-vision pipelines built on YOLO26 + OpenCV DNN +
supervision ByteTrack + FastMCP. You are reviewing an error captured
from a live Colab run of a recycling-line perception stack.

Given the error below, produce a concise diagnosis in this format:

  Root cause (1-2 sentences)
  Why it happens (1-2 sentences)
  How to fix (1-3 numbered steps)
  How to prevent (1 sentence)

Be specific. If the error mentions a known library (ultralytics, opencv,
supervision, fastmcp, roboflow, pydantic, fastapi), give the
exact version-aware fix. If the error is a CUDA OOM, give the
batch size + imgsz combination that will fit. If the error is a
Roboflow 404, give the dataset URL format that's known to work.

ERROR:
  type: {error.get('type')}
  message: {error.get('message')}
  hint: {error.get('hint', '(none)')}

STACK TRACE (last 30 lines):
{chr(10).join((error.get('stack') or '').splitlines()[-30:])}

{f'EXTRA CONTEXT: {extra_context}' if extra_context else ''}
"""


def coach_review(
    state: SessionState,
    *,
    include_metrics: bool = True,
    model: str = "gemini-2.0-flash",
) -> str:
    """Ask Gemini to review a successful end-to-end run.

    Unlike `coach_diagnose` (which is for errors), this reviews a
    completed run and surfaces anything worth flagging — performance
    anomalies, missing modules, suspicious toggles.
    """
    api_key = _get_gemini_key()
    if not api_key:
        return (
            "No GEMINI_API_KEY configured. Skipping Coach review.\n"
            "Set it via: from google.colab import userdata; "
            "userdata.set('GEMINI_API_KEY', 'your-key')"
        )

    try:
        import google.generativeai as genai  # type: ignore[import-not-found]
    except ImportError:
        return "google-generativeai SDK not installed. Skipping Coach review."

    genai.configure(api_key=api_key)
    summary = state.summary_table()
    toggles_summary = ", ".join(
        f"{k.split(':',1)[1]}={'on' if v else 'off'}"
        for k, v in state.toggles.items()
    )
    metrics_block = (
        "\nMetrics:\n" + "\n".join(f"  {k}: {v}" for k, v in state.metrics.items())
        if include_metrics and state.metrics
        else ""
    )
    prompt = f"""You are the Conveyor Perception Coach. Review the following
end-to-end run of an industrial perception stack and produce:

  1. One-line summary verdict
  2. Any anomalies (slow inference, missing metrics, odd toggle combos)
  3. Two concrete next steps the user could take to improve the run

Be brief. Skip the "looks good" filler.

{summary}

Toggles: {toggles_summary}
{metrics_block}
"""
    try:
        gm = genai.GenerativeModel(model)
        resp = gm.generate_content(prompt)
        return resp.text or "(empty response)"
    except Exception as e:
        return f"Gemini call failed: {e}"


# --- download_session_log -------------------------------------------------


def download_session_log() -> Any:
    """Trigger a browser download of the session log as JSON.

    Returns the download object (the user clicks it). Colab-specific.
    """
    state = get_state()
    json_text = state.to_json()
    try:
        from google.colab import files  # type: ignore[import-not-found]

        path = f"/tmp/{state.session_id}.json"
        with open(path, "w") as f:
            f.write(json_text)
        return files.download(path)
    except ImportError:
        # Not in Colab — write to local cwd
        path = f"{state.session_id}.json"
        with open(path, "w") as f:
            f.write(json_text)
        return path


# --- remediation_hints ---------------------------------------------------

# Static hints keyed by exception type. Used by the notebook when Gemini
# isn't available. Keep these short and version-aware.
REMEDIATION_HINTS: dict[str, str] = {
    "FileNotFoundError": (
        "Check the path. Colab cwd is /content, not the repo root. "
        "Use %cd /content/conveyor-perception before file ops."
    ),
    "ModuleNotFoundError": (
        "Run the pip install cell again. Sometimes Colab's first "
        "install takes >60s and times out silently. "
        "Also check: `pip show <module>` to see if it's actually installed."
    ),
    "ImportError": (
        "Same as ModuleNotFoundError, but for relative imports. "
        "If the error mentions 'attempted relative import with no known parent package', "
        "make sure you're running the file with -m, not as a script."
    ),
    "CUDA out of memory": (
        "Reduce --batch from 32 to 16, or --imgsz from 640 to 416. "
        "T4 has 16 GB; batch 32 at imgsz 640 is right on the edge."
    ),
    "RoboflowError": (
        "Check the dataset slug. The format is workspace/project. "
        "Try `rf.workspace('zkf624').project('-recycling').version(3).download('yolov11')`."
    ),
    "ValueError": (
        "Usually a shape mismatch. Check that the model's class count matches "
        "the data.yaml's nc. For the recycling dataset: nc=4 (Glass, metal, plastic, vinyl)."
    ),
    "RuntimeError": (
        "Generic torch error. Check that all tensors are on the same device "
        "(cuda:0 vs cpu). Try model.to('cuda:0') and inputs.to('cuda:0')."
    ),
    "TimeoutError": (
        "Colab runtime disconnected. Click the 'Reconnect' button. "
        "State is preserved in `state` (the SessionState singleton)."
    ),
}


def hint_for(exc: BaseException) -> str:
    """Pick the best static hint for a given exception.

    Matches against (in order):
    1. Special substrings in the message ("Roboflow", "CUDA out of memory", "timeout")
    2. The full MRO of the exception class (handles subclasses)
    3. A generic fallback
    """
    msg_lower = str(exc).lower()

    # Special substrings that should match even if the type name doesn't
    special_substrings = {
        "roboflow": "RoboflowError",
        "cuda out of memory": "CUDA out of memory",
        "timeout": "TimeoutError",
    }
    for needle, hint_key in special_substrings.items():
        if needle in msg_lower and hint_key in REMEDIATION_HINTS:
            return REMEDIATION_HINTS[hint_key]

    # Class name match (handles subclasses via MRO)
    for cls in type(exc).__mro__:
        if cls.__name__ in REMEDIATION_HINTS:
            return REMEDIATION_HINTS[cls.__name__]

    # Fallback
    return "No static hint. Try the Coach diagnose cell."


# --- public cell wrapper --------------------------------------------------


def run_cell(cell_id: str, action: str, fn: Callable[[], Any]) -> Any:
    """Wrap `fn()` with logging + error capture. Returns fn()'s result.

    Convenience wrapper for cells that prefer a function over a
    context manager. The `cell()` context manager is preferred for
    most cases.

    Usage:
        result = run_cell("cell-1", "install", lambda: subprocess.run([...]))
    """
    state = get_state()
    with cell(cell_id, action):
        result = fn()
        if result is not None:
            state.log(cell_id, action=action, status="result", result=result)
        return result
