# Colab Notebooks

Two notebooks live here. They serve different purposes.

## `demo_v2.ipynb` — The 17-cell Coach-powered walkthrough (recommended)

The production demo. Open this in Colab, set the runtime to T4 GPU, and click **Run all**.

- **17 cells** organized into 4 sections (§1 setup, §2 walkthrough, §3 comparison, §4 Coach)
- **Self-logging**: every cell writes to a shared `SessionState` singleton (logs, errors, metrics, toggles)
- **Module toggles**: 4 abstractions + 8 modules, each enable/disable via checkboxes
- **Coach integration**: when `GEMINI_API_KEY` is set in Colab secrets, the Coach cell asks Gemini to diagnose any captured errors and review the run
- **Downloads the session log** as JSON at the end for the post-call review

**Open with**:
```
https://colab.research.google.com/github/roniejosephv-star/conveyor-perception/blob/main/notebooks/demo_v2.ipynb
```

**Source-of-truth** is `build_demo_v2.py` — re-run it to regenerate the `.ipynb` from clean Python.

## `demo.ipynb` — The original 9-cell walkthrough (deprecated)

The first demo, kept for reference. Replaced by `demo_v2.ipynb` which adds the Coach, the toggles, the state machinery, and the §3 comparison. New runs should use `demo_v2.ipynb`.

If you have an old saved copy of `demo.ipynb` it still works — the test count + benchmarks are the same.

## The helpers (`colab_session.py`)

Both notebooks depend on `notebooks/colab_session.py` for the runtime machinery. It provides:

- `SessionState` — the singleton state object
- `get_state()` / `reset_state()` — accessors
- `cell()` / `run_cell()` — context managers that auto-capture exceptions + timing
- `env_check()` — runtime detection (GPU, RAM, disk, Python, is_colab)
- `toggle_ui()` — the ipywidgets checkbox form
- `coach_diagnose()` / `coach_review()` — Gemini-powered helpers
- `hint_for()` — static fallback hints
- `download_session_log()` — browser download of the JSON log

23 unit tests cover the pure-Python parts. The Colab-specific bits (`ipywidgets`, `google.colab.userdata`, `google.generativeai`) are imported lazily and not unit-tested.

## Rebuilding

To regenerate `demo_v2.ipynb` from the source:

```bash
python notebooks/build_demo_v2.py
```

This is run automatically by the test suite (the test fixture rebuilds the notebook before asserting on it).
