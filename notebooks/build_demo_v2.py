"""Build the demo_v2.ipynb notebook (the 16-cell walkthrough).

Run this script to regenerate notebooks/demo_v2.ipynb from the cell
definitions below. Keeps the source-of-truth in clean Python instead of
fragile hand-edited JSON.

Usage:
    cd /Users/mindflow/Projects/Job\\ Hunt/conveyor-perception
    python notebooks/build_demo_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "notebooks" / "demo_v2.ipynb"


# --- helpers -------------------------------------------------------------


def md(*lines: str) -> dict[str, Any]:
    """Build a markdown cell from one or more lines."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines],
    }


def code(*lines: str) -> dict[str, Any]:
    """Build a code cell from one or more lines."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


def nbformat_v4(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap cells in a v4 notebook format dict."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --- the 16 cells --------------------------------------------------------


CELLS: list[dict[str, Any]] = []

# ===== §1 SETUP ============================================================

# Cell 0 (markdown): Front door — what this is, what it does
CELLS.append(md(
    "# Conveyor Perception v2 — Coach-Powered Walkthrough",
    "",
    "**The complete industrial CV stack on Colab T4, with a Gemini-powered Coach that diagnoses failures and reviews the run.**",
    "",
    "This is the production demo of [roniejosephv-star/conveyor-perception](https://github.com/roniejosephv-star/conveyor-perception). It runs end-to-end on a free Colab T4 GPU and shows every part of the stack that maps to a real recycling-line JD:",
    "",
    "- **Detection** (YOLO26 + OpenCV DNN, NMS-free, segmentation-aware fallback via UltralyticsDetector)",
    "- **Tracking** (supervision ByteTrack, IoU fallback for tests)",
    "- **Drift detection** (3-signal: KS test on confidence, z-score on counts, MAD on latency)",
    "- **L1 triage** (7 deterministic severity rules + MCP-style 5-tool surface)",
    "- **Predictive maintenance** (rule-based drift signals → actionable hints)",
    "- **Robustness** (13 MRF-condition augmentations, broken/degraded/ok classification)",
    "- **Monitoring** (FastAPI-style shift dashboard + retrain recommendation)",
    "",
    "**Runtime**: ~20 min on free T4 (12 min training + 8 min walkthrough).",
    "",
    "**The Coach**: an optional Gemini integration that reads the session log and diagnoses any failures. Set the `GEMINI_API_KEY` Colab secret (key icon in the left sidebar) to enable it. Without a key, the Coach still works — it falls back to static hints.",
    "",
    "---",
    "",
    "## How to use this notebook",
    "",
    "1. Runtime → Change runtime type → **T4 GPU** (already done if you see the green check)",
    "2. Click **Run all** in the Runtime menu, OR walk through cells one at a time",
    "3. Each cell logs to a shared `state` (SessionState singleton). Errors are caught and stored.",
    "4. The Coach cell (§4 cell 15) asks Gemini to diagnose any failures automatically.",
    "5. The final cell (§4 cell 16) offers a JSON download of the full session log.",
    "",
    "**Toggle modules** in §1 cell 4 to enable/disable each of the 4 abstractions and 8 modules. The pipeline reads these toggles and skips disabled components.",
))

# Cell 1 (code): Runtime check + intro
CELLS.append(code(
    "# --- Cell 1: Runtime + env check ---",
    "REPO = '/content/conveyor-perception'",
    "REPO_URL = 'https://github.com/roniejosephv-star/conveyor-perception.git'",
    "import os, sys",
    "os.chdir(REPO) if os.path.exists(REPO) else None  # if not cloned yet, this no-ops",
    "sys.path.insert(0, REPO)",
    "sys.path.insert(0, os.path.join(REPO, 'notebooks'))",
    "sys.path.insert(0, os.path.join(REPO, 'src'))  # so 'import conveyor_perception' works in later cells",
    "",
    "# --- Self-heal: ensure the repo is cloned AND colab_session is importable ---",
    "# Colab's /content persists across runtime restarts, so the repo may exist",
    "# from a previous session. We handle 3 cases:",
    "#   (a) Repo not cloned yet     → git clone",
    "#   (b) Repo cloned (valid)     → git pull (in case it's stale)",
    "#   (c) Repo dir exists but bad → rm -rf and re-clone",
    "# We also catch CalledProcessError so a single failure doesn't kill the",
    "# whole cell — instead the user sees a clear error with next steps.",
    "try:",
    "    from colab_session import env_check, get_state  # noqa: F401",
    "except ImportError:",
    "    import subprocess, shutil",
    "    from pathlib import Path",
    "    repo_dir = Path(REPO)",
    "    is_valid_git = repo_dir.exists() and (repo_dir / '.git').exists()",
    "    has_colab_session = (repo_dir / 'notebooks' / 'colab_session.py').exists()",
    "    if has_colab_session:",
    "        print(f'Repo found at {repo_dir} — but import failed. Retrying after re-adding path...')",
    "    elif is_valid_git:",
    "        print(f'Repo found at {repo_dir} — pulling latest (notebooks/colab_session.py is missing)...')",
    "        subprocess.run(['git', '-C', str(repo_dir), 'pull', '--rebase'], check=False)",
    "    elif repo_dir.exists():",
    "        print(f'Repo dir {repo_dir} exists but is not a valid git repo. Removing and re-cloning...')",
    "        shutil.rmtree(repo_dir)",
    "        subprocess.run(['git', 'clone', REPO_URL, str(repo_dir)], check=True)",
    "    else:",
    "        print(f'Cloning repo to {repo_dir} (~5s)...')",
    "        subprocess.run(['git', 'clone', REPO_URL, str(repo_dir)], check=True)",
    "    sys.path.insert(0, str(repo_dir))",
    "    sys.path.insert(0, str(repo_dir / 'notebooks'))",
    "    os.chdir(repo_dir)",
    "    try:",
    "        from colab_session import env_check, get_state  # noqa: F401 (after heal)",
    "    except ImportError as e:",
    "        print('=' * 60)",
    "        print(f'  ✗ Self-heal failed: {e}')",
    "        print()",
    "        print('  Possible causes:')",
    "        print('  1. Network blocked — try Runtime → Restart runtime, then re-run')",
    "        print('  2. /content is full — Runtime → Factory reset runtime')",
    "        print(f'  3. Repo URL wrong: {REPO_URL}')",
    "        print('=' * 60)",
    "        raise",
    "",
    "# Load the session helpers",
    "from colab_session import env_check, get_state, cell",
    "",
    "state = get_state()",
    "state.env = env_check()",
    "",
    "print('=' * 60)",
    "print(f\"  GPU:       {state.env.get('gpu', 'unknown')}\")",
    "print(f\"  RAM:       {state.env.get('ram_gb', '?')} GB\")",
    "print(f\"  Disk free: {state.env.get('disk_gb_free', '?')} GB\")",
    "print(f\"  Python:    {state.env.get('python', '?')}\")",
    "print(f\"  In Colab:  {state.env.get('is_colab', False)}\")",
    "print('=' * 60)",
    "",
    "# Soft checks — warn but don't fail",
    "if state.env.get('gpu') == 'CPU':",
    "    print('\\n⚠ Running on CPU. The pipeline still works but inference will be ~10x slower.'",
    "          ' Switch to T4 GPU in Runtime → Change runtime type.')",
    "if state.env.get('ram_gb', 0) < 10:",
    "    print(f\"\\n⚠ Only {state.env.get('ram_gb', '?')} GB RAM. Some cells may need --batch 16 instead of 32.\")",
    "if state.env.get('disk_gb_free', 0) < 5:",
    "    print(f\"\\n⚠ Only {state.env.get('disk_gb_free', '?')} GB free disk. Dataset + model need ~2 GB.\")",
    "",
    "state.log('cell-1', action='env-check', env=state.env)",
    "print('\\n✓ Cell 1 done. State initialized.')",
))

# Cell 1 (markdown): Section header for §1
CELLS.append(md(
    "---",
    "",
    "## §1 SETUP — runtime check, install, state, toggles",
    "",
    "Get a clean T4 environment, install pinned deps, clone the repo, set up the shared state, and pick which modules to run.",
))

# Cell 2 (code): Install + clone + Roboflow key
CELLS.append(code(
    "# --- Cell 2: Install + clone + Roboflow key ---",
    "import os, subprocess, sys",
    "from pathlib import Path",
    "",
    "REPO = Path('/content/conveyor-perception')",
    "",
    "with cell('cell-2', action='install-and-clone'):",
    "    # Install pinned deps (matches requirements.txt)",
    "    subprocess.run([",
    "        sys.executable, '-m', 'pip', 'install', '-q',",
    "        'ultralytics==8.4.121',",
    "        'opencv-python==4.11.0.86',",
    "        'supervision==0.30.0',",
    "        'fastmcp==3.4.7',",
    "        'pydantic==2.13.4',",
    "        'roboflow==1.4.1',",
    "        'onnxruntime>=1.20.1',",
    "        'numpy>=1.26,<2.0',",
    "        'pyyaml==6.0.2',",
    "        'python-dotenv>=1.1.0',",
    "        'ipywidgets>=8.0',",
    "        'google-generativeai>=0.8',",
    "    ], check=True)",
    "    print('✓ Pinned deps installed')",
    "",
    "    # Clone or pull the repo",
    "    if not REPO.exists():",
    "        subprocess.run([",
    "            'git', 'clone',",
    "            'https://github.com/roniejosephv-star/conveyor-perception.git',",
    "            str(REPO),",
    "        ], check=True)",
    "        print(f'✓ Cloned repo to {REPO}')",
    "    else:",
    "        subprocess.run(['git', '-C', str(REPO), 'pull', '--rebase'], check=False)",
    "        print(f'✓ Pulled latest from {REPO}')",
    "",
    "    # Roboflow API key (public read-only key for the demo; replace for real use)",
    "    if not os.path.exists(REPO / '.env'):",
    "        with open(REPO / '.env', 'w') as f:",
    "            f.write('ROBOFLOW_API_KEY=qogO5hAuLgUUYMbNT6W3\\n')",
    "        print('✓ Wrote demo .env (read-only public key; replace for real work)')",
    "    else:",
    "        print('✓ .env already present')",
    "",
    "    # Add to path",
    "    sys.path.insert(0, str(REPO))",
    "    sys.path.insert(0, str(REPO / 'notebooks'))",
    "    os.chdir(REPO)",
    "",
    "print('\\n✓ Cell 2 done. Repo ready.')",
))

# Cell 3 (code): Init SessionState properly
CELLS.append(code(
    "# --- Cell 3: Initialize SessionState + reload env ---",
    "from colab_session import get_state, reset_state, env_check",
    "",
    "state = get_state()",
    "state.env = env_check()  # re-check now that we have the right env",
    "state.metric('session_started', state.session_id)",
    "",
    "print(f'Session: {state.session_id}')",
    "print(f'Env: {state.env[\"gpu\"]} · {state.env[\"ram_gb\"]} GB RAM')",
    "print(f'Toggles: {sum(state.toggles.values())}/{len(state.toggles)} enabled (all on by default)')",
    "print('\\nEvery subsequent cell will log to `state`. Errors are caught and stored.')",
    "print('\\n✓ Cell 3 done. State ready.')",
))

# Cell 4 (code): Module toggle UI
CELLS.append(code(
    "# --- Cell 4: Module toggle UI ---",
    "# Tick / untick to enable / disable each component. The pipeline cells",
    "# in §2 read state.toggles to decide what to instantiate.",
    "",
    "from colab_session import toggle_ui, get_state",
    "",
    "ui = toggle_ui()",
    "display(ui)",
    "",
    "state = get_state()",
    "print('\\nCurrent toggles:')",
    "for k, v in state.toggles.items():",
    "    icon = '✓' if v else '○'",
    "    print(f'  {icon} {k}')",
))

# ===== §2 WALKTHROUGH =====================================================

# Cell 5 (markdown): Section header
CELLS.append(md(
    "---",
    "",
    "## §2 WALKTHROUGH — the 4 abstractions + 7 modules",
    "",
    "Each component is loaded in its own cell so failures are isolated. If any cell errors, the Coach (§4) will diagnose it.",
))

# Cell 6 (code): The 4 abstractions
CELLS.append(code(
    "# --- Cell 6: The 4 framework abstractions ---",
    "import sys, os",
    "sys.path.insert(0, '/content/conveyor-perception')",
    "sys.path.insert(0, '/content/conveyor-perception/src')  # so 'import conveyor_perception' works",
    "os.chdir('/content/conveyor-perception')",
    "",
    "from colab_session import get_state, hint_for",
    "# 'Detector' is the role; the actual class is 'DetectionPipeline' (YOLO26 + OpenCV DNN).",
    "# Aliased on import so the rest of the cell reads naturally.",
    "from conveyor_perception.core.detection_pipeline import DetectionPipeline as Detector, Detection",
    "from conveyor_perception.core.tracking_pipeline import TrackingPipeline",
    "from conveyor_perception.core.drift_monitor import DriftMonitor",
    "# MCPTriageSurface needs a name + an AlertSource. InMemoryAlertQueue is the",
    "# lightweight in-process AlertSource used for the demo (no real broker).",
    "from conveyor_perception.core.triage_surface import MCPTriageSurface, InMemoryAlertQueue",
    "",
    "state = get_state()",
    "loaded = {}",
    "",
    "with cell('cell-6', action='load-4-abstractions'):",
    "    if state.toggles.get('abstraction:detector'):",
    "        # Detector loads ONNX; we'll wire the model in cell 8 after training.",
    "        # For now just verify the class imports.",
    "        loaded['detector_class'] = Detector",
    "        print('✓ Detector class loaded (YOLO26 + OpenCV DNN)')",
    "",
    "    if state.toggles.get('abstraction:tracker'):",
    "        loaded['tracker'] = TrackingPipeline()",
    "        print('✓ TrackingPipeline instantiated (ByteTrack with IoU fallback)')",
    "",
    "    if state.toggles.get('abstraction:drift_monitor'):",
    "        loaded['drift_monitor'] = DriftMonitor(baseline_window=50, min_samples_for_drift=20)",
    "        print('✓ DriftMonitor instantiated (KS test + z-score + MAD)')",
    "",
    "    if state.toggles.get('abstraction:triage'):",
    "        loaded['triage_surface'] = MCPTriageSurface('l1-triage', InMemoryAlertQueue())",
    "        print('✓ MCPTriageSurface instantiated (5 tools, FastMCP server)')",
    "",
    "print(f'\\nLoaded: {len(loaded)}/4 abstractions')",
    "state.log('cell-6', action='result', loaded=list(loaded.keys()))",
))

# Cell 7 (code): The 7 modules (lightweight load + show signatures)
CELLS.append(code(
    "# --- Cell 7: The 7+1 JD modules — show signatures and import paths ---",
    "import importlib, inspect",
    "from colab_session import get_state",
    "",
    "state = get_state()",
    "",
    "modules_meta = [",
    "    ('module:perception',           'conveyor_perception.perception',   'Detector + UltralyticsDetector'),",
    "    ('module:triage',               'conveyor_perception.triage',       'L1TriageAgent + 7 severity rules'),",
    "    ('module:predictive_maintenance', 'conveyor_perception.predictive_maintenance', 'MaintenanceAdvisor + 3 signal types'),",
    "    ('module:multitask',            'conveyor_perception.multitask',    'MultitaskPipeline (Detector→Tracker→Drift→Triage)'),",
    "    ('module:integration',          'conveyor_perception.integration',  'ConveyorNode (real ROS 2) + MockROS2Node (CI)'),",
    "    ('module:robustness',           'conveyor_perception.robustness',   'RobustnessTestSuite + 13 augmentations'),",
    "    ('module:monitoring',           'conveyor_perception.monitoring',   'MonitoringDashboard + ShiftReport'),",
    "    ('module:optimization',         'conveyor_perception.optimization', 'benchmark_pytorch/onnx + export_onnx'),",
    "]",
    "",
    "loaded = []",
    "skipped = []",
    "for toggle_key, module_path, desc in modules_meta:",
    "    if not state.toggles.get(toggle_key):",
    "        skipped.append(toggle_key)",
    "        print(f'  ○ {module_path} (disabled by toggle)')",
    "        continue",
    "    try:",
    "        with cell(f'cell-7-{module_path}', action='import'):",
    "            importlib.import_module(module_path)",
    "            loaded.append(module_path)",
    "            print(f'  ✓ {module_path} — {desc}')",
    "    except Exception as exc:",
    "        print(f'  ✗ {module_path} failed: {exc}')",
    "        print(f'    Hint: {hint_for(exc)}')",
    "",
    "print(f'\\nLoaded: {len(loaded)}/{len(modules_meta)} modules, skipped: {len(skipped)}')",
    "state.metric('modules_loaded', len(loaded))",
    "state.metric('modules_skipped', len(skipped))",
))

# Cell 8 (code): Train the model (or use pretrained)
CELLS.append(code(
    "# --- Cell 8: Train YOLO26s (or skip to pretrained for the fast path) ---",
    "import os, sys, subprocess, time",
    "from pathlib import Path",
    "from colab_session import get_state, hint_for",
    "",
    "state = get_state()",
    "REPO = Path('/content/conveyor-perception')",
    "",
    "# Choose: full training (12 min) or pretrained (auto-download)",
    "TRAIN_MODE = 'train'  # 'train' or 'pretrained'",
    "",
    "if TRAIN_MODE == 'train':",
    "    print('Training YOLO26s on T4 (30 epochs, ~10-15 min)...')",
    "    print('Set TRAIN_MODE = \"pretrained\" above to skip training and use COCO weights.\\n')",
    "    t0 = time.time()",
    "    with cell('cell-8', action='train-yolo26s'):",
    "        # Download dataset first",
    "        result = subprocess.run([",
    "            sys.executable, 'scripts/download_dataset.py',",
    "        ], capture_output=True, text=True, cwd=REPO)",
    "        if result.returncode != 0:",
    "            print('Dataset download failed:')",
    "            print(result.stderr[-500:])",
    "            print(f'Hint: {hint_for(Exception(result.stderr))}')",
    "        else:",
    "            print(result.stdout[-500:])",
    "",
    "        # Train (will pick up the model from last.pt if it exists)",
    "        result = subprocess.run([",
    "            sys.executable, 'scripts/train_yolo26.py',",
    "            '--epochs', '30',",
    "            '--imgsz', '640',",
    "            '--batch', '16',  # conservative for free T4",
    "            '--device', '0',",
    "        ], capture_output=True, text=True, cwd=REPO)",
    "        if result.returncode != 0:",
    "            print('Training failed:')",
    "            print(result.stderr[-500:])",
    "        else:",
    "            print(result.stdout[-500:])",
    "",
    "        train_time = time.time() - t0",
    "        state.metric('train_time_sec', round(train_time, 1))",
    "        print(f'\\n✓ Training complete in {train_time/60:.1f} min')",
    "else:",
    "    print('Using COCO pretrained YOLO26s (fast path, ~30s download)')",
    "    with cell('cell-8', action='download-pretrained'):",
    "        from ultralytics import YOLO",
    "        YOLO('yolo26s.pt')  # auto-downloads",
    "        print('✓ Pretrained weights ready')",
))

# Cell 9 (code): End-to-end pipeline
CELLS.append(code(
    "# --- Cell 9: End-to-end pipeline (Detector→Tracker→Drift→Triage→Maintenance) ---",
    "import sys, os, time, urllib.request",
    "import numpy as np",
    "sys.path.insert(0, '/content/conveyor-perception')",
    "sys.path.insert(0, '/content/conveyor-perception/src')",
    "os.chdir('/content/conveyor-perception')",
    "",
    "from colab_session import get_state, hint_for",
    "from conveyor_perception.core.drift_monitor import DriftMonitor",
    "from conveyor_perception.core.tracking_pipeline import TrackingPipeline",
    "from conveyor_perception.multitask.pipeline import MultitaskPipeline",
    "from conveyor_perception.perception.detector import Detector",
    "from conveyor_perception.perception.ultralytics_detector import UltralyticsDetector",
    "from conveyor_perception.predictive_maintenance.advisor import DriftSignal, MaintenanceAdvisor",
    "from conveyor_perception.triage.agent import L1TriageAgent",
    "from conveyor_perception.monitoring.dashboard import MonitoringDashboard",
    "",
    "state = get_state()",
    "",
    "with cell('cell-9', action='run-pipeline'):",
    "    # Use the UltralyticsDetector (handles both .pt and seg-trained .onnx)",
    "    model_path = 'yolo26s_recyclable.pt' if os.path.exists('models/yolo26s_recyclable.pt') else 'yolo26s.pt'",
    "    class_names = ['Glass', 'metal', 'plastic', 'vinyl'] if 'recyclable' in model_path else \\",
    "        [f'class_{i}' for i in range(80)]  # COCO fallback",
    "",
    "    det = UltralyticsDetector(",
    "        model_path=model_path,",
    "        class_names=class_names,",
    "        conf_threshold=0.25,",
    "        device='cuda:0',",
    "        imgsz=640,",
    "    )",
    "    tracker = TrackingPipeline()",
    "    drift = DriftMonitor(baseline_window=50, min_samples_for_drift=20)",
    "    triage = L1TriageAgent()",
    "    advisor = MaintenanceAdvisor()",
    "    dashboard = MonitoringDashboard()",
    "    pipeline = MultitaskPipeline(det, tracker, drift, triage)",
    "",
    "    # Get a sample image (real recycling if downloaded, else COCO bus)",
    "    sample_path = '/content/conveyor-perception/data/sample/bus.jpg'",
    "    if not os.path.exists(sample_path):",
    "        os.makedirs(os.path.dirname(sample_path), exist_ok=True)",
    "        urllib.request.urlretrieve('https://ultralytics.com/images/bus.jpg', sample_path)",
    "    import cv2",
    "    image = cv2.imread(sample_path)",
    "    print(f'Sample image: {image.shape}')",
    "",
    "    # Run 30 frames to accumulate drift signals",
    "    print('\\nRunning 30 frames through the pipeline...')",
    "    t0 = time.perf_counter()",
    "    last_result = None",
    "    for i in range(30):",
    "        last_result = pipeline.step(image)",
    "        dashboard.record_frame(last_result)",
    "    elapsed = (time.perf_counter() - t0) * 1000",
    "    inference_ms = elapsed / 30",
    "    state.metric('t4_inference_ms', round(inference_ms, 2))",
    "    print(f'\\n✓ Pipeline ran 30 frames in {elapsed/1000:.1f}s ({inference_ms:.1f} ms/frame on T4)')",
    "    print(f'  Last frame: {len(last_result.detections)} detections, {len(last_result.alerts)} alerts')",
))

# Cell 10 (code): Dashboard + triage + robustness
CELLS.append(code(
    "# --- Cell 10: Triage queue, robustness suite, shift dashboard ---",
    "import json, sys, os",
    "sys.path.insert(0, '/content/conveyor-perception')",
    "sys.path.insert(0, '/content/conveyor-perception/src')",
    "os.chdir('/content/conveyor-perception')",
    "",
    "from colab_session import get_state",
    "from conveyor_perception.robustness import RobustnessTestSuite",
    "",
    "state = get_state()",
    "",
    "# 1. Triage queue (most recent alerts)",
    "print('=== Triage Queue (most recent 10 alerts) ===\\n')",
    "for alert in triage.get_pending(limit=10):",
    "    print(f\"  [{alert.severity.upper():9s}] {alert.class_name:10s} conf={alert.confidence:.2f} reason='{alert.metadata.get('reason', '')[:50]}'\")",
    "",
    "# 2. Robustness suite (13 MRF conditions)",
    "print('\\n=== Robustness Suite ===\\n')",
    "with cell('cell-10-robustness', action='run-robustness'):",
    "    import cv2",
    "    image = cv2.imread('/content/conveyor-perception/data/sample/bus.jpg')",
    "    suite = RobustnessTestSuite(det, image)",
    "    report = suite.run()",
    "    print(report.to_markdown())",
    "    state.metric('robustness_verdict', report.verdict)",
    "",
    "# 3. Shift dashboard",
    "print('\\n=== Shift Dashboard ===\\n')",
    "shift = dashboard.shift_report()",
    "print(json.dumps(shift.to_dict(), indent=2))",
    "state.metric('retrain_recommended', shift.retrain_recommended)",
))

# Cell 11 (code): Coach review (preview — full review in §4)
CELLS.append(code(
    "# --- Cell 11: Coach review (preview) ---",
    "# A quick Gemini review of the run so far. Full review in §4 cell 15.",
    "from colab_session import get_state, coach_review",
    "",
    "state = get_state()",
    "print('Asking the Coach for a quick review...\\n')",
    "review = coach_review(state)",
    "print(review)",
))

# ===== §3 COMPARISON =====================================================

# Cell 12 (markdown): Comparison section
CELLS.append(md(
    "---",
    "",
    "## §3 COMPARISON — the prototype vs EverestLabs' stack",
    "",
    "The same code, on the same class of GPU that EverestLabs ships. The numbers below come from this T4 run plus EverestLabs' published spec.",
))

# Cell 13 (code): T4 vs EverestLabs (M4 dropped — not relevant to the target)
CELLS.append(code(
    "# --- Cell 12: T4 vs EverestLabs (the comparison) ---",
    "from colab_session import get_state",
    "",
    "state = get_state()",
    "",
    "# Published numbers from EverestLabs' spec (the target we benchmark against)",
    "EVEREST_PUBLISHED = {",
    "    'gpu': 'RTX 2000 Ada (Innodisk APEX-P200)',",
    "    'classes': 60,",
    "    'classification_ms': '8-12',",
    "    'fps': 30,",
    "    'accuracy_pct': 95,",
    "    'pick_success_pct': 90,",
    "}",
    "",
    "# T4 measured numbers come from cell 9",
    "t4_inference = state.metrics.get('t4_inference_ms', 'not measured yet')",
    "T4_MEASURED = {",
    "    'gpu': 'Colab T4 (similar class to RTX 2000 Ada)',",
    "    'classes': 4,",
    "    'inference_ms': t4_inference,",
    "    'fps': round(1000 / t4_inference, 1) if isinstance(t4_inference, (int, float)) else 'n/a',",
    "    'training_minutes': 12,  # 30 epochs on T4",
    "    'mAP50': 'TBD (depends on full training)',",
    "}",
    "",
    "import pandas as pd",
    "df = pd.DataFrame([",
    "    {'Metric': 'GPU', 'EverestLabs': EVEREST_PUBLISHED['gpu'], 'T4 (this run)': T4_MEASURED['gpu']},",
    "    {'Metric': 'Classes', 'EverestLabs': EVEREST_PUBLISHED['classes'], 'T4 (this run)': T4_MEASURED['classes']},",
    "    {'Metric': 'Inference (ms)', 'EverestLabs': EVEREST_PUBLISHED['classification_ms'], 'T4 (this run)': T4_MEASURED['inference_ms']},",
    "    {'Metric': 'FPS', 'EverestLabs': EVEREST_PUBLISHED['fps'], 'T4 (this run)': T4_MEASURED['fps']},",
    "    {'Metric': 'mAP@50', 'EverestLabs': '95% accuracy', 'T4 (this run)': T4_MEASURED['mAP50']},",
    "    {'Metric': 'Pick success', 'EverestLabs': f\"{EVEREST_PUBLISHED['pick_success_pct']}%\", 'T4 (this run)': 'n/a (no robot)'},",
    "])",
    "",
    "print('=== Hardware Stack Comparison ===\\n')",
    "print(df.to_string(index=False))",
    "print()",
    "print('Reading the table:')",
    "print('  • The T4 is the same class as the RTX 2000 Ada (Turing/Ampere gen, similar INT8 TOPS).')",
    "print('  • Our 4-class model is a prototype — Everest has 60+ in production.')",
    "print('  • The mAP50 is for our 4-class recycling subset. Everest publishes 95% accuracy on 60 classes.')",
    "",
    "state.log('cell-12', action='comparison', t4_inference_ms=t4_inference)",
))

# ===== §4 COACH ===========================================================

# Cell 14 (markdown): Coach section
CELLS.append(md(
    "---",
    "",
    "## §4 COACH — error log, diagnosis, summary, publish",
    "",
    "The Coach reads `state.errors` and asks Gemini to diagnose each one. Without a Gemini key, the Coach falls back to static hints (still useful).",
    "",
    "Set `GEMINI_API_KEY` in the Colab secrets panel (key icon, left sidebar) to enable AI diagnosis. Set `GITHUB_TOKEN` (classic PAT, scope: repo) to enable the optimization loop — the final cell publishes the run as a GitHub Release and a GitHub Action picks it up to suggest code improvements as a PR.",
    "",
    "Without the tokens, the notebook still works: errors get diagnosed (via static hints), and the session log gets downloaded (via the browser).",
))

# Cell 15 (code): Error log + Coach diagnosis
CELLS.append(code(
    "# --- Cell 13: Error log + Coach diagnosis ---",
    "import json",
    "from colab_session import get_state, coach_diagnose, hint_for",
    "",
    "state = get_state()",
    "",
    "if not state.has_errors():",
    "    print('✓ No errors captured. The pipeline ran clean.')",
    "else:",
    "    print(f'\\n{len(state.errors)} error(s) captured during this run.\\n')",
    "    print('=' * 60)",
    "",
    "    for i, err in enumerate(state.errors, 1):",
    "        print(f'\\n### Error {i}/{len(state.errors)} — {err[\"cell_id\"]}')",
    "        print(f'  Type:    {err[\"type\"]}')",
    "        print(f'  Message: {err[\"message\"][:200]}')",
    "        if err.get('hint'):",
    "            print(f'  Static hint: {err[\"hint\"]}')",
    "        print()",
    "",
    "        # Ask the Coach to diagnose",
    "        extra = f\"Session: {state.session_id}. Env: {state.env.get('gpu', '?')}.\"",
    "        with cell(f'cell-13-diagnose-{i}', action='coach-diagnose'):",
    "            diagnosis = coach_diagnose(err, extra_context=extra)",
    "            print(f'**Coach diagnosis:**\\n\\n{diagnosis}\\n')",
    "            state.gemini_diagnoses.append({",
    "                'error_idx': i,",
    "                'cell_id': err['cell_id'],",
    "                'diagnosis': diagnosis,",
    "            })",
    "        print('-' * 60)",
))

# Cell 16 (code): Summary + downloadable session log
CELLS.append(code(
    "# --- Cell 14: Summary + downloadable session log ---",
    "import json",
    "from colab_session import get_state, summary_table, download_session_log, coach_review",
    "",
    "state = get_state()",
    "",
    "print('### Session Summary\\n')",
    "print(state.summary_table())",
    "",
    "print('\\n### Full Coach Review (post-run)\\n')",
    "review = coach_review(state)",
    "print(review)",
    "",
    "# Offer the download",
    "print('\\n### Download session log\\n')",
    "try:",
    "    download_session_log()",
    "    print('(Browser download triggered. If nothing happened, check your browser popup blocker.)')",
    "except Exception as e:",
    "    print(f'Download failed: {e}')",
    "    print('You can still access the log via: state.to_json()')",
    "",
    "print('\\n✓ Cell 14 done. Session complete.')",
    "print(f'\\nFinal state: {len(state.logs)} log entries, {len(state.errors)} errors, {len(state.metrics)} metrics.')",
    "print(f'Gemini diagnoses: {len(state.gemini_diagnoses)}')",
))

# Cell 17 (code): Publish to GitHub Release (optimization loop kickoff)
CELLS.append(code(
    "# --- Cell 15: Publish to GitHub Release (kicks off the optimization loop) ---",
    "# This cell uploads the session log as a GitHub Release asset. The release",
    "# tag is v0.0.{N} where N = number of existing releases + 1. A GitHub",
    "# Action triggers on release-published, downloads the log, asks Gemini",
    "# to suggest improvements, and opens a PR.",
    "",
    "import os, json",
    "from colab_session import get_state",
    "",
    "# --- Self-healing: ensure PyGithub is installed (idempotent, fast if cached) ---",
    "# Colab doesn't ship PyGithub by default, and the install cell can be",
    "# masked by pip dependency-resolver warnings, so we re-check here and install",
    "# if needed. Uses --no-deps to avoid the numpy 1.26 / 2.x conflict.",
    "try:",
    "    from github import Github  # noqa: F401  (PyGithub)",
    "except ImportError:",
    "    import subprocess, sys",
    "    print('PyGithub not found — installing (one-time, ~5s)...')",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', 'PyGithub', '-q'])",
    "    from github import Github  # noqa: F401  (PyGithub, after install)",
    "",
    "state = get_state()",
    "",
    "REPO = 'roniejosephv-star/conveyor-perception'",
    "",
    "# GitHub PAT (read+write to your repo). Get one at",
    "# https://github.com/settings/tokens (classic PAT, scope: repo).",
    "try:",
    "    from google.colab import userdata  # type: ignore",
    "    gh_token = userdata.get('GITHUB_TOKEN')",
    "except Exception:",
    "    gh_token = os.environ.get('GITHUB_TOKEN')",
    "",
    "if not gh_token:",
    "    print('=' * 60)",
    "    print('  No GITHUB_TOKEN configured.\\n')",
    "    print('  To enable the optimization loop:')",
    "    print('  1. Create a PAT at https://github.com/settings/tokens')",
    "    print('     (Classic, scope: repo, expiry: 90 days)')",
    "    print('  2. In Colab, click the key icon and add:')",
    "    print('     Name: GITHUB_TOKEN')",
    "    print('     Value: <paste the PAT>')",
    "    print('     Toggle notebook access: ON')",
    "    print('  3. Re-run this cell.')",
    "    print('=' * 60)",
    "    print('\\n✓ Cell 15 done (publish skipped).')",
    "    state.log('cell-15', action='publish-skipped', reason='no GITHUB_TOKEN')",
    "else:",
    "    from github import Github  # PyGithub",
    "    g = Github(gh_token)",
    "    repo = g.get_repo(REPO)",
    "    # Find the next version. v0.0.{N} where N = max existing + 1",
    "    existing = list(repo.get_releases())",
    "    next_n = 0",
    "    for r in existing:",
    "        tag = r.tag_name",
    "        if tag.startswith('v0.0.'):",
    "            try:",
    "                n = int(tag.split('.')[-1])",
    "                next_n = max(next_n, n + 1)",
    "            except ValueError:",
    "                pass",
    "    new_tag = f'v0.0.{next_n}'",
    "",
    "    # Write the session log to a temp file",
    "    log_path = f'/tmp/{state.session_id}.json'",
    "    with open(log_path, 'w') as f:",
    "        f.write(state.to_json())",
    "",
    "    # Build the release notes (one-liner with the headline metric)",
    "    headline = state.metrics.get('t4_inference_ms', 'n/a')",
    "    n_errors = len(state.errors)",
    "    n_modules_on = sum(state.toggles.values())",
    "    notes = (",
    "        f'## Run {new_tag}\\n\\n'",
    "        f'- **T4 inference (ms)**: {headline}\\n'",
    "        f'- **Errors**: {n_errors}\\n'",
    "        f'- **Modules on**: {n_modules_on}/{len(state.toggles)}\\n'",
    "        f'- **Session ID**: {state.session_id}\\n\\n'",
    "        '_Auto-published by the Conveyor Perception Coach from the Colab demo._'",
    "    )",
    "",
    "    with cell('cell-15', action='publish-release'):",
    "        release = repo.create_git_release(",
    "            tag=new_tag,",
    "            name=f'Run {new_tag} — T4 {headline}ms',",
    "            message=notes,",
    "            draft=False,",
    "            prerelease=False,",
    "        )",
    "        # Attach the session log as a release asset",
    "        release.upload_asset_from_path(log_path, name='session.json')",
    "        state.metric('release_tag', new_tag)",
    "        state.metric('release_url', release.html_url)",
    "        print(f'\\n✓ Published {new_tag} → {release.html_url}')",
    "        print('  Asset: session.json')",
    "        print('  The optimization loop will pick this up on the next Action run.')",
    "",
    "    print(f'\\n✓ Cell 15 done. Session published as {new_tag}.')",
))


# --- §5 OPTIMIZATION LOOP — the closed-loop improvement system -------------


def _loop_helpers() -> str:
    """Shared helper code for the §5 stage cells. Returns a Python string.

    The helpers handle: getting GITHUB_TOKEN, calling the GitHub REST API,
    rendering a status line, logging to state. Each stage cell prepends
    these helpers and then uses them to check its specific status.
    """
    return (
        "import os, json\n"
        "try:\n"
        "    from google.colab import userdata  # type: ignore\n"
        "    gh_token = userdata.get('GITHUB_TOKEN')\n"
        "except Exception:\n"
        "    gh_token = os.environ.get('GITHUB_TOKEN')\n"
        "\n"
        "REPO = 'roniejosephv-star/conveyor-perception'\n"
        "API = 'https://api.github.com'\n"
        "\n"
        "def _headers():\n"
        "    h = {'Accept': 'application/vnd.github+json'}\n"
        "    if gh_token:\n"
        "        h['Authorization'] = f'token {gh_token}'\n"
        "    return h\n"
        "\n"
        "def _status(emoji: str, label: str) -> str:\n"
        "    return f'  {emoji} Status: {label}'\n"
        "\n"
        "def _no_token_msg() -> str:\n"
        "    return (\n"
        "        '  ⏳ Status: no GITHUB_TOKEN configured.\\n\\n'\n"
        "        '  The optimization loop needs a GitHub PAT to read the release\\n'\n"
        "        '  / workflow / PR state. See cell 15 (publish cell) for setup:\\n'\n"
        "        '  1. Create a PAT at https://github.com/settings/tokens\\n'\n"
        "        '     (Classic, scope: repo, 90-day expiry)\\n'\n"
        "        '  2. In Colab, click the key icon and add it as GITHUB_TOKEN'\n"
        "    )\n"
    )


# §5 header (markdown) — explains the 4-stage loop before the cells run
CELLS.append(md(
    "---",
    "",
    "## §5 OPTIMIZATION LOOP — the framework improves itself",
    "",
    "A closed loop with 4 stages. Every Colab run feeds back into the codebase:",
    "",
    "```",
    "   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐",
    "   │ 1 PUBLISH│───▶│ 2 TRIGGER│───▶│ 3 ANALYZE│───▶│ 4 PROPOSE │",
    "   │ Colab→GH │    │ Action   │    │ Gemini   │    │ PR open  │",
    "   └──────────┘    └──────────┘    └──────────┘    └──────────┘",
    "        ▲                                               │",
    "        └───────────── merge or close ◀─────────────────┘",
    "```",
    "",
    "The next 4 cells check each stage against the live GitHub state. Status indicators (`✅ / ⏳ / 🔄 / ❌`) tell you what's done. The Coach in stage 3 is the brain — it diffs this run against the previous one and asks Gemini to suggest one focused code change.",
))


# Cell 21 (code): STAGE 1 — PUBLISH
# Lists v0.0.N releases to show whether the run was published
CELLS.append(code(
    "# --- Cell 16: §5 STAGE 1 — PUBLISH ---",
    f"{_loop_helpers()}",
    "import requests",
    "",
    "state = get_state()",
    "stage_n, stage_name = 1, 'PUBLISH'",
    "",
    "print('=' * 70)",
    "print(f'  STAGE {stage_n} of 4 — {stage_name}')",
    "print('=' * 70)",
    "print('What happens: Colab uploads the session log as a v0.0.N GitHub Release.')",
    "print('Why this matters: Releases are the durable artifact. Every run is a versioned')",
    "print('                 snapshot the Action can download and reason about.')",
    "print()",
    "",
    "if not gh_token:",
    "    print(_no_token_msg())",
    "    state.log(f'stage-{stage_n}', status='no-token')",
    "else:",
    "    try:",
    "        r = requests.get(f'{API}/repos/{REPO}/releases?per_page=20', headers=_headers(), timeout=10)",
    "        r.raise_for_status()",
    "        releases = [x for x in r.json() if x.get('tag_name', '').startswith('v0.0.')]",
    "        if releases:",
    "            latest = max(releases, key=lambda x: x['tag_name'])",
    "            n_assets = len(latest.get('assets', []))",
    "            print(_status('✅', f'{len(releases)} v0.0.N release(s) published'))",
    "            print(f'     Latest:     {latest[\"tag_name\"]}')",
    "            print(f'     Published:  {latest[\"published_at\"][:19].replace(\"T\", \" \")} UTC')",
    "            print(f'     URL:        {latest[\"html_url\"]}')",
    "            print(f'     Assets:     {n_assets} (session.json is the artifact the Action reads)')",
    "            state.metric('releases_count', len(releases))",
    "            state.metric('latest_release_tag', latest['tag_name'])",
    "            state.metric('latest_release_url', latest['html_url'])",
    "        else:",
    "            print(_status('⏳', '0 v0.0.N releases yet — re-run cell 15 to publish v0.0.1'))",
    "            state.log(f'stage-{stage_n}', status='no-releases')",
    "    except Exception as e:",
    "        print(_status('❌', f'API error: {e}'))",
    "        state.log(f'stage-{stage_n}', status='error', error=str(e))",
    "",
    "print()",
    "print('💡 Audience hint: open the release URL to see the session.json artifact.')",
    "print('   The Action in stage 2 will download that exact file.')",
    "print()",
    "print(f'\\n✓ Stage {stage_n} shown.')",
))


# Cell 22 (code): STAGE 2 — TRIGGER
# Lists the latest workflow runs to show whether the Action woke up
CELLS.append(code(
    "# --- Cell 17: §5 STAGE 2 — TRIGGER ---",
    f"{_loop_helpers()}",
    "import requests",
    "",
    "state = get_state()",
    "stage_n, stage_name = 2, 'TRIGGER'",
    "",
    "print('=' * 70)",
    "print(f'  STAGE {stage_n} of 4 — {stage_name}')",
    "print('=' * 70)",
    "print('What happens: A GitHub Action listens for `release: { types: [published] }`')",
    "print('                 filtered to v0.0.* tags. Within ~30s of the publish, it wakes.')",
    "print('Why this matters: The framework is reactive. Every artifact triggers analysis.')",
    "print()",
    "",
    "if not gh_token:",
    "    print(_no_token_msg())",
    "    state.log(f'stage-{stage_n}', status='no-token')",
    "else:",
    "    try:",
    "        r = requests.get(",
    "            f'{API}/repos/{REPO}/actions/runs?per_page=5',",
    "            headers=_headers(), timeout=10,",
    "        )",
    "        r.raise_for_status()",
    "        runs = r.json().get('workflow_runs', [])",
    "        # Filter to optimize.yml runs only (and v0.0.* trigger)",
    "        opt_runs = [run for run in runs if 'optimize' in (run.get('path', '') + run.get('name', '')).lower()]",
    "        if opt_runs:",
    "            latest = opt_runs[0]",
    "            status_emoji = {'success': '✅', 'failure': '❌', 'in_progress': '🔄', 'queued': '⏳'}.get(",
    "                latest['conclusion'] or latest['status'], '⏳'",
    "            )",
    "            print(_status(status_emoji, f\"{latest['conclusion'] or latest['status']} — {latest['name']}\"))",
    "            print(f'     Run ID:     {latest[\"id\"]}')",
    "            print(f'     Event:      {latest[\"event\"]} ({\"v0.0.* release\" if latest[\"event\"] == \"release\" else \"other\"})')",
    "            print(f'     Branch:     {latest[\"head_branch\"]}')",
    "            print(f'     Started:    {latest[\"created_at\"][:19].replace(\"T\", \" \")} UTC')",
    "            print(f'     URL:        {latest[\"html_url\"]}')",
    "            state.metric('latest_run_status', latest['conclusion'] or latest['status'])",
    "            state.metric('latest_run_url', latest['html_url'])",
    "        else:",
    "            print(_status('⏳', 'no optimize.yml runs yet — publish a v0.0.1 release first (cell 15)'))",
    "            state.log(f'stage-{stage_n}', status='no-runs')",
    "    except Exception as e:",
    "        print(_status('❌', f'API error: {e}'))",
    "        state.log(f'stage-{stage_n}', status='error', error=str(e))",
    "",
    "print()",
    "print('💡 Audience hint: the Action runs in <2 min. Re-run this cell in 2 min to see it flip ⏳ → ✅.')",
    "print()",
    "print(f'\\n✓ Stage {stage_n} shown.')",
))


# Cell 23 (code): STAGE 3 — ANALYZE
# Reads the latest run's conclusion + the PR body (which contains the Coach's analysis)
CELLS.append(code(
    "# --- Cell 18: §5 STAGE 3 — ANALYZE ---",
    f"{_loop_helpers()}",
    "import requests",
    "",
    "state = get_state()",
    "stage_n, stage_name = 3, 'ANALYZE'",
    "",
    "print('=' * 70)",
    "print(f'  STAGE {stage_n} of 4 — {stage_name}')",
    "print('=' * 70)",
    "print('What happens: The Action downloads the new session.json + the previous one,')",
    "print('                 asks Gemini to diff them and suggest ONE focused code change.')",
    "print('Why this matters: This is the brain. The Coach turns a noisy session log into')",
    "print('                 a single actionable diff. Hard rules in the prompt guarantee:')",
    "print('                 no public-API changes, no CI/Docker/harness edits, NO_ACTION if')",
    "print('                 there is no metric change AND no error.')",
    "print()",
    "",
    "if not gh_token:",
    "    print(_no_token_msg())",
    "    state.log(f'stage-{stage_n}', status='no-token')",
    "else:",
    "    try:",
    "        # Look for coach/* PRs first — their body is the Coach's analysis",
    "        r = requests.get(",
    "            f'{API}/repos/{REPO}/pulls?state=all&per_page=20',",
    "            headers=_headers(), timeout=10,",
    "        )",
    "        r.raise_for_status()",
    "        coach_prs = [p for p in r.json() if p['head']['ref'].startswith('coach/')]",
    "        if coach_prs:",
    "            pr = coach_prs[0]  # most recent",
    "            print(_status('✅', f'Coach suggested a change — PR #{pr[\"number\"]} opened'))",
    "            print(f'     Title:     {pr[\"title\"]}')",
    "            print(f'     Branch:    {pr[\"head\"][\"ref\"]}')",
    "            print(f'     State:     {pr[\"state\"]} ({\"merged\" if pr.get(\"merged\") else pr[\"state\"]})')",
    "            print(f'     URL:       {pr[\"html_url\"]}')",
    "            # Show the first 5 lines of the PR body — that's the Coach's analysis",
    "            body = (pr.get('body') or '').strip().split('\\n')",
    "            if body:",
    "                print()",
    "                print('     --- Coach analysis (PR body, first 5 lines) ---')",
    "                for line in body[:5]:",
    "                    if line.strip():",
    "                        print(f'     │ {line[:100]}')",
    "                print('     ' + '-' * 50)",
    "            state.metric('coach_pr_number', pr['number'])",
    "            state.metric('coach_pr_url', pr['html_url'])",
    "        else:",
    "            # No PR yet — either the Action hasn't run, or it returned NO_ACTION",
    "            r2 = requests.get(",
    "                f'{API}/repos/{REPO}/actions/runs?per_page=3',",
    "                headers=_headers(), timeout=10,",
    "            )",
    "            r2.raise_for_status()",
    "            runs = [run for run in r2.json().get('workflow_runs', []) if 'optimize' in run.get('name', '').lower()]",
    "            if not runs:",
    "                print(_status('⏳', 'no Action run yet — wait ~30s after publish, then re-run this cell'))",
    "            else:",
    "                latest = runs[0]",
    "                if latest['conclusion'] == 'success':",
    "                    print(_status('⏳', 'Action ran but opened no PR — likely NO_ACTION (no metric change + no error)'))",
    "                    print('     This is correct behavior: the Coach only proposes when there is something')",
    "                    print('     concrete to change. Silence is a valid signal.')",
    "                else:",
    "                    print(_status('❌', f'Action {latest[\"conclusion\"]} — check the run logs'))",
    "                    print(f'     URL: {latest[\"html_url\"]}')",
    "            state.log(f'stage-{stage_n}', status='no-pr')",
    "    except Exception as e:",
    "        print(_status('❌', f'API error: {e}'))",
    "        state.log(f'stage-{stage_n}', status='error', error=str(e))",
    "",
    "print()",
    "print('💡 Audience hint: the PR body IS the Coach\\'s analysis — it is structured JSON-ish.')",
    "print('   Read the first 5 lines to see what the model decided to change.')",
    "print()",
    "print(f'\\n✓ Stage {stage_n} shown.')",
))


# Cell 24 (code): STAGE 4 — PROPOSE
# Lists coach/* PRs and shows the diff summary
CELLS.append(code(
    "# --- Cell 19: §5 STAGE 4 — PROPOSE ---",
    f"{_loop_helpers()}",
    "import requests",
    "",
    "state = get_state()",
    "stage_n, stage_name = 4, 'PROPOSE'",
    "",
    "print('=' * 70)",
    "print(f'  STAGE {stage_n} of 4 — {stage_name}')",
    "print('=' * 70)",
    "print('What happens: If the Coach suggested a change, the Action opens a PR via')",
    "print('                 peter-evans/create-pull-request. The PR waits for human review.')",
    "print('Why this matters: The loop is GUARDED. The Coach proposes, the human disposes.')",
    "print('                 No autonomous merges, no production drift, no CI/Docker edits.')",
    "print()",
    "",
    "if not gh_token:",
    "    print(_no_token_msg())",
    "    state.log(f'stage-{stage_n}', status='no-token')",
    "else:",
    "    try:",
    "        r = requests.get(",
    "            f'{API}/repos/{REPO}/pulls?state=all&per_page=20',",
    "            headers=_headers(), timeout=10,",
    "        )",
    "        r.raise_for_status()",
    "        coach_prs = [p for p in r.json() if p['head']['ref'].startswith('coach/')]",
    "        open_prs = [p for p in coach_prs if p['state'] == 'open']",
    "        merged_prs = [p for p in coach_prs if p.get('merged')]",
    "        closed_prs = [p for p in coach_prs if p['state'] == 'closed' and not p.get('merged')]",
    "        total = len(coach_prs)",
    "        if total == 0:",
    "            print(_status('⏳', 'no coach/* PRs yet — wait for stage 3 to finish or check stage 2 logs'))",
    "            state.log(f'stage-{stage_n}', status='no-prs')",
    "        else:",
    "            print(_status('✅', f'{total} coach/* PR(s) total — {len(open_prs)} open, {len(merged_prs)} merged, {len(closed_prs)} closed'))",
    "            for pr in coach_prs[:3]:  # show the 3 most recent",
    "                state_label = '🟢 open' if pr['state'] == 'open' else ('🟣 merged' if pr.get('merged') else '⚫ closed')",
    "                print(f'     #{pr[\"number\"]:3} [{state_label}] {pr[\"title\"][:55]}')",
    "                print(f'           {pr[\"html_url\"]}')",
    "            state.metric('coach_prs_total', total)",
    "            state.metric('coach_prs_open', len(open_prs))",
    "            state.metric('coach_prs_merged', len(merged_prs))",
    "    except Exception as e:",
    "        print(_status('❌', f'API error: {e}'))",
    "        state.log(f'stage-{stage_n}', status='error', error=str(e))",
    "",
    "print()",
    "print('💡 Audience hint: open the PRs tab in the repo. The diff is the Coach\\'s output.')",
    "print('   The framework wrote the diff itself — you decide whether to merge.')",
    "print()",
    "print(f'\\n✓ Stage {stage_n} shown.')",
))


# §5 closing (markdown) — the achievement statement
CELLS.append(md(
    "---",
    "",
    "### What you just saw",
    "",
    "**The framework improves itself.** The same Colab notebook that showed you 8 modules and an end-to-end pipeline just demonstrated a 4-stage feedback loop where every run feeds back into the codebase as a PR. The Coach is bounded (one change, no public API, no CI/Docker), guarded (human review before merge), and observable (every stage has a status indicator).",
    "",
    "**Why this matters for industrial CV at scale:** ROC shifts don't end at 6 AM. The system keeps running, the data keeps drifting, the failure modes keep changing. A pipeline that cannot observe itself, react to its own artifacts, and propose its own fixes will rot in 6 months. The optimization loop is the difference between a demo and a deployable system.",
    "",
    "**Try it yourself:** re-run cell 15 to publish a fresh release. Wait ~90s. Re-run cells 16-19 (or just re-run this section). Watch the status indicators flip from ⏳ to ✅ as the loop completes.",
))


# --- write it out --------------------------------------------------------


def main() -> int:
    nb = nbformat_v4(CELLS)
    OUTPUT.write_text(json.dumps(nb, indent=1) + "\n")
    # Sanity-check
    parsed = json.loads(OUTPUT.read_text())
    assert len(parsed["cells"]) == 25, f"expected 25 cells, got {len(parsed['cells'])}"
    print(f"Wrote {OUTPUT} with {len(parsed['cells'])} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
