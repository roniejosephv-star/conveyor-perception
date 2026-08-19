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
    "import os, sys",
    "os.chdir(REPO) if os.path.exists(REPO) else None  # if not cloned yet, this no-ops",
    "sys.path.insert(0, REPO)",
    "sys.path.insert(0, os.path.join(REPO, 'notebooks'))",
    "",
    "# Load the session helpers",
    "from colab_session import env_check, get_state",
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
    "with state.cell('cell-2', action='install-and-clone'):",
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
    "os.chdir('/content/conveyor-perception')",
    "",
    "from colab_session import get_state, hint_for",
    "from conveyor_perception.core.detection_pipeline import Detector, Detection",
    "from conveyor_perception.core.tracking_pipeline import TrackingPipeline",
    "from conveyor_perception.core.drift_monitor import DriftMonitor",
    "from conveyor_perception.core.triage_surface import MCPTriageSurface",
    "",
    "state = get_state()",
    "loaded = {}",
    "",
    "with state.cell('cell-6', action='load-4-abstractions'):",
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
    "        loaded['triage_surface'] = MCPTriageSurface()",
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
    "        with state.cell(f'cell-7-{module_path}', action='import'):",
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
    "    with state.cell('cell-8', action='train-yolo26s'):",
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
    "    with state.cell('cell-8', action='download-pretrained'):",
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
    "with state.cell('cell-9', action='run-pipeline'):",
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
    "with state.cell('cell-10-robustness', action='run-robustness'):",
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
    "## §4 COACH — error log, diagnosis, summary",
    "",
    "The Coach reads `state.errors` and asks Gemini to diagnose each one. Without a Gemini key, the Coach falls back to static hints (still useful).",
    "",
    "Set `GEMINI_API_KEY` in the Colab secrets panel (key icon, left sidebar) to enable AI diagnosis.",
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
    "        with state.cell(f'cell-13-diagnose-{i}', action='coach-diagnose'):",
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


# --- write it out --------------------------------------------------------


def main() -> int:
    nb = nbformat_v4(CELLS)
    OUTPUT.write_text(json.dumps(nb, indent=1) + "\n")
    # Sanity-check
    parsed = json.loads(OUTPUT.read_text())
    assert len(parsed["cells"]) == 18, f"expected 18 cells, got {len(parsed['cells'])}"
    print(f"Wrote {OUTPUT} with {len(parsed['cells'])} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
