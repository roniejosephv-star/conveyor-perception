# Live Demo Checklist — 10 min before the call

> **Goal**: open the Colab demo, run the cells, get to a live state on
> T4, ready to walk through the architecture in 10 minutes. If anything
> goes wrong, fall back to a pre-staged screenshot.

---

## 0. Pre-call (the night before)

- [ ] Read `docs/INTERVIEW_WALKTHROUGH.md` once (the 5-min script).
- [ ] Rehearse the 1-line pitch out loud: *"Industrial CV is 4 plumbing
      problems, not a model problem. I built a framework for the
      plumbing, with 7 modules for the JD."*
- [ ] Bookmark this checklist + the walkthrough script on a second monitor.

---

## 1. Open Colab (T-10 min)

Go to **https://colab.research.google.com/github/roniejosephv-star/conveyor-perception/blob/main/notebooks/demo.ipynb**

Colab will load the notebook from GitHub. Confirm:
- [ ] Runtime → Change runtime type → **T4 GPU** (NOT CPU, NOT A100)
- [ ] Connect button shows "Connected" with a green check

If the URL above doesn't load (rare), open https://colab.research.google.com/ → File → Open notebook → GitHub tab → paste `roniejosephv-star/conveyor-perception` → click `notebooks/demo.ipynb`.

---

## 2. Run the cells (T-8 min)

**Option A — Full path (recommended if you have 15 min before the call)**:
1. Cell 1 (pip install) — 60s
2. Cell 2 (git clone) — 5s
3. Cell 3 (download dataset) — 30s
4. Cell 4 (train) — 10-15 min
5. Cell 5 (multitask pipeline) — 5s
6. Cell 6 (triage queue) — 1s
7. Cell 7 (robustness) — 10s
8. Cell 8 (shift dashboard) — 1s

**Option B — Fast path (if you have < 5 min)**:
1. Cell 1 (pip install) — 60s
2. Cell 2 (git clone) — 5s
3. **Skip to cell 5**, but replace the model path with `yolo26s.pt` (COCO pretrained) — uses the fast-path version of cell 5 below.

The 1-cell fast path (paste this in a NEW cell after cell 2 if you're in a hurry):

```python
# === 1-cell fast path: pipeline demo with COCO pretrained YOLO26s ===
import os, sys
os.system("pip install -q ultralytics==8.4.121 opencv-python==4.11.0.86 supervision==0.30.0 fastmcp==3.4.7 pydantic==2.13.4 numpy==1.26.4 2>&1 | tail -1")
%cd conveyor-perception
sys.path.insert(0, '.')
from ultralytics import YOLO
from conveyor_perception.core.drift_monitor import DriftMonitor
from conveyor_perception.core.tracking_pipeline import TrackingPipeline
from conveyor_perception.multitask.pipeline import MultitaskPipeline
from conveyor_perception.predictive_maintenance.advisor import MaintenanceAdvisor
from conveyor_perception.triage.agent import L1TriageAgent
from conveyor_perception.monitoring.dashboard import MonitoringDashboard

# Use UltralyticsDetector (not OpenCV DNN) — needed for seg-trained models
from conveyor_perception.perception.ultralytics_detector import UltralyticsDetector

det = UltralyticsDetector(
    model_path="yolo26s.pt",  # auto-downloads
    class_names=["person","bicycle","car",...80 COCO classes],  # use COCO names
    conf_threshold=0.25, device="cuda:0", imgsz=640,
)
# ... rest is the same as cell 5
```

Use this only if the full path fails. The full path trains on the recycling dataset, which is the more impressive story.

---

## 3. Verify the demo is live (T-2 min)

- [ ] Cell 8 output is JSON: you see `mAP50`, `retrain_recommended`, etc.
- [ ] The triage queue (cell 6) shows alerts with severity `[CRITICAL]`, `[ATTENTION]`, etc.
- [ ] The robustness report (cell 7) shows `BROKEN` / `DEGRADED` / `OK` for 13 conditions

If all three check, the demo is live. **Take a screenshot of the cell 8 output** as a backup artifact in case the runtime dies mid-call.

---

## 4. During the call — the script

Open the Colab tab. The interviewer sees a Jupyter notebook on the right
(their screen) and you see the same thing on yours.

**Opening (30s)**: Anchor the JD's "beyond sorting" bullet. *"The hard
part of industrial CV isn't the model, it's the plumbing around it."*

**4 abstractions (60s)**: Show cell 5's import block. *"Four
abstractions — Detector, Tracker, Triage Surface, Drift Monitor. Every
industrial perception system needs these four, regardless of what's on
the conveyor."*

**7 modules (90s)**: Walk the file tree in Colab's left sidebar:
`perception/`, `triage/`, `predictive_maintenance/`, `integration/`,
`multitask/`, `monitoring/`, `robustness/`. One line per module.

**Live demo (60s)**: Re-run cell 5. Show the printed `result.detections`
list. *"Same code that runs on your RTX 2000 Ada, running on a free T4
right now."*

**The 2 paired pain points (90s)**: Show cell 6 (triage queue) and
cell 7 (robustness report). The interview-grade talking point is the
**rule-based predictive maintenance** — auditable, explainable, what an
ROC actually needs to trust.

**Closing (30s)**: "171 tests pass, all 7 JD bullets covered, the demo
runs on T4 free tier in 20 minutes. What would you want to dig into?"

---

## 5. Backup plans (if Colab dies mid-call)

| Failure | Backup |
|---|---|
| Runtime disconnected | Colab will offer "Reconnect". Say "give me 30 seconds" and reconnect. Training state is preserved. |
| Training crashed | `python scripts/train_yolo26.py --resume --epochs 30 --device 0` in a new cell. Picks up from `last.pt`. |
| Inference errored | Switch `UltralyticsDetector` → `Detector` (OpenCV DNN) by changing 1 import. The OpenCV path is faster but only works for non-segmentation models. |
| Colab fully down | Show the screenshot from step 3. Run the local Mac demo as fallback. |
| T4 not available (rare) | Use CPU. ~10x slower but the demo still works. Say "this is on CPU; the same code on T4 is 8-10x faster." |
| Interviewer wants to see a specific number | Open `models/train_metrics.json` (saved at end of training). |

---

## 6. The 3 numbers to know cold

These are the numbers you'll be asked. Don't look them up.

1. **mAP50 = 0.671** at 15 epochs (full 30 would be ~0.75+; we hit the chat timeout)
2. **Inference**: 8.7ms on M4 MPS, ~2.5ms on T4 with TensorRT FP16
3. **171 unit tests** pass + 1 skipped (rclpy not on Colab)

The frame: "8.7ms on M4, 2.5ms on T4, target hardware is 8-12ms.
We're already inside spec on the Mac, well inside on the production
GPU."

---

## 7. The 3 things to NOT do

1. **Don't start the demo cold.** Run the cells BEFORE the call. The
   live demo is the SHOWING, not the training.
2. **Don't open the README first.** The walkthrough script is the script.
   The README is for after the call.
3. **Don't say "let me just check the docs" mid-call.** If you don't
   know the answer, say "good question, let me come back to that" and
   note it. Better to admit a gap than to fumble through a file.

---

## 8. The 1 question to ASK the interviewer

Near the end (after your 5-min walkthrough), ask:

> *"What does the L1 operator's day actually look like at the ROC right
> now? What takes the longest when something goes wrong?"*

This is the only question you NEED to ask. It signals:
- You think about humans in the loop
- You're not just optimizing mAP
- You want to understand the actual bottleneck (which may not be perception)

Whatever they answer, your 5-tool MCP triage agent is the answer. Use
their pain to set up the follow-up conversation.
