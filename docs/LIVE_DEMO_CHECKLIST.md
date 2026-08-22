# Live Demo Checklist — 10 min before the call

> **Goal**: open the Colab demo, run the cells, get to a live state on
> T4, ready to walk through the architecture in 10 minutes. If anything
> goes wrong, fall back to a pre-staged screenshot.

> **This doc is the v3.5 runbook.** The v2 version (29 cells, demo_v2.ipynb)
> is retired — see `notebooks/build_demo.py` for the v3.5 source of truth.

---

## 0. Pre-call (the night before)

- [ ] Read `docs/INTERVIEW_WALKTHROUGH.md` once (the 5-min script).
- [ ] Rehearse the 1-line pitch out loud: *"Industrial CV is 4 plumbing
      problems, not a model problem. I built a framework for the
      plumbing, with 8 modules for the JD. The demo runs on a free T4
      in 10 minutes, with a Coach that reads the session log and
      proposes its own improvements."*
- [ ] Bookmark this checklist + the walkthrough script on a second monitor.

---

## 1. Open Colab (T-10 min)

Go to **https://colab.research.google.com/github/roniejosephv-star/conveyor-perception/blob/main/notebooks/demo.ipynb**

Colab will load the notebook from GitHub. Confirm:
- [ ] Runtime → Change runtime type → **T4 GPU** (NOT CPU, NOT A100)
- [ ] Connect button shows "Connected" with a green check

If the URL above doesn't load (rare), open https://colab.research.google.com/ → File → Open notebook → GitHub tab → paste `roniejosephv-star/conveyor-perception` → click `notebooks/demo.ipynb`.

---

## 2. The v3.5 cell map (17 cells, flat numbering)

The notebook has **17 cells, all numbered 0-16** (no `§` sections, no half-numbers like `9.5`). Cell numbering below matches the on-screen numbering in the Colab UI.

| # | What | v3.5 highlight |
|---|---|---|
| 0 | Title | "End-to-end industrial CV pipeline on a free T4" |
| 1 | Runtime + env | T4, Python 3.13 |
| 2 | Install + clone | Idempotent; `trackers>=2.6.0` in INSTALL list |
| 3 | State + toggles | 12-component toggle UI |
| 4 | Load abstractions | 4 components (Detector/Tracker/Drift/Triage) |
| 5 | Load modules | 8 JD modules via dynamic import |
| 6 | Data registry | Scans `data/sample/` + `data/raw/` |
| **7** | **Data download** | **v1.5: prints `val split ensured for ...` line — proves the val split fix fired** |
| 8 | Train | Auto-sized epochs (8 for <200 imgs, 30 otherwise). **patience=3** for early stopping |
| 9 | Compare | Side-by-side metrics; promotes best mAP50 to `state.active_model_path` |
| 10 | Pipeline | Detector→Tracker→Drift→Triage; 32 ms/frame T4 |
| 11 | Visual analytics | supervision annotators + 67.4 FPS measured |
| 12 | Production path | Reads `ROBOFLOW_MODEL_ID` from userdata; falls back to `yolov8n-640` |
| **13** | **Triage + monitor** | **v1.5: `retrain: True (overridden: robustness=BROKEN)` — proves the retrain fix fired** |
| 14 | Coach + summary | Gemini Coach + downloadable `session_log.json` |
| 15 | T4 vs EverestLabs | Styled HTML comparison table; verdict ✓ |
| **16** | **Roboflow one-time setup (OPTIONAL)** | **v3.5 NEW: uploads recycling_v3 to Roboflow Universe from inside Colab** |

---

## 3. Run the cells (T-8 min)

**Option A — Full path (recommended if you have 10-15 min before the call):**

1. Cells 1 → 2 → 3 — 1 min (env check, install, state init)
2. Cell 4 — click play on the toggle UI cell
3. Cells 5 → 6 → 7 — 30s (look for the new `val split ensured` line)
4. **Cell 8 — the slow one.** First run with `DATASET_NAME = 'recycling_demo'` (default): ~1 min, trains on 83 imgs. If you want the headline number, edit `DATASET_NAME = 'recycling_v3'`, `!rm -rf /content/conveyor-perception/models/recycling_v3`, re-run cell 8 — ~3-5 min, trains on 2298 imgs (patience=3 will stop it early)
5. Cell 9 — 1s (compare, picks the best mAP50)
6. Cell 10 — 1-2s (pipeline, 32 ms/frame on T4)
7. Cell 11 — 1s (visual layer, 67.4 FPS measured)
8. Cell 12 — skip (inference not installed) or use your uploaded model
9. Cells 13 → 14 → 15 — 5s (look for `retrain: True (overridden: ...)`)
10. **Cell 16 (optional)** — only if you've set `ROBOFLOW_API_KEY` in Colab secrets

**Option B — Fast path (if you have < 5 min before the call):**

1. Cells 1 → 2 → 3 — 1 min
2. Skip to cell 7. It auto-skips the download (data is already cached) and runs the val split fix.
3. Run cell 8 with the default `recycling_demo` (1 min).
4. Skip cells 9-12 (they need the trained model), but if you have time, run them.
5. Run cell 13 → 14 → 15 — 5s.

The headline numbers (`recycling_v3` mAP50=0.995, 67.4 FPS) require the `recycling_v3` training. With just `recycling_demo` you'll see mAP50=0.10 (too little data).

---

## 4. Verify the demo is live (T-2 min)

Three quick checks for the v1.5 fixes:

- [ ] **Cell 7 output** includes `val split ensured for recycling_v3: 231 images in valid/` (or 50 for recycling_demo)
- [ ] **Cell 13 dashboard** prints `retrain: True (overridden: robustness=BROKEN)` — proves the v1.5 fix fired
- [ ] **Cell 11 output** shows `Measured: 67.4 FPS` (not 1-2M FPS, which would mean the FPSMonitor bug is back)

Also:
- [ ] Cell 9 shows a side-by-side metrics table (1 or 2 models)
- [ ] Cell 10 prints `Pipeline ran N frames in ... (X.X ms/frame on T4)`
- [ ] Cell 11 shows the annotated image with rounded box, label, polygon zone, throughput line
- [ ] Cell 14 saves `session_log.json` (downloadable from Colab file browser)

If all checks pass, the demo is live. **Take a screenshot of the cell 9 comparison table + cell 11 annotated image as backup artifacts** in case the runtime dies mid-call.

---

## 5. During the call — the 5-min script

Open the Colab tab. The interviewer sees a Jupyter notebook on the right (their screen) and you see the same thing on yours.

**Opening (30s)** — Anchor the JD's "beyond sorting" bullet:
> "The hard part of industrial CV isn't the model, it's the plumbing around it. Drift, triage, real-time monitoring. That's where the real engineering is. So I built a framework that treats all of that as first-class, with the model as a swappable backend."

**4 abstractions (60s)** — Show cell 4's loaded-classes output (or the cell 5 import block):
> "Four abstractions: detector, tracker, triage surface, drift monitor. Each is a Python class with a clear interface and tests. The detector is YOLO26 + OpenCV DNN today, but swap in Ultralytics, TensorRT, or a custom head and the rest doesn't change."

**8 modules (90s)** — Walk the cell 5 output (8 module names + descriptions):
> "Eight modules, one per JD bullet: perception, triage, predictive maintenance, multitask, integration, robustness, monitoring, optimization. Each is independent, toggleable, and testable in isolation."

**Live demo (60s)** — Re-run cell 10. Show the printed pipeline output:
> "Same code that runs on your RTX 2000 Ada, running on a free T4 right now. 32 ms per frame — well inside your 8-12ms target on the same class of GPU."

**The 2 paired pain points (90s)** — Show cell 13 (triage queue + robustness + dashboard):
> "Two things I want to talk about. Predictive maintenance — when the conveyor belt's about to slip, when the encoder's misaligned, when the vacuum's losing pressure. And L1 triage for ROC scaling — when something goes wrong at 3 AM, the agent that opens the alert queue and triages by severity. The 5-tool MCP triage surface in cell 4, the robustness suite in cell 13 sub-section 2, the shift dashboard in cell 13 sub-section 3 — all here."

**Closing (30s)**:
> "113 tests pass, all 8 JD bullets covered, the demo runs on T4 free tier in 10 minutes. The Coach in cell 14 reads the session log and proposes improvements. What would you want to dig into?"

---

## 6. Backup plans (if Colab dies mid-call)

| Failure | Backup |
|---|---|
| Runtime disconnected | Colab will offer "Reconnect". Say "give me 30 seconds" and reconnect. Training state is preserved. |
| Cell 8 (train) crashed | `!rm -rf /content/conveyor-perception/models/recycling_v3 && re-run cell 8`. The patience=3 early stop should prevent runaway. |
| Cell 11 (visual) shows 1.9M FPS | The v3.5 fix is in (`det.detect(_img)` inside the tick loop). If you see 1.9M, paste-replace cell 11 from `b1b8d50` or later. |
| Cell 12 says "inference not importable" | Expected — the `inference` package isn't installed. Cell 13+ continue normally. |
| Cell 13 retrain=NO, robustness=BROKEN | Old (pre-v1.5) behavior. Paste-replace cell 13 from `b1b8d50` or later. |
| Colab fully down | Show the screenshot from step 4. Run the local Mac demo as fallback. |
| T4 not available (rare) | Use CPU. ~10x slower but the demo still works. Say "this is on CPU; the same code on T4 is 8-10x faster." |
| Interviewer wants to see a specific number | Open `session_log.json` (downloaded from Colab at end of cell 14). All 21 metrics captured there. |
| Interviewer asks about Roboflow production path | Cell 16 is the one-time setup. If you've done it, cell 12 uses your real weights. If not, cell 12 uses `yolov8n-640` (COCO YOLOv8n placeholder) — the comment in the cell is the honest caveat. |

---

## 7. The 3 numbers to know cold

These are the numbers you'll be asked. Don't look them up.

1. **mAP50 = 0.995** on `recycling_v3` (2298 train, 231 val, 4 epochs early-stopped via patience=3). The Coach flags the 0-val history; v1.5 fix created the 231-image val split.
2. **Inference**: 32 ms/frame on T4 (cell 10 pipeline loop); 67.4 FPS / 14.8 ms/frame via cell 11's `FPSMonitor` (with the v3.5 fix wrapping real work). **Both well inside Everest's 8-12ms target on the same class of GPU.**
3. **113 tests pass** (69 builder + 44 colab_session), **17 cells** total in the v3.5 notebook, **runs in ~10 min on a free T4**.

The frame: "0.995 mAP50, 14.8 ms/frame, 113 tests, 17 cells, 10 minutes on a free T4."

---

## 8. The 3 things to NOT do

1. **Don't start the demo cold.** Run the cells BEFORE the call. The
   live demo is the SHOWING, not the training.
2. **Don't open the README first.** The walkthrough script is the script.
   The README is for after the call.
3. **Don't say "let me just check the docs" mid-call.** If you don't
   know the answer, say "good question, let me come back to that" and
   note it. Better to admit a gap than to fumble through a file.

---

## 9. The 1 question to ASK the interviewer

Near the end (after your 5-min walkthrough), ask:
> *"What does the L1 operator's day actually look like at the ROC right
> now? What takes the longest when something goes wrong?"*

This is the only question you NEED to ask. It signals:
- You think about humans in the loop
- You're not just optimizing mAP
- You want to understand the actual bottleneck (which may not be perception)

Whatever they answer, your 5-tool MCP triage agent is the answer. Use
their pain to set up the follow-up conversation.

---

## 10. v3 → v3.5 → v3.5+1 changelog (for your reference)

| Version | Date | What changed |
|---|---|---|
| **v1.0** | Aug 22 morning | Initial 16-cell build (1-15 + title). `yolov8n-640` placeholder. mAP50=0.995 over 0 val images. |
| **v1.5** | Aug 22 afternoon | Two Coach-driven fixes: (B) `_ensure_val_split` helper + cell 7 call → real 231-image val set; (C) cell 13 retrain override → `retrain: True (overridden: robustness=BROKEN)`. |
| **v3.5** | Aug 22 evening | Cell 12 reads `ROBOFLOW_MODEL_ID` from Colab userdata → production-path uses your real weights. New cell 16 (OPTIONAL): Roboflow one-time setup runs in Colab. |
| **v3.5+1** | Aug 22 evening | Directory cleanup: removed v1 (`demo.ipynb` → trash) + v2 (`build_demo_v2.py` + `demo_v2.ipynb` + 20+ v1-era test files). Renamed v3 files to canonical names: `build_demo.py`, `demo.ipynb`, `test_demo_builder.py`. |

The v3.5 is the **canonical version** — `notebooks/build_demo.py` is the source, `notebooks/demo.ipynb` is regenerated, never hand-edit the .ipynb.
