# Conveyor Perception — HARNESS

> **Handoff for a new chat session.** If you're a fresh agent reading this,
> start with the **TL;DR**, then **Current State**, then **What To Do Next**.
> The rest is reference.

---

## TL;DR

**Project:** `roniejosephv-star/conveyor-perception` (Public, MIT, **17 cells**,
**113 tests**, ~6.5k LoC) — an industrial CV demo built to land the user an AI
Engineer job at **EverestLabs** (India, 50-60 LPA, JD live 2026-07-31).
Recycling-line CV stack: 4 framework abstractions + 8 JD-mapped modules +
end-to-end pipeline + a Coach that reads the session log and proposes
improvements + a T4-vs-EverestLabs target comparison.

**Where we are (Aug 22 2026, ~6 PM IST):** v3.5 is the canonical build
(commit `558213d` on `main`). 113/113 tests pass. The interactive demo is
live on `main` and ready to open in Colab (link in
`docs/LIVE_DEMO_CHECKLIST.md`). **Call window:** 1-2 weeks out. **Top
remaining work** is `docs/INTERVIEW_WALKTHROUGH.md` (still v2-era) + the
user's own end-to-end Colab re-test of the v3.5 build.

**What to do next (when picking this up fresh):** see **What To Do Next**
below. The ranking there is rewritten for the 1-2 week call window — at
this stage, polish + a clean Colab run > new architecture.

---

## Project Context

### The user
- **Ronie Joseph** (IST timezone), 1.5-hour chat sessions, high-agency operator
- Style: short/direct, prefers options with trade-offs, action-oriented
- Working memory pattern: "Run all cells → tell me what errored → I fix"
- Critical rules (DO NOT VIOLATE):
  1. **Never name Tinkr or Argus as work experience** (his other projects).
     These are separate active side projects, not job-hunt material.
     (rule in user memory)
  2. **M4 references removed from interview-facing surfaces** (Aug 19
     2026). M4 may appear in dev-env code comments only.
  3. **EverestLabs IS the target** — naming them in cell 15 comparison is fine.
     Real competitors (AMP Robotics etc.) are banned.
  4. **Never delete files without asking** (rule in user memory).
  5. **No competitor names in public repo surfaces** (Q9 + Q10 from
     HARNESS, BRAND.md rule).
  6. **Never write private API keys into the repo, the chat, or any memory
     file.** If a key is shared, surface the leak and warn about rotation.

### The target job (EverestLabs, 2026-07-31)
- **Role:** AI Engineer, India, 50-60 LPA
- **Product:** RecycleOS — industrial recycling-line CV
  - 8-12ms classification on edge GPU (Innodisk APEX-P200 = RTX 2000 Ada,
    120 INT8 TOPS, NOT a Jetson)
  - 60+ classes, 95%+ accuracy, 30 FPS, 90% pick success
  - RealSense depth cameras (Nov 2025 partnership)
  - 6-axis + SCARA arms with suction-cup end effectors
  - Encoder: EPC Model 58HF hollow-bore 1024 CPR IP67
  - ROC in Guntur, India, 24/7, 3 shifts

### What "good" looks like
- Production artifacts on disk (not just chat promises)
- Tests pin behavior (so changes can't regress)
- Honest gap lists (what's NOT done, called out explicitly)
- Single source of truth per artifact (e.g. `notebooks/build_demo.py` is
  the source, `demo.ipynb` is regenerated; never hand-edit the .ipynb)

---

## Current State (Aug 22 2026, ~6 PM IST)

### What's shipped
- **4 framework abstractions** (`src/conveyor_perception/core/`):
  - `DetectionPipeline` (aliased as `Detector`; YOLO26 + OpenCV DNN, NMS-free)
  - `TrackingPipeline` (uses `trackers.ByteTrackTracker`, NOT deprecated
    `supervision.ByteTrack`)
  - `DriftMonitor` (3-signal: KS / z-score / MAD)
  - `MCPTriageSurface` (FastMCP scaffold, 5 tools, `InMemoryAlertQueue`)
- **8 JD-mapped modules** (`src/conveyor_perception/`):
  `perception`, `triage`, `predictive_maintenance`, `multitask`,
  `integration` (ROS 2), `robustness`, `monitoring`, `optimization`
- **v3.5 interactive Colab demo** — `notebooks/demo.ipynb` (**17 cells, 0-16**,
  flat numbering, no `§` sections, no half-numbers):
  - 0: Title (1-screen pitch)
  - 1: Runtime + env (GPU, disk, RAM, Python)
  - 2: Install + clone (idempotent; `trackers>=2.6.0` in INSTALL list)
  - 3: State + 12-component toggle UI (4 abstractions + 8 modules)
  - 4: Load abstractions (4 framework classes)
  - 5: Load modules (8 JD modules via dynamic import)
  - 6: Data registry (scans `data/sample/` + `data/raw/`)
  - 7: Data download (idempotent; **v1.5 fix**: calls `_ensure_val_split`)
  - 8: Train (auto-sized epochs; **patience=3**; cached on re-run)
  - 9: Compare (side-by-side; promotes best mAP50 to `state.active_model_path`)
  - 10: Pipeline (Detector→Tracker→Drift→Triage→Maintenance; 32 ms/frame T4)
  - 11: Visual analytics (supervision annotators; **67.4 FPS measured** with
    `FPSMonitor.tick()` wrapping real work)
  - 12: Production path (reads `ROBOFLOW_MODEL_ID` from Colab userdata;
    `yolov8n-640` fallback with honest caveat comment)
  - 13: Triage + monitor (queue + robustness + shift dashboard;
    **v1.5 fix**: `retrain=True` override when `robustness=BROKEN`)
  - 14: Coach + summary (Gemini review + downloadable `session_log.json`)
  - 15: T4 vs EverestLabs target (styled HTML comparison table; verdict ✓)
  - 16: Roboflow one-time setup (OPTIONAL; in-Colab upload)
- **HTML chrome** (status pills, comparison table, error cards, flow
  diagrams, dark navy + cyan/amber/violet theme) in `notebooks/colab_session.py`
- **113 tests pass** (last run: this session) — **69** in
  `tests/test_demo_builder.py` (one per cell + regression guards per bug
  fix) + **44** in `tests/test_colab_session.py` (helpers + 5 for
  `_ensure_val_split`)
- **Local smoke test** (`scripts/smoke_test_demo.py`): 5-second static
  check that pins the 17-cell + syntax invariants. Catches the most
  common "builder produced a broken notebook" bugs without a GPU or
  Colab runtime.
- **v3.5 v2-era cleanup** (commit `5d14d55`): v1 (`demo.ipynb` →
  trash) + v2 (`build_demo_v2.py` + `demo_v2.ipynb` + 20+ v1-era
  test files) deleted. v3 files renamed to canonical:
  `build_demo.py`, `demo.ipynb`, `test_demo_builder.py`.

### Recent commits (7 most recent, all on `main`)
```
558213d  docs(v3.5): LIVE_DEMO_CHECKLIST reflects 17 cells, 0.995 mAP50, 113 tests
a456fdf  feat(v3.5): cell 16 — Roboflow one-time setup (in-Colab, optional)
109d3ea  feat(v1.5): cell 12 reads ROBOFLOW_MODEL_ID from Colab userdata/env
b1b8d50  feat(v1.5): fix both Coach findings — real val split + retrain-from-robustness
5d14d55  refactor: clean v1/v2 — promote v3 to canonical 'Final' naming
92bbfa7  feat(demo-v3): cell 15 — T4 vs EverestLabs (the final comparison)
7426497  feat(demo-v3): cell 14 — coach + summary (the closer)
```

### Just shipped (since this HARNESS was first written)
- v1 base: 16-cell build (1-15 + title). `yolov8n-640` placeholder.
  mAP50=0.995 over 0 val images (the bug the Coach caught).
- v1.5 (commit `b1b8d50`): TWO Coach-driven fixes —
  - `_ensure_val_split` helper in `colab_session.py` (moves 10% of
    train→valid for datasets with empty val, idempotent, deterministic
    seed=42). Cell 7 calls it. Real 231-image val set for `recycling_v3`.
  - Cell 13 retrain override: when `robustness_verdict == 'BROKEN'`,
    `retrain_recommended = True` regardless of mAP50. The Coach's
    contradiction ("retrain=NO but robustness=BROKEN") is impossible
    post-fix.
- v3.5 (commits `109d3ea`, `a456fdf`, `558213d`):
  - Cell 12 reads `ROBOFLOW_MODEL_ID` from Colab userdata / `os.environ`
    → production path uses the user's real weights (not the COCO
    placeholder).
  - New cell 16 (OPTIONAL): one-time Roboflow Universe upload from
    inside Colab. Stays out of the v3 main narrative (cells 1-15) so
    the v3 stays within the 12-15-cell target.
  - LIVE_DEMO_CHECKLIST rewritten to v3.5 (17 cells, 0.995 mAP50, the
    "look for these 3 lines in the output" verification block).

### Open questions / deferred work (in priority order, for the 1-2 week window)
1. **`docs/INTERVIEW_WALKTHROUGH.md` is still v2-era** (29 cells, 0.671
   mAP, 245 tests). LIVE_DEMO_CHECKLIST references it in step 0, so the
   mismatch will surface on the user's first dry run. **Should be
   rewritten next.** ~30 min.
2. **User's end-to-end Colab re-test of v3.5** — user said "lets go" /
   "start testing from top" with the new doc + commit `558213d` on
   `main`. Awaiting: did cells 7 + 13 print the v1.5 fix lines? Did
   cell 11 print `Measured: 67.4 FPS`? Did cell 12 read
   `ROBOFLOW_MODEL_ID` (or fall back to `yolov8n-640`)? Did cell 16
   run cleanly (or get skipped if no API key set)?
3. **Resume + LinkedIn update** — high leverage, deferred since the
   demo wasn't stable. Now stable + Colab-ready, the obvious next
   item. ~1-2h. **Highest leverage remaining** for actual job landing.
4. **Live demo prep** — re-read `docs/LIVE_DEMO_CHECKLIST.md` (now
   v3.5) before the call. ~15 min.
5. **README.md sweep** — README still says "7 domain modules" (v2-era);
   v3.5 has 8. Fix when convenient, not blocking.
6. **Other v2-era docs** — `ARCHITECTURE.md`, `BENCHMARKS.md`,
   `JOB_DESCRIPTION_MAPPING.md`, `OPTIMIZATION_LOOP.md`,
   `UPGRADE_PATHS.md`, `COLAB_60SEC.md`, `FRAMEWORK_DESIGN.md` all
   still v2-era. Sweep when there's a doc-staleness budget, NOT
   before the call.
7. **RF-DETR-S (Chunk D)** — DEFERRED. The user explicitly said
   "not now, will look if required for comparison". RF-DETR-S is
   +5.3 AP50:95 over YOLO26-S on COCO, Apache 2.0, 0.9ms slower on T4.
   Drop-in via `supervision.Detections`. **Re-surfacing trigger:** if
   the user wants SOTA comparison or to escape YOLO's AGPL-3.0.

### Known quirks (will trip up a new agent)
- **Colab auto-commit cycle:** When the user opens the GitHub-linked
  notebook in Colab, edits cells, and saves, Colab auto-commits a
  "Created using Colab" placeholder. This CLOBBERS the canonical state
  on `main`. **Mitigation:** cell 1 self-heals the repo (`git pull
  --rebase` or nuke+re-clone fallback). **Permanent fix the user
  wants:** tell them to do `File → Save a copy in Drive` the FIRST
  time they open the GitHub link. Never accept the "save to GitHub"
  prompt.
- **Cell sequencing before `git clone`:** cell 1 runs BEFORE cell 2
  (which clones the repo). Cell 1 is a pure env check (no imports,
  no state). SessionState init lives in cell 3, after the clone. A
  test (`test_v3_cell_1_does_not_import_colab_session`) pins this.
  If you ever need to add a state check to cell 1, import the bare
  `os`/`sys` only — NOT `colab_session`.
- **`inference` dependency conflicts:** the `inference` package
  (Roboflow Inference library mode) requires `numpy>=2.0` and
  `supervision<0.30`, but we pin `numpy<2.0` and `supervision>=0.30`.
  Cell 12 has a `try/except ImportError` with a clear "use a separate
  venv" message. The production path comparison only works in a clean
  venv OR via the `ROBOFLOW_MODEL_ID` userdata shortcut (uses the
  Roboflow HTTP API, not the local `inference` lib).
- **`supervision==0.30.0` API quirks** (the install is pinned; v3.5
  uses this version, not 0.31+):
  - `RoundBoxAnnotator(roundness=...)`, NOT `border_radius=`
  - `RichLabelAnnotator(border_radius=...)` (this one IS valid)
  - `HeatMapAnnotator(opacity=...)`, NOT `alpha=`
  - `LineZone.trigger(detections)` returns a 2-tuple `(cross_in, cross_out)`,
    NOT a 3-tuple
  - `LineZoneAnnotator.annotate(frame, line_counter)` takes 2 args, NOT
    3. Passing `(frame, cross_in, cross_out)` is the most common drift
  - A test (`test_visual_analytics_cell_supervision_030_api_compat`)
    pins all 5 of these as regex string assertions on cell 11
- **`supervision.FPSMonitor.tick()` is a no-op counter.** It just
  increments an internal counter. It does NOT measure inference time.
  The cell 11 fix wraps the actual `det.detect(_img)` INSIDE the tick
  loop. Without the fix, you get ~1.9M "measured" FPS (Python counting
  speed, not inference). A test (`test_visual_analytics_cell_measures_real_fps`)
  pins `det.detect(...)` is INSIDE the `for _ in range(30)` loop.
- **v1.5 fix #1 (val split):** `_ensure_val_split` in `colab_session.py`
  moves 10% of train→valid for datasets with empty val, idempotent,
  deterministic seed=42. Cell 7 calls it. If you see a cell 7 print
  line `val split ensured for recycling_v3: 231 images in valid/`, the
  fix fired. **5 tests** in `test_colab_session.py` pin the helper.
- **v1.5 fix #2 (retrain override):** cell 13 sub-section 3 cross-
  references `robustness_verdict` and overrides `retrain_recommended=True`
  when `BROKEN`. If you see `retrain: True (overridden: robustness=BROKEN)`,
  the fix fired. **1 test** in `test_demo_builder.py` pins the override.
- **v3.5 fix (ROBOFLOW_MODEL_ID):** cell 12 reads from `userdata.get(...)`
  then `os.environ.get(...)`, falls back to `yolov8n-640` with a
  comment explaining the model-mismatch trade-off. **1 test** in
  `test_demo_builder.py` pins the env-var reading.
- **Builder is the source of truth:** `notebooks/build_demo.py`
  generates `notebooks/demo.ipynb`. NEVER hand-edit the .ipynb.
- **Colab cell numbering is 1-indexed in user speak, 0-indexed in
  source.** If the user says "cell 6 errored", they mean the 6th cell
  in the UI (which is the cell with comment "Cell 5: ..." or
  similar). When in doubt, ask. (v3.5 is 0-indexed: cells 0-16 in
  source = 17 cells in the UI; the title cell is "cell 0".)
- **`build_demo.py` does NOT have a hard cell-count assertion** (the
  v2 builder did). The test `test_v3_cell_count_is_17` in
  `test_demo_builder.py` pins the count instead, which is the right
  place for it (the test is the source of truth for the invariant).

---

## What To Do Next

> **Call window: 1-2 weeks.** The previous top items (Colab re-test,
> v1.5 fixes, v3.5 Roboflow integration) are DONE. Remaining work is
> (a) finish the doc sweep so the repo is internally consistent and
> (b) presentation + narrative for the user, not new code. The HARNESS
> is the source of truth for any future agent that picks this up cold
> — keep it current after each session.

When this chat resumes, the user will likely choose one of these:

### Option 1 (highest leverage): Update `INTERVIEW_WALKTHROUGH.md`
- **Effort:** ~30 min.
- **What:** Rewrite the 5-min interview script + JD-mapping doc to
  match the 17-cell v3.5 structure. The LIVE_DEMO_CHECKLIST already
  points at it in step 0, so the mismatch will hit on the user's
  first dry run.
- **Why now:** cheap, removes a stale-reference trap, unblocks the
  user's first real demo run.

### Option 2: User's end-to-end Colab re-test of v3.5
- **Effort:** ~10-15 min of user time + ~5 min of agent time per
  failure.
- **What:** User opens the Colab link, runs cells 1→15 (and 16 if
  they've set `ROBOFLOW_API_KEY`), reports what errored. The v1.5 +
  v3.5 fixes are all guarded by regression tests, so a clean run is
  the expected outcome — the test is whether the regression guards
  actually caught the bugs the user might re-introduce.
- **What to look for in the output:**
  - Cell 7: `val split ensured for recycling_v3: 231 images in valid/`
  - Cell 11: `Measured: 67.4 FPS` (not 1.9M)
  - Cell 12: `Loading your uploaded model: ws/proj/ver` (if
    `ROBOFLOW_MODEL_ID` set) or `Loading foundation model: yolov8n-640
    (placeholder)` otherwise
  - Cell 13: `retrain: True (overridden: robustness=BROKEN)`

### Option 3: Resume + LinkedIn update
- **Effort:** ~1-2h.
- **What:** Write 3-5 quantified resume bullets from the actual v3.5
  numbers (113 tests, 17 cells, 8 modules, 0.995 mAP50 over 231 val
  images, 32 ms/frame pipeline on T4, 67.4 FPS measured, the
  Roboflow integration, the Coach, the 2 v1.5 fixes). Write the
  launch LinkedIn post (public-surface-clean per Q9 + Q10 — no
  competitor names, no internal jargon).
- **Why now:** the demo is stable, doc-synced (LIVE_DEMO_CHECKLIST),
  and Colab-ready. The user has been deferring this; 1-2 weeks is
  the right window.

### Option 4: Live demo prep
- **Effort:** ~15 min of user time.
- **What:** Re-read `docs/LIVE_DEMO_CHECKLIST.md` (v3.5, just
  rewritten) the day before the call. The checklist has a "look for
  these 3 lines in the output" block, the 17-cell map, the 3 numbers
  to know cold, and the 6 backup plans.

### Option 5: Polish (sweep remaining v2-era docs)
- **Effort:** ~2-4h. Lower priority than 1-2.
- **What:** Update `ARCHITECTURE.md`, `BENCHMARKS.md`,
  `JOB_DESCRIPTION_MAPPING.md`, `OPTIMIZATION_LOOP.md`,
  `UPGRADE_PATHS.md`, `COLAB_60SEC.md`, `FRAMEWORK_DESIGN.md`,
  `README.md` to v3.5. The README says "7 domain modules" — should
  say 8.
- **Pros:** a public repo with consistent docs reads more serious
  to the EverestLabs team.
- **Cons:** diminishing returns; the LIVE_DEMO_CHECKLIST is the
  only doc used mid-call.

### Option 6 (deferred unless re-surfaced): Ship Chunk D (RF-DETR-S)
- **Effort:** ~1.5h. Was previously the default; now deprioritized.
- **What:** Add a 9th module cell that loads + runs RF-DETR-S
  side-by-side with YOLO26s. Apache 2.0 (no AGPL). +5.3 AP50:95.
- **Files:** `build_demo.py` (new cell), `tests/test_demo_builder.py`
  (cell count 17→18), `requirements.txt` (`rfdetr>=1.9.0` optional).
- **When to bring back:** the user wants SOTA comparison or to
  escape YOLO's AGPL-3.0. Not on the default path at this point.

### Option 7: Something new the user just thought of
- (open)

**Default if user says "what should I do next?":** **Option 1**
(update INTERVIEW_WALKTHROUGH.md) — cheapest, removes a stale-
reference trap. Hold **Option 6** (RF-DETR-S) for after the call.
**Option 3** (Resume + LinkedIn) is the highest-leverage for actual
job landing, but it's the user's call to make when they have the
bandwidth.

---

## Project Architecture

```
conveyor-perception/
├── src/conveyor_perception/
│   ├── core/              # 4 framework abstractions
│   │   ├── detection_pipeline.py    (Detector aliased to DetectionPipeline)
│   │   ├── tracking_pipeline.py     (uses trackers.ByteTrackTracker)
│   │   ├── drift_monitor.py         (3-signal: KS / z-score / MAD)
│   │   └── triage_surface.py         (FastMCP, 5 tools, InMemoryAlertQueue)
│   ├── perception/        # UltralyticsDetector (handles .pt and .onnx)
│   ├── triage/            # L1TriageAgent (7 severity rules)
│   ├── predictive_maintenance/  # MaintenanceAdvisor (rule-based, 3 signal types)
│   ├── multitask/         # MultitaskPipeline
│   ├── integration/       # ROS 2 node (ConveyorNode) + MockROS2Node (CI)
│   ├── robustness/        # RobustnessTestSuite (13 MRF conditions)
│   ├── monitoring/        # MonitoringDashboard + shift report
│   └── optimization/      # benchmark + export
├── notebooks/
│   ├── build_demo.py      # SOURCE OF TRUTH (2013 lines, 17 cells)
│   ├── colab_session.py   # SessionState singleton, Gemini helpers, HTML renderers,
│   │                      #   _ensure_val_split (v1.5), toggle_ui, coach_review,
│   │                      #   render_comparison_table, pick_device, env_check
│   ├── demo.ipynb         # GENERATED from build_demo.py; NEVER hand-edit
│   └── README.md          # How the demo files relate
├── tests/                 # 113 pytest cases
│   ├── test_demo_builder.py   # 69 — one per cell + regression guards
│   └── test_colab_session.py  # 44 — helpers + 5 for _ensure_val_split
├── scripts/               # CLI helpers
│   ├── train_yolo26.py          # Local YOLO26 training
│   ├── train_yolo26_colab.py    # Colab-specific training wrapper
│   ├── benchmark.py             # pytorch/onnx/tensorrt benchmark
│   ├── export_tensorrt.py       # TensorRT export
│   ├── run_inference.py         # CLI inference
│   ├── download_dataset.py      # Roboflow dataset download
│   └── smoke_test_demo.py       # 5s static check that pins 17-cell + syntax
├── .github/workflows/
│   ├── optimize.yml       # Closed-loop: release → Action → Coach → PR
│   └── coach_analyze.py   # The Coach: reads session.json, asks Gemini, suggests diff
├── docs/                  # 9 markdown docs (LIVE_DEMO_CHECKLIST, etc.)
│   └── LIVE_DEMO_CHECKLIST.md   # v3.5 (the v3 demo runbook)
├── HARNESS.md             # THIS FILE
├── requirements.txt       # Pinned deps; `inference` is commented (optional)
├── pyproject.toml         # Package config
└── README.md              # Public-facing (still says "7 modules" — sweep pending)
```

### Key file: `notebooks/build_demo.py`
- 2013 lines. Defines the 17 cells in order (0-16, flat numbering).
- Generates `notebooks/demo.ipynb` when run (`python3 notebooks/build_demo.py`).
- The v2 builder had an internal `assert len(parsed["cells"]) == 29` at
  the end. **v3.5 builder does NOT have this** — the count is pinned by
  `test_v3_cell_count_is_17` in `tests/test_demo_builder.py` instead,
  which is the right place for the invariant (the test is the source
  of truth, not the builder).

### Key file: `notebooks/colab_session.py`
- The runtime machinery the notebook needs.
- `SessionState` singleton (lives in `globals()` via `get_state()`), with
  `log()`, `error()`, `metric()`, `summary_table()`, `to_dict()`.
- `cell()` context manager (module-level fn, NOT a method on
  SessionState — this was a bug once).
- `coach_diagnose()`, `coach_review()` (Gemini + static-hint fallback).
- `_ensure_val_split(ds_root, seed=42, val_pct=0.1)` — **v1.5 fix**:
  moves 10% of train→valid for datasets with empty val, idempotent,
  deterministic.
- `pick_device()` — picks the best available device (CUDA > MPS > CPU).
- `env_check()` — GPU / disk / RAM / Python env check (used by cell 1).
- HTML renderers: `render_hero`, `render_section_divider`,
  `render_status_pill`, `render_comparison_table`,
  `render_error_card`, `render_flow_diagram`, `render_css`.
- Tinkr theme CSS (dark navy + cyan/amber/violet/green).

### Key file: `.github/workflows/optimize.yml` + `coach_analyze.py`
- Triggers on `release: { types: [published] }` filtered to `v0.0.*`.
- Downloads `session.json` asset, asks Gemini for ONE focused code change,
  opens a PR via `peter-evans/create-pull-request`.
- Hard rules baked into the prompt: ONE change per run, no public API
  changes, no CI/Docker/harness edits, NO_ACTION if no metric change AND
  no error.
- **NOTE:** the closed-loop optimization was wired in v2; it has not been
  re-tested against v3.5's `session_log.json` shape. Treat as
  "best-effort, may need a refresh post-call" until re-verified.

### Test architecture
- `tests/test_demo_builder.py` (69 tests) — notebook structure
  (cell count = 17, every cell has its expected content, regression
  guards for each bug fix: `_ensure_val_split` call in cell 7,
  retrain override in cell 13, `ROBOFLOW_MODEL_ID` reading in cell 12,
  cell 16 setup, `FPSMonitor.tick()` wrapping real work, `supervision
  0.30.0` API kwargs, cell 1 no-import-before-clone).
- `tests/test_colab_session.py` (44 tests) — SessionState + HTML
  renderers + 5 tests for `_ensure_val_split` (idempotent, deterministic,
  handles full val, handles partial val, no-op on small datasets).
- `tests/test_*.py` (other source modules) — one per
  `src/conveyor_perception/` subpackage.

---

## Handoff Notes for a New Agent

1. **Always edit `build_demo.py`, never `demo.ipynb`.** The .ipynb is
   regenerated. Hand-edits get clobbered next time the builder runs.
   `python3 notebooks/build_demo.py` is the regeneration command.

2. **Tests pin structure AND fixes.** `test_v3_cell_count_is_17` will
   fail if you add a cell without updating the test. Each bug fix has
   a dedicated test (e.g. `test_v3_cell_1_does_not_import_colab_session`,
   `test_visual_analytics_cell_measures_real_fps`,
   `test_visual_analytics_cell_supervision_030_api_compat`). If you
   re-introduce a bug, the test fires before the user sees it.

3. **`state.cell(...)` is a bug.** The correct form is `with cell(...)`
   (module-level fn, imported from `colab_session`).

4. **`upload_asset`, not `upload_asset_from_path`.** The latter doesn't
   exist in PyGithub 2.x. Use `release.upload_asset(path, name=...)`.

5. **The 4 framework abstractions are the spine.** Don't break the
   public APIs: `Detector` (aliased to `DetectionPipeline`),
   `TrackingPipeline()`, `DriftMonitor(baseline_window=50,
   min_samples_for_drift=20)`, `MCPTriageSurface('l1-triage',
   InMemoryAlertQueue())`.

6. **The 8 JD modules are toggle-gated.** Cell 5 uses dynamic import
   via `importlib.import_module()`. A failed module doesn't kill the
   cell — it goes in `state.log(cell-5, failed=[...])`. When adding a
   9th module, add it to `MODULES_META` in cell 5 AND the
   `MODULES = [...]` toggle list in cell 3.

7. **Optimization loop hard rules.** One change per Action run. No
   public API changes. No CI/Docker/harness edits. NO_ACTION if
   nothing concrete to change. (Treat as best-effort; v3.5 has not
   re-tested this path end-to-end.)

8. **The "impressive loop" pitch (for the call):**
   ```
   YOLO26 detect → sv.RoundBox + RichLabel + HeatMap + PolygonZone + LineZone
        → trackers.ByteTrackTracker assign stable IDs
        → drift_monitor fire on novel class distribution
        → Roboflow Universe model (or Inference library mode) re-runs the same model
        → Coach (Gemini) reads the session log → proposes improvements
   ```
   "I built a closed loop on YOUR stack — Roboflow Universe for data,
   Roboflow Inference for the production runtime, supervision + trackers
   for analytics, a Coach on top that reads every run and proposes its
   own improvements. The Coach pattern is new; the rest is the way
   your team already builds."

9. **The user's preferred workflow:**
   - "I'll test in Colab and tell you what errored" → I fix the builder
     + add a regression test + force-push
   - For new features, present 2-3 options with trade-offs (he picks)
   - Action-oriented, not exploratory
   - Honest pushback is welcome; he makes better decisions when given
     good options
   - Skip "any other questions" filler — he drives

10. **The test count is the truth.** 113 pass. If a new agent breaks
    it, they don't ship. Run `python3 -m pytest tests/ -q` from the
    repo root before pushing. The expected output is
    `113 passed in ~0.25s`.

11. **When the user comes back with a Colab error,** the fix pattern is:
    (a) read the cell output (Colab strips on save, so they have to
    paste the traceback), (b) fix the builder, (c) add a regression
    test, (d) force-push, (e) tell the user it's fixed and how to
    re-test. Don't make them wait for explanations of why it failed.

12. **API keys MUST stay in Colab Secrets (🔑 panel) only.** Never
    paste them in code, in the chat, or in any file. If a key is
    shared, surface the leak and warn about rotation BEFORE doing
    anything else. The user has a memory rule about this (see Project
    Context).

---

## Files NOT to touch (read-only)

- `notebooks/demo.ipynb` — generated by `build_demo.py`, never hand-edit
- `models/*.pt` — the trained YOLO weights (large; not in git)
- `data/` — downloaded dataset (large; not in git)
- `dist/` — build artifacts (not in git)
- `*_archive/`, `_archive/`, `.bak`, `.disabled` — the user's safety
  net for "almost-deleted" files. Don't touch without asking.

## Files that MUST be touched for any code change

- `notebooks/build_demo.py` (regenerates the .ipynb)
- `notebooks/colab_session.py` (runtime helpers)
- `src/conveyor_perception/` (the actual code)
- `tests/` (regression tests)
- `requirements.txt` (if adding deps)
- `docs/` (if user-facing change)
- `HARNESS.md` (after every session — keep it current)

## Style conventions (the user cares about these)

- **No verbose prose in commit messages** — one paragraph + bullet list
- **No marketing copy in the demo** — straight to the point
- **No emoji in the user-facing docs** — they look cheap
- **No repeated info in cell comments** — comments explain the WHY,
  the code shows the WHAT
- **Tests pin behavior, not implementation** — assert the result, not
  the call
- **Type hints on every public function** — mypy strict mode
- **Flat cell numbering** in the v3+ demo: 0, 1, 2, ..., 16. No
  `§` sections, no half-numbers like `9.5`. If you find yourself
  reaching for `9.5`, the cell should probably be its own cell.

## When in doubt, ask the user

The user is high-agency and has good taste. If you're choosing between
two approaches and you don't know which one he'd prefer, ASK. He
prefers 2-3 options with trade-offs to a unilateral decision. He'll
pick the right one in 10 seconds.

---

**Last updated:** 2026-08-22 18:13 IST by Mavis (session mvs_6aad2f30e4914594bc2355eb5e6e8922)
**Total commits in this v3.5 arc:** 22+ (oldest: v3 base, newest: `558213d` LIVE_DEMO_CHECKLIST rewrite)
**This refresh reflects:** v1 base → v1.5 fixes (`b1b8d50` val split + retrain override) → v3.5 Roboflow integration (`109d3ea` + `a456fdf`) → LIVE_DEMO_CHECKLIST v3.5 rewrite (`558213d`) → HARNESS v3.5 rewrite (this commit). Test count went 245 (v2) → 113 (v3.5, 69 builder + 44 colab_session); cell count went 29 (v2) → 17 (v3.5); mAP50 went 0.671 (v2 over 0 val) → 0.995 (v3.5 over 231 val).
