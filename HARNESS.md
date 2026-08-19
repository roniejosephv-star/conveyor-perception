# Conveyor Perception — HARNESS

> **Handoff for a new chat session.** If you're a fresh agent reading this,
> start with the **TL;DR**, then **Current State**, then **What To Do Next**.
> The rest is reference.

---

## TL;DR

**Project:** `roniejosephv-star/conveyor-perception` (Public, MIT, 29 cells,
245 tests, ~6.5k LoC) — an industrial CV demo built to land the user an AI
Engineer job at **EverestLabs** (India, 50-60 LPA, JD live 2026-07-31).
Recycling-line CV stack: 4 framework abstractions + 8 JD-mapped modules +
end-to-end pipeline + a closed-loop Coach that reads the session log and
proposes its own improvements via GitHub PRs.

**Where we are (Aug 19 2026, ~3:40 PM IST):** Code is feature-complete and
**doc-synced across the public surface** (commit `0a0df0b` — module count,
test stats, demo URL fixed in README, LIVE_DEMO_CHECKLIST, INTERVIEW_WALKTHROUGH,
JOB_DESCRIPTION_MAPPING, FRAMEWORK_DESIGN, ARCHITECTURE, BENCHMARKS). 245 tests
pass + 1 skipped. The interactive demo is live on `main` at
`github.com/roniejosephv-star/conveyor-perception`. **Call window:** 2-3 weeks
out. **Top remaining work** is polish, resume/LinkedIn, and a clean Colab
re-test — not new code. RF-DETR-S (Chunk D) stays deferred unless the user
explicitly re-surfaces it.

**What to do next (when picking this up fresh):** see **What To Do Next**
below. The ranking there is rewritten for the 2-3 week call window — at this
stage, polish > new architecture.

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
  3. **EverestLabs IS the target** — naming them in §3 comparison is fine.
     Real competitors (AMP Robotics etc.) are banned.
  4. **Never delete files without asking** (rule in user memory).
  5. **No competitor names in public repo surfaces** (Q9 + Q10 from
     HARNESS, BRAND.md rule).

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
- Single source of truth per artifact (e.g. `notebooks/build_demo_v2.py` is
  the source, `demo_v2.ipynb` is regenerated; never hand-edit the .ipynb)

---

## Current State (Aug 19 2026, ~3:40 PM IST)

### What's shipped
- **4 framework abstractions** (`src/conveyor_perception/core/`):
  DetectionPipeline (YOLO26 + OpenCV DNN, NMS-free), TrackingPipeline
  (now using `trackers.ByteTrackTracker`, NOT deprecated `supervision.ByteTrack`),
  DriftMonitor (3-signal: KS / z-score / MAD), MCPTriageSurface (5 FastMCP tools)
- **8 JD-mapped modules** in `src/conveyor_perception/`: perception, triage,
  predictive_maintenance, multitask, integration (ROS 2), robustness,
  monitoring, optimization
- **End-to-end pipeline** + **Coach** (Gemini integration with static-hint
  fallback) + **closed-loop optimization** (Colab → GitHub Release →
  Action → Coach → PR)
- **Interactive Colab demo** `notebooks/demo_v2.ipynb` (29 cells, 5 sections):
  §1 Setup, §2 Walkthrough (now includes visual analytics + production path),
  §3 Comparison, §4 Coach, §5 Optimization Loop + interactive 4-tab widget
  dashboard
- **HTML chrome** (hero, section dividers, stat cards, styled comparison
  table, error cards, flow diagrams) in `notebooks/colab_session.py`
- **245 tests pass + 1 skipped** (last run: this session, `.venv`)
- **Doc-sync** (commit `0a0df0b`): module count, test stats, demo URL
  aligned across the public surface (README, LIVE_DEMO_CHECKLIST,
  INTERVIEW_WALKTHROUGH, JOB_DESCRIPTION_MAPPING, FRAMEWORK_DESIGN,
  ARCHITECTURE, BENCHMARKS)

### Recent commits (7 most recent, all on `main`)
```
0a0df0b  docs: sync module count + test stats + demo URL across the public surface
fea4eb4  docs: HARNESS.md — handoff document for a new chat session
d187f72  feat(notebooks): production-path cell — Roboflow Inference (library mode)
ccb6f76  feat(notebooks): visual analytics cell — modern supervision annotators
8781b09  refactor(tracking): migrate supervision.ByteTrack → trackers.ByteTrackTracker
bad8133  feat(notebooks): interactive hybrid demo — HTML chrome + widget dashboard
e08a116  fix(notebooks): 3 runtime errors from user's Colab run
```

### Just shipped (since this HARNESS was first written at fea4eb4)
- `0a0df0b` — the public-surface doc-sync (7 files, 1 commit).
  The new chat that picked this up cold found 3 urgent issues and fixed
  them in one bounded pass: wrong demo URL in LIVE_DEMO_CHECKLIST,
  stale test counts (171 / 209 / 221 / 238 → 245), "7 modules" → "8 modules"
  in the JD-mapped module pitch. This HARNESS update is the wrap-up of
  that pass.

### Open questions / deferred work (in priority order, for the 2-3 week window)
1. **Local smoke test of the 29-cell notebook** — `scripts/smoke_test_demo.py`
   that mocks the heavy stuff (YOLO, Gemini, PyGithub) and confirms all 29
   cells parse + import in <5s. Replaces "open Colab and pray" with a
   5-second local check. Effort: ~45 min. **Highest leverage — cheapest
   risk reduction left.**
2. **Colab re-test of all 29 cells end-to-end** — the user has run cells
   incrementally (multiple manual saves, auto-commits). A clean restart
   + run-all on the latest `main` is the highest-confidence test that
   the demo still works for the call. Effort: ~30 min of user time.
3. **Resume + LinkedIn update** — user has deferred since the demo
   wasn't stable. Now stable + doc-synced, this is the obvious next item.
   Effort: ~1-2h.
4. **Live demo prep** — `docs/LIVE_DEMO_CHECKLIST.md` was rewritten in
   `0a0df0b` to point at `demo_v2.ipynb` (the right notebook) and the
   5-section / 29-cell structure. User should re-read it before the call.
5. **Optimization loop first PR** — Action is wired (commits 0ad232b +
   ba18a24), `GEMINI_API_KEY` is set in repo secrets, publish cell fixed
   in f53fd7c (`upload_asset` not `upload_asset_from_path`). The first
   user-initiated publish + Action run will produce the first PR from
   the Coach. **No work to do** — wait for the user to publish.
6. **Chunk D: RF-DETR-S as 9th module** — DEFERRED. The user explicitly
   said "not now, will look if required for comparison". RF-DETR-S is
   +5.3 AP50:95 over YOLO26-S on COCO, Apache 2.0, 0.9ms slower on T4.
   Drop-in via `supervision.Detections`. **Re-surfacing trigger:** if
   the user wants SOTA comparison or to escape YOLO's AGPL-3.0.

### Known quirks (will trip up a new agent)
- **Colab auto-commit cycle:** When the user opens the GitHub-linked
  notebook in Colab, edits cells, and saves, Colab auto-commits a
  "Created using Colab" placeholder. This CLOBBERS the canonical state
  on `main`. We've been doing 5+ force-pushes per session. **Mitigation
  in place:** cell 1 self-heals the repo, cell 9.7 (production path)
  self-heals `inference`. **Permanent fix the user wants:** tell them
  to do `File → Save a copy in Drive` the FIRST time they open the
  GitHub link. Never accept the "save to GitHub" prompt.
- **`inference` dependency conflicts:** the `inference` package (for
  Roboflow Inference library mode) requires `numpy>=2.0` and
  `supervision<0.30`, but we pin `numpy<2.0` and `supervision>=0.30`.
  Cell 9.7 has a `try/except ImportError` with a clear "use a separate
  venv" message. The production path comparison only works in a
  clean venv.
- **`from inference.models.utils import get_model`** is intentionally
  wrapped in `try/except` so the cell degrades gracefully.
- **Builder is the source of truth:** `notebooks/build_demo_v2.py`
  generates `notebooks/demo_v2.ipynb`. NEVER hand-edit the .ipynb.
  The tests pin the cell count (29) and key features.
- **Colab cell numbering is 1-indexed in user speak, 0-indexed in
  source.** If the user says "cell 6 errored", they mean the 6th
  cell in the UI (which is the cell with comment "Cell 5: ..." or
  similar). When in doubt, ask.
- **`build_demo_v2.py` has an internal assertion** at the end
  (`assert len(parsed["cells"]) == 29`). Update this when adding cells.

---

## What To Do Next

> **Call window: 2-3 weeks.** This re-ranks the original options. At this
> stage, polish + narrative + a clean Colab run > new architecture.
> The HARNESS is the source of truth for any future agent that picks
> this up cold — keep it current after each session.

When this chat resumes, the user will likely choose one of these:

### Option 1 (highest leverage): Local smoke test of the 29 cells
- **Effort:** ~45 min. One new file: `scripts/smoke_test_demo.py`.
- **What:** Mock the heavy parts (YOLO, Gemini, PyGithub, the GitHub
  REST API calls in §5), then `python -m jupyter nbconvert --execute
  --to notebook --inplace` against a copy of `demo_v2.ipynb`, asserting
  the cell count is 29 and no cell raises. Should run in <5s.
- **Why now:** the user has done 5+ force-push recoveries from Colab
  auto-commits. A local check that says "all 29 cells parse + import
  cleanly" is the highest-confidence pre-call sanity test that's
  cheaper than opening Colab.
- **Files:** `scripts/smoke_test_demo.py`, `tests/test_smoke_test_demo.py`,
  maybe `pyproject.toml` to register the script.

### Option 2: Colab re-test of all 29 cells on T4
- **Effort:** ~30 min of user time.
- **What:** Open the GitHub-linked `demo_v2.ipynb` in Colab (use
  `File → Save a copy in Drive` on first open to detach from
  GitHub auto-commit). Restart runtime → Run all. Paste any errors
  back to the agent.
- **Why this matters:** the smoke test in Option 1 is a substitute,
  not a replacement. The user must still see the demo run end-to-end
  on the real T4 runtime at least once before the call.

### Option 3: Resume + LinkedIn update
- **Effort:** ~1-2h.
- **What:** Write 3-5 quantified resume bullets from the actual
  numbers (245 tests, 29 cells, 8 modules, the 3 Roboflow chunks,
  the closed-loop Coach). Write the launch LinkedIn post
  (public-surface-clean per Q9 + Q10 — no competitor names,
  no internal jargon).
- **Why now:** the demo has been stable long enough to write
  about with confidence. The user has been deferring this;
  2-3 weeks is the right window.

### Option 4: Polish (CI tightening, golden-file tests, CONTRIBUTING)
- **Effort:** ~2-4h. Lower priority than 1-3.
- **What:** Add `notebooks/build_demo_v2.py` golden-file snapshot
  tests; add a `CONTRIBUTING.md`; tighten the action workflow;
  add release notes for the v0.0.0 optimization-loop work.
- **Pros:** a public repo with tests + docs reads more serious
  to the EverestLabs team.
- **Cons:** diminishing returns; the demo already works.

### Option 5 (deferred unless re-surfaced): Ship Chunk D (RF-DETR-S)
- **Effort:** ~1.5h. Was previously the default; now deprioritized.
- **What:** Add a 9th module cell that loads + runs RF-DETR-S
  side-by-side with YOLO26s. Apache 2.0 (no AGPL). +5.3 AP50:95.
- **Files:** `build_demo_v2.py` (new cell), `tests/test_demo_v2_builder.py`
  (cell count 29→30), `requirements.txt` (rfdetr>=1.9.0 optional).
- **When to bring back:** the user wants SOTA comparison or to
  escape YOLO's AGPL-3.0. Not on the default path at this point.

### Option 6: Something new the user just thought of
- (open)

**Default if user says "what should I do next?":** run **Option 1** (the
smoke test) first — cheapest risk reduction left. Then **Option 2** (the
Colab re-test) — highest confidence. Hold **Option 5** (RF-DETR-S) for
after the call.

---

## Project Architecture

```
conveyor-perception/
├── src/conveyor_perception/
│   ├── core/              # 4 framework abstractions
│   │   ├── detection_pipeline.py    (Detector aliased to DetectionPipeline)
│   │   ├── tracking_pipeline.py     (uses trackers.ByteTrackTracker)
│   │   ├── drift_monitor.py         (3-signal: KS / z-score / MAD)
│   │   └── triage_surface.py         (FastMCP, 5 tools)
│   ├── perception/        # UltralyticsDetector (handles .pt and .onnx)
│   ├── triage/            # L1TriageAgent (7 severity rules)
│   ├── predictive_maintenance/  # MaintenanceAdvisor (rule-based)
│   ├── multitask/         # MultitaskPipeline
│   ├── integration/       # ROS 2 node (ConveyorNode)
│   ├── robustness/        # RobustnessTestSuite (13 MRF conditions)
│   ├── monitoring/        # MonitoringDashboard + shift report
│   └── optimization/      # benchmark + export
├── notebooks/
│   ├── build_demo_v2.py   # SOURCE OF TRUTH for the .ipynb (regenerates it)
│   ├── colab_session.py   # SessionState singleton, Gemini helpers, HTML renderers
│   ├── demo_v2.ipynb      # GENERATED; never hand-edit
│   ├── demo.ipynb         # LEGACY 9-cell demo (deprecated, but kept for history)
│   └── README.md          # How the demo files relate
├── tests/                 # 245 pytest cases
├── scripts/               # CLI helpers (train, benchmark, export, download dataset)
├── .github/workflows/
│   ├── optimize.yml       # Closed-loop: release → Action → Coach → PR
│   └── coach_analyze.py   # The Coach: reads session.json, asks Gemini, suggests diff
├── docs/                  # 10 markdown docs (ARCHITECTURE, BENCHMARKS, etc.)
├── HARNESS.md             # THIS FILE
├── requirements.txt       # Pinned deps; `inference` is commented (optional)
├── pyproject.toml         # Package config
└── README.md              # Public-facing
```

### Key file: `notebooks/build_demo_v2.py`
- ~1300 lines. Defines the 29 cells in order.
- Internal assertion: `assert len(parsed["cells"]) == 29` at the end.
- When you add a cell, bump that number and the corresponding test
  (`test_cell_count_is_29` in `tests/test_demo_v2_builder.py`).

### Key file: `notebooks/colab_session.py`
- The runtime machinery the notebook needs.
- `SessionState` singleton (lives in `globals()`), with `log()`,
  `error()`, `metric()`, `summary_table()`, `to_dict()`.
- `cell()` context manager (module-level fn, NOT a method on
  SessionState — this was a bug once).
- `coach_diagnose()`, `coach_review()` (Gemini + static-hint fallback).
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

### Test architecture
- `tests/test_tracking_pipeline.py` — IoU fallback + new trackers test
- `tests/test_detection_pipeline.py` — DetectionPipeline
- `tests/test_*.py` — one per source module
- `tests/test_colab_session.py` — SessionState + HTML renderers
- `tests/test_demo_v2_builder.py` — notebook structure (cell count,
  imports, self-heal guards, no banned tokens like "M4" or "Tinkr")
- `tests/test_coach_analyze.py` — Coach prompts + JSON parsing
- `tests/test_train_yolo26_resume.py` — `--resume` flag

---

## Handoff Notes for a New Agent

1. **Always edit `build_demo_v2.py`, never `demo_v2.ipynb`.** The .ipynb
   is regenerated. Hand-edits get clobbered next time the builder runs.

2. **Tests pin structure.** `test_cell_count_is_29` will fail if you add
   a cell without bumping. `test_no_banned_pii_in_cells` will fail if
   you write "Tinkr" or "M4" in the notebook.

3. **`state.cell(...)` is a bug.** The correct form is `with cell(...)`
   (module-level fn, imported from `colab_session`).

4. **`upload_asset`, not `upload_asset_from_path`.** The latter doesn't
   exist in PyGithub 2.x. Use `release.upload_asset(path, name=...)`.

5. **The 4 framework abstractions are the spine.** Don't break the
   public APIs: `Detector` (aliased to `DetectionPipeline`),
   `TrackingPipeline()`, `DriftMonitor(baseline_window=50,
   min_samples_for_drift=20)`, `MCPTriageSurface('l1-triage',
   InMemoryAlertQueue())`.

6. **Optimization loop hard rules.** One change per Action run. No
   public API changes. No CI/Docker/harness edits. NO_ACTION if
   nothing concrete to change.

7. **The "impressive loop" pitch (for the call):**
   ```
   YOLO26 detect → sv.RoundBox + RichLabel + HeatMap + PolygonZone + LineZone
        → trackers.ByteTrackTracker assign stable IDs
        → drift_monitor fire on novel class distribution
        → Roboflow Inference library mode re-runs the same model
        → Coach (Gemini) reads the session log → proposes a PR
   ```
   "I built a closed loop on YOUR stack — Roboflow Universe for data,
   Roboflow Inference for the production runtime, supervision + trackers
   for analytics, a Coach on top that reads every run and proposes its
   own improvements. The Coach pattern is new; the rest is the way
   your team already builds."

8. **The user's preferred workflow:**
   - "I'll test in Colab and tell you what errored" → I fix the builder
     + add a regression test + force-push
   - For new features, present 2-3 options with trade-offs (he picks)
   - Action-oriented, not exploratory
   - Honest pushback is welcome; he makes better decisions when given
     good options
   - Skip "any other questions" filler — he drives

9. **The test count is the truth.** 245 pass + 1 skipped. If a new
   agent breaks it, they don't ship. Run `source .venv/bin/activate &&
   python -m pytest tests/ -q` from the repo root before pushing.

10. **When the user comes back with a Colab error,** the fix pattern is:
    (a) read the cell output (Colab strips on save, so they have to
    paste the traceback), (b) fix the builder, (c) add a regression
    test, (d) force-push, (e) tell the user it's fixed and how to
    re-test. Don't make them wait for explanations of why it failed.

---

## Files NOT to touch (read-only)

- `notebooks/demo.ipynb` — legacy 9-cell demo, kept for history
- `models/*.pt` — the trained YOLO weights (large; not in git)
- `data/` — downloaded dataset (large; not in git)
- `dist/` — build artifacts (not in git)

## Files that MUST be touched for any code change

- `notebooks/build_demo_v2.py` (regenerates the .ipynb)
- `notebooks/colab_session.py` (runtime helpers)
- `src/conveyor_perception/` (the actual code)
- `tests/` (regression tests)
- `requirements.txt` (if adding deps)
- `docs/` (if user-facing change)

## Style conventions (the user cares about these)

- **No verbose prose in commit messages** — one paragraph + bullet list
- **No marketing copy in the demo** — straight to the point
- **No emoji in the user-facing docs** — they look cheap
- **No repeated info in cell comments** — comments explain the WHY,
  the code shows the WHAT
- **Tests pin behavior, not implementation** — assert the result, not
  the call
- **Type hints on every public function** — mypy strict mode

## When in doubt, ask the user

The user is high-agency and has good taste. If you're choosing between
two approaches and you don't know which one he'd prefer, ASK. He
prefers 2-3 options with trade-offs to a unilateral decision. He'll
pick the right one in 10 seconds.

---

**Last updated:** 2026-08-19 15:40 IST by Mavis (session mvs_2303365e3b3843109d69d82257986d42)
**Total commits in this arc:** 21 (oldest: `e3303a1` colab_session.py, newest: `0a0df0b` doc-sync)
**Doc-sync that motivated this refresh:** `0a0df0b` — 7 files, 1 commit, +73/-57. Test count went 238→245; module count went 7→8 across the public surface.
