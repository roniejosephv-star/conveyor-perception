# Job Description → Showcase Mapping (the index file)

**Purpose:** This repo is the proof. Each JD responsibility has a module that demonstrates it. **This doc is the index.**

**Date:** 2026-08-19
**Status:** Active — updated as modules ship.
**End state:** Option 1 + Option 3 combined — the **4 framework abstractions** + the **7 JD modules** + 1 end-to-end example. See `FRAMEWORK_DESIGN.md` for the layered architecture.

---

## The layered architecture (summary)

```
Layer 1: Core abstractions (4) → FRAMEWORK_DESIGN.md
Layer 2: Domain modules (7 + bonus) → this file
Layer 3: End-to-end example (1) → examples/conveyor_demo.py
```

Each module in Layer 2 uses one or more abstractions from Layer 1. The example in Layer 3 wires everything together.

---

## The 7 JD bullets → 7 modules + 1 bonus

| # | JD responsibility (verbatim) | Module | JD bullet summary | Time |
|---|---|---|---|---|
| 1 | "Design and optimize deep learning models for real-time object detection and classification on streaming visual data" | `perception/` (detector + tracker + infer) | Real-time object detection on streaming video | 2.5 h |
| 2 | "Build perception models beyond sorting (throughput analytics, equipment monitoring, predictive maintenance)" | `predictive_maintenance/` | Perception for the *next* 10x (the differentiator) | 1.5 h |
| 3 | "Work across object detection, classification, anomaly detection, and time-series analysis on visual data" | `classifier.py` + `anomaly.py` + `time_series.py` | All 4 model types in the JD | 1.5 h |
| 4 | "Integrate models into RecycleOS and the robots" | `integration/` (robot_pick_simulator + recycleos_bridge) | Vision ↔ robot ↔ cloud | 1 h |
| 5 | "Make perception robust to chaotic industrial environments" | `robustness/` (augmentations + domain_shift_tests) | Dust, vibration, lighting, occlusion | 1 h |
| 6 | "Monitor in production, catch drift, design retraining loops" | `monitoring/` (drift + health + retraining) | Production observability | 1 h |
| 7 | "Squeeze every millisecond out of edge inference" | `optimization/` (tensorrt + profile) | TensorRT FP16/INT8, per-step latency | 1 h |
| **Bonus** | (not in JD explicitly) — L1 ROC triage agent | `triage/` (FastMCP server) | The tool that lets the ROC scale to 100+ sites | 2 h |

**Total:** ~11.5 h over 3 days (~4 h/day). Each module ships as a runnable artifact + a 60-second interview answer.

---

## The AI-augmented workflow (using Mavis as a force multiplier)

**Rule:** Use Mavis for the 70% work (research, review, docs). Keep the 30% for yourself (architecture, debugging, integration).

| When to use Mavis (me) | When to do yourself |
|---|---|
| **Research** (YOLOv8 quirks, MCP patterns, drift detection methods, TensorRT APIs) | **Architecture decisions** (which modules, which libraries, which trade-offs) |
| **Code review** after you write a module (catch missed edge cases, suggest tests) | **Implementation** (the actual code — this is what makes it *yours*) |
| **Test case suggestions** (the corner cases you didn't think of) | **Debugging** (real bugs need your eyes + intuition) |
| **Doc generation** (README, BENCHMARKS, JD mapping, ARCHITECTURE) | **Integration glue** (wiring modules together — the *glue* is the showcase) |
| **Interview Q&A** based on the work (draft the 60-sec answers) | **Final polish** (the last 10% — the magic that makes it a portfolio piece) |
| **Catching the "you said X but did Y" gap** (consistency between docs and code) | **Performance choices** (which optimization, which trade-off) |

**The principle:** *The architecture and integration are yours.* That's what makes the showcase a portfolio piece, not a tutorial clone. **The AI helps you ship faster, not think for you.**

**The cadence for each module:**
1. **You** design + implement (60% of module time)
2. **Mavis** reviews the code + suggests tests (10%)
3. **Mavis** drafts the module's README + the interview Q&A (15%)
4. **You** polish + commit (15%)

That split keeps the work *yours* but cuts the boring parts in half.

---

## The 3-day cadence (each day = 3-4 hours)

### Day 1 — Perception stack + integration (~4 h)
- **09:00-09:30** Step 1: Foundation + mental model — repo skeleton, env, README diagram
- **09:30-10:30** Step 2: Data + domain — Roboflow download, exploration notebook
- **10:30-11:30** Step 3: YOLO26s training + ONNX export → JD bullet 1 (detection)
- **11:30-12:30** Step 4: OpenCV DNN inference loop → JD bullet 1 (real-time)
- **12:30-13:00** Step 5: ByteTrack tracking → JD bullet 1 (multi-object)
- **13:00-14:30** Step 6: End-to-end `conveyor.py` + robot pick simulator → JD bullets 1 + 4

### Day 2 — Beyond sorting + multi-task (~4 h)
- **09:00-10:30** Step 7: Predictive maintenance perception → JD bullet 2 (the differentiator)
- **10:30-11:00** Step 8a: `classifier.py` (image classification) → JD bullet 3
- **11:00-11:30** Step 8b: `anomaly.py` (statistical anomaly detection) → JD bullet 3
- **11:30-12:00** Step 8c: `time_series.py` (latency/throughput over time) → JD bullet 3
- **12:00-13:00** Step 9: MCP triage surface (5 tools + CLI driver) → bonus
- **13:00-14:00** Step 10: Robustness (augmentations + domain shift tests) → JD bullet 5

### Day 3 — Production-grade + push (~3.5 h)
- **09:00-10:00** Step 11: Drift detection + health check + retraining loop → JD bullet 6
- **10:00-11:00** Step 12: TensorRT benchmark + profiling → JD bullet 7
- **11:00-12:00** Step 13: Docs — README, ARCHITECTURE, BENCHMARKS, this file
- **12:00-12:30** Step 14: Wire end-to-end demo, run full pipeline
- **12:30-13:00** Step 15: Push to public repo, share link with recruiter

---

## Module specs (what each proves + the 60-sec interview answer)

### 1. `perception/` — Real-time object detection (JD bullet 1)
**Artifact:** `infer.py` runs at 30+ FPS on CPU, 60+ FPS on Jetson with TensorRT. `best.onnx` + `best.pt` in `models/`.
**Q:** "Walk me through your inference loop."
**A (60 sec):** "Three steps. (1) Preprocess with `cv2.dnn.blobFromImage` — letterbox to 640x640, scale to 0-1, swap BGR→RGB. (2) Forward pass returns `[1, 4+C, 8400]`. (3) Postprocess: parse the tensor, filter by confidence, apply NMS, scale boxes back accounting for letterbox padding. The two bugs that waste 80% of debugging time: forgetting the letterbox offset and the BGR swap. I write unit tests for both."

### 2. `predictive_maintenance/` — Beyond sorting (JD bullet 2 — the differentiator)
**Artifact:** `pm_model.py` watches pick patterns + encoder signal + air pressure, predicts failures 4h ahead.
**Q:** "How would you build predictive maintenance for the robot cells?"
**A (60 sec):** "Three signal sources. (1) Pick attempt patterns — suction cup slipping = late picks. (2) Encoder signal quality — drift = missed counts. (3) Air pressure trend — vacuum loss = pick failures. A simple time-series anomaly detector on these signals catches 80% of failures. The model is small because it runs on the same APEX-P200 as the perception stack. This is the next 10x — once the robots are deployed, the value is in *uptime*, not picking more."

### 3. `classifier.py` + `anomaly.py` + `time_series.py` — Multi-task (JD bullet 3)
**Artifacts:** 3 small models, one per type.
- `classifier.py` — image-level (e.g., "is this PET clean or contaminated?")
- `anomaly.py` — per-class count anomaly (KS test + z-score)
- `time_series.py` — latency/throughput over time + drift detection
**Q:** "How would you set up anomaly detection for a conveyor?"
**A (60 sec):** "Two layers. Layer 1: per-class count anomaly — if PET count drops 50% in 5 minutes, upstream contamination. Layer 2: per-class confidence distribution — if model confidence on PET drops 0.85→0.70, distribution shift. Both are simple, both run on the APEX-P200, both feed into the same triage queue. The 80/20 is layer 1 — count anomalies catch 80% of real-world issues."

### 4. `integration/` — RecycleOS + robots (JD bullet 4)
**Artifact:** `robot_pick_simulator.py` + `recycleos_bridge.py`.
**Q:** "How does your model integrate with the robot?"
**A (60 sec):** "Three messages. (1) Vision-to-robot: 'object at (x,y,z), class PET, confidence 0.92, ID 42, will arrive at pick point in 80ms.' (2) Robot-to-vision: 'pick attempted, success/fail.' (3) Vision-to-cloud: 'class count, pick rate, drift indicators.' The pick simulator validates the round trip. In production, the simulator becomes a 6-axis arm trajectory planner with inverse kinematics."

### 5. `robustness/` — Chaotic industrial environments (JD bullet 5)
**Artifact:** `augmentations.py` (motion blur, occlusion, lighting, dust) + `domain_shift_tests.py` (mAP drop report).
**Q:** "How do you handle dust, vibration, variable lighting?"
**A (60 sec):** "Three layers. (1) Augmentations in training: motion blur, occlusion, brightness/contrast jitter, Gaussian noise. (2) Self-lit camera enclosures (EverestLabs' design choice) reduce ambient variation. (3) Test on a held-out 'chaotic' split that simulates real MRF conditions — if mAP drops >5%, the augmentation set is missing something. **Domain shift is a data problem, not a model problem.** The real fix is collecting more diverse training data via the ROC's misclassification log."

### 6. `monitoring/` — Drift + retraining (JD bullet 6)
**Artifact:** `drift_detector.py` (KS test on per-class confidence) + `health_check.py` (latency, throughput) + `retraining_loop.py` (trigger + data pipeline).
**Q:** "How do you catch model drift in production?"
**A (60 sec):** "Three signals. (1) **Confidence distribution drift** — KS test on per-class confidence, p<0.05 = drift. (2) **Class count anomaly** — if PET counts drop 30% in a week, input distribution changed. (3) **ROC feedback loop** — when the ROC marks a misclass, that image is flagged for retraining. The retraining loop is weekly, triggered by any of the three. **The retraining data comes from production**, not from the original training set. That's the active learning piece."

### 7. `optimization/` — Edge inference (JD bullet 7)
**Artifact:** `tensorrt_benchmark.py` (FP32 vs FP16 vs INT8) + `profile.py` (per-step latency).
**Q:** "How do you optimize for edge?"
**A (60 sec):** "Three levels. (1) **Architecture**: YOLO26s not nano/medium, 640x640 not 320x320. (2) **Quantization**: FP16 is the default (2x speedup, sub-1% mAP loss). INT8 needs calibration (3-4x speedup, 2-3% mAP loss on corner classes). (3) **Runtime**: TensorRT on Jetson, not PyTorch — same model, 4x faster. The wins stack: architecture 1.5x + quantization 2x + runtime 1.3x ≈ 4x total. The benchmark numbers are in `BENCHMARKS.md`."

### Bonus: `triage/` — L1 ROC agent (the differentiator — not in JD but the highest-leverage pitch)
**Artifact:** FastMCP server with 5 tools + CLI driver.
**Q:** "How would you help the ROC scale to 100+ sites?"
**A (60 sec):** "An L1 triage agent. It pulls recent perception events via MCP, classifies them routine/attention/escalate, and either auto-resolves routine events or pages a human. **The MCP surface is the key** — declarative tool contracts, input validation at the boundary, full audit trail. Same architecture the ROC needs, just automated. The 5 tools — get_recent, classify, escalate, get_system_health, log_resolution — are exactly what an L1 triage agent needs."

---

## The 4 showcase docs (the front door)

| Doc | What it does | Time to write |
|---|---|---|
| `README.md` | Front door. Problem → architecture → results → how to run. | 20 min |
| `ARCHITECTURE.md` | System diagram + 3-layer model + production deployment path. | 20 min |
| `BENCHMARKS.md` | YOLO26s + Jetson + TensorRT numbers + profiling results. | 15 min |
| `JOB_DESCRIPTION_MAPPING.md` | **This file.** 7 JD bullets → 7 modules. The showcase index. | 5 min (just update) |

---

## What "done" means

- ✅ All 7 modules + bonus triage shipped
- ✅ All 4 docs written, cross-linked
- ✅ Tests pass for the critical paths
- ✅ Repo is public (or share-link ready)
- ✅ **Each of the 7 interview Qs above has a 60-second answer you own**

**The repo IS the showcase. The JD mapping is the proof.**

---

## Cross-references

- `SPEC.md` — the technical contract (frozen, unchanged)
- `FRAMEWORK_DESIGN.md` — the 4 core abstractions + layered architecture
- `BUILD_LEARN_PREP_PLAN.md` — the 3-layer overlay (Build + Learn + Interview Prep per step)
- `../mavis-deep-research/20260815_061300_everestlabs_interview_prep/architecture_deepdive.md` — the EverestLabs deployed architecture this prototype is positioned against
