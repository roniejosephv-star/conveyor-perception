# conveyor-perception

> **End-to-end industrial perception stack for conveyor-based visual inspection.**
> YOLO26s + OpenCV DNN + ByteTrack + ROS 2 + MCP-style L1 triage.
> 4 core abstractions + 7 modules. Model-agnostic via framework.
> Live demo via Colab (free T4 GPU) + Docker Compose (multi-container ROS 2 + RViz).
> Jetson Orin Nano deployment path documented.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![YOLO26](https://img.shields.io/badge/YOLO26-current-orange.svg)](https://docs.ultralytics.com/models/yolo26)
[![ROS 2 Jazzy](https://img.shields.io/badge/ROS_2-Jazzy-blue.svg)](https://docs.ros.org/en/jazzy/)

---

## What is this?

A reference implementation of a production-pattern industrial perception
stack. The same architecture runs in recycling plants, warehouses, and
manufacturing lines: a camera on a conveyor → a real-time detector → a
tracker → an event stream → an L1 triage agent.

**The 4 framework abstractions** (model-agnostic, reusable):
- `DetectionPipeline` — YOLO26 + OpenCV DNN inference, NMS-free
- `TrackingPipeline` — ByteTrack multi-object tracking with stable IDs
- `MCPTriageSurface` — FastMCP server scaffold for L1 alert triage
- `DriftMonitor` — KS test on per-class confidence + count anomaly + latency regression

**The 7 domain modules** (one per JD responsibility, built on the framework):
- `perception/` — detector + tracker + inference loop (JD: real-time detection)
- `predictive_maintenance/` — pick-pattern + encoder health + air pressure (JD: beyond sorting)
- `multitask/` — classifier + anomaly + time-series (JD: 4 model types)
- `integration/` — ROS 2 node + robot pick simulator + Jetson deploy (JD: integrate with robots)
- `robustness/` — augmentations + domain-shift tests (JD: chaotic environments)
- `monitoring/` — drift + retraining trigger (JD: monitor + catch drift)
- `triage/` — 5-tool MCP L1 agent (bonus: the ROC's secret weapon)

---

## Quick start

### Option A: Run the live demo in Colab (easiest)

1. Open the Colab notebook: [TBD — added on Day 6]
2. Click "Run all"
3. Watch the full pipeline train + infer + triage in ~20 minutes

### Option B: Run locally on Mac (full Docker Compose stack)

```bash
# 1. Clone
git clone https://github.com/roniejosephv-star/conveyor-perception.git
cd conveyor-perception

# 2. Create venv + install
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Start the Docker Compose stack (perception + camera_sim + RViz + MCP triage)
cd docker
docker compose up
# RViz opens in a new window. Conveyor images flow. Bounding boxes appear.

# 4. Run the CLI driver to walk the alert queue
python -m conveyor_perception.triage.cli
```

### Option C: Run just the perception pipeline (no Docker)

```bash
# Train a YOLO26s model on the Roboflow "Recyclable Waste" dataset
python scripts/train_yolo26.py

# Run inference on a sample video
python -m conveyor_perception.app.conveyor --source data/sample/video.mp4
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: End-to-end example                                     │
│   examples/conveyor_demo.py — wires Layer 1 + uses Layer 2      │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Domain modules (one per JD bullet)                     │
│   perception/             → DetectionPipeline                   │
│   predictive_maintenance/ → DetectionPipeline                   │
│   multitask/              → DetectionPipeline                   │
│   integration/            → DetectionPipeline + TrackingPipeline │
│   robustness/             → DetectionPipeline                   │
│   monitoring/             → DriftMonitor                        │
│   optimization/           → DetectionPipeline + TensorRT        │
│   triage/                 → MCPTriageSurface                    │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Core abstractions (the framework)                     │
│   DetectionPipeline    — YOLO26 + OpenCV DNN wrapper           │
│   TrackingPipeline     — ByteTrack wrapper                      │
│   MCPTriageSurface     — FastMCP server scaffold                │
│   DriftMonitor         — drift detection + health check         │
└─────────────────────────────────────────────────────────────────┘
```

See `docs/FRAMEWORK_DESIGN.md` for the full design and `docs/JOB_DESCRIPTION_MAPPING.md`
for the JD → module index.

---

## The YOLO model choice (and how to upgrade)

**We ship YOLO26s by default.** It's the current Ultralytics default
(Jan 2026 release), +1.6 mAP better than YOLO11s at the same speed, and
NMS-free architecture simplifies production deployment.

| Model | mAP (COCO) | T4 TRT10 | NMS? | Status |
|---|---|---|---|---|
| **YOLO26s** | **48.6** | **2.5ms** | **No (native end-to-end)** | **Current** |
| YOLO11s | 47.0 | 2.5ms | Required | Previous default |
| YOLOv8s | 44.9 | 2.66ms | Required | Legacy |

The `DetectionPipeline` abstraction is **model-agnostic** — swap the
model file, the pipeline still works. The pipeline auto-detects
NMS-free vs NMS-required output. To upgrade to a future YOLO version:

```python
# Just change the model path
pipeline = DetectionPipeline(
    model_path="models/yolo27s.onnx",  # was: yolo26s.onnx
    class_names=class_names,
)
```

**See `docs/UPGRADE_PATHS.md` for the full upgrade strategy.**

---

## Project structure

```
conveyor-perception/
├── README.md                          # this file
├── LICENSE                            # MIT
├── pyproject.toml                     # project metadata + deps
├── requirements.txt                   # pinned versions
├── docs/
│   ├── FRAMEWORK_DESIGN.md            # the 4 core abstractions
│   ├── JOB_DESCRIPTION_MAPPING.md     # JD → module index
│   ├── BENCHMARKS.md                  # YOLO26 vs 11 vs 8 numbers
│   ├── ARCHITECTURE.md                # system diagram + production path
│   └── UPGRADE_PATHS.md               # how to upgrade each dep
├── src/conveyor_perception/
│   ├── core/                          # Layer 1: the 4 abstractions
│   │   ├── detection_pipeline.py
│   │   ├── tracking_pipeline.py
│   │   ├── triage_surface.py
│   │   └── drift_monitor.py
│   ├── perception/                    # Layer 2: JD bullet 1
│   ├── triage/                        # Layer 2: bonus (L1 agent)
│   ├── integration/                   # Layer 2: JD bullet 4 (ROS 2)
│   ├── predictive_maintenance/        # Layer 2: JD bullet 2
│   ├── multitask/                     # Layer 2: JD bullet 3
│   ├── robustness/                    # Layer 2: JD bullet 5
│   ├── monitoring/                    # Layer 2: JD bullet 6
│   ├── optimization/                  # Layer 2: JD bullet 7
│   └── app/
│       └── conveyor.py                # end-to-end demo entry point
├── tests/                             # pytest unit tests
├── notebooks/                         # Colab demo notebook
├── docker/                            # multi-container ROS 2 stack
├── scripts/                           # train, download, benchmark scripts
├── examples/                          # end-to-end example
├── data/                              # sample + raw + processed
└── models/                            # trained ONNX files (gitignored)
```

---

## Development

### Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,notebooks]"
```

### Test

```bash
pytest tests/ -v
pytest tests/ --cov=conveyor_perception  # with coverage
```

### Lint

```bash
ruff check src/ tests/
mypy src/
```

### Run a single test file

```bash
pytest tests/test_detection_pipeline.py -v
```

---

## The interview pitch

> *"I built a 4-abstraction framework (DetectionPipeline, TrackingPipeline,
> MCPTriageSurface, DriftMonitor) that ships YOLO26s as the default but
> lets you swap to YOLO11s or YOLOv8s in one line. The live demo runs in
> Colab on a free T4 — you click the link, run all cells, and see the full
> pipeline train + infer + triage in 20 minutes. The YOLO26 NMS-free
> architecture means the deployment is simpler than the tutorials you'll
> find online, which still use YOLOv8. The MCP triage surface is the
> differentiator — it's the tool that lets an L1 alert agent scale the ROC
> to 100+ sites without 100 humans."*

**The 7 interview talking points** (one per JD bullet) are in
`docs/JOB_DESCRIPTION_MAPPING.md`. Each is a 60-second whiteboard answer.

---

## Roadmap

| Phase | What | Status |
|---|---|---|
| **Day 1, A** | Foundation + 4 abstractions | ✅ Done (this commit) |
| **Day 1, B** | `perception/` module + YOLO26s training | ⏳ Day 1 |
| **Day 2, C** | Triage + PM + multitask modules | ⏳ Day 2 |
| **Day 2, D** | Integration (ROS 2) + robustness | ⏳ Day 2 |
| **Day 3, E** | Monitoring + optimization + 3-way benchmark | ⏳ Day 3 |
| **Day 3, F** | Colab demo + docs + push | ⏳ Day 3 |
| **Day 4** | Real TensorRT benchmark on Colab T4 + Jetson deploy package | ⏳ Day 4 |
| **Day 5** | Mavis `conveyor-perception-coach` custom agent | ⏳ Day 5 |
| **Day 6** | Docker Compose stack + final docs + share link | ⏳ Day 6 |

---

## License

MIT — see [LICENSE](LICENSE).
