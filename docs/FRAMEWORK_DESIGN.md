# Framework Design — The 4 Core Abstractions

**Date:** 2026-08-19
**Status:** Active

The framework has **4 core abstractions** in `industrial_cv_prototype/core/`. These are the *engines* — generic, reusable, no domain logic. The **8 domain modules** (in `industrial_cv_prototype/`) are *use cases* of these engines.

This is the **Option 1 + Option 3 combined** end state: the framework mindset (abstraction + reuse) AND the JD coverage (8 modules, one per JD responsibility).

---

## The layered architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: End-to-end example                                     │
│   examples/conveyor_demo.py — wires Layer 1 + uses Layer 2      │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Domain modules (one per JD bullet)                     │
│   perception/             → DetectionPipeline      (JD bullet 1)│
│   predictive_maintenance/ → DetectionPipeline       (JD bullet 2)│
│   multitask/              → DetectionPipeline       (JD bullet 3)│
│   integration/            → Detection + Tracking   (JD bullet 4)│
│   robustness/             → DetectionPipeline       (JD bullet 5)│
│   monitoring/             → DriftMonitor            (JD bullet 6)│
│   optimization/           → DetectionPipeline       (JD bullet 7)│
│   triage/                 → MCPTriageSurface        (bonus)      │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Core abstractions (the framework)                     │
│   DetectionPipeline    — YOLO26s + OpenCV DNN wrapper           │
│   TrackingPipeline     — ByteTrack wrapper                      │
│   MCPTriageSurface     — FastMCP server scaffold                │
│   DriftMonitor         — drift detection + health check         │
└─────────────────────────────────────────────────────────────────┘
```

---

## The 4 abstractions (Layer 1)

### 1. `DetectionPipeline` — `core/detection_pipeline.py`
**Purpose:** Wrap the YOLO26s + OpenCV DNN inference loop. Take frames, return detections.

```python
class DetectionPipeline:
    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.4, device: str = "cpu"):
        ...
    def load(self) -> None: ...                            # Load ONNX into OpenCV DNN
    def preprocess(self, frame): ...                       # Letterbox + scale + BGR→RGB
    def infer(self, frame) -> List[Detection]: ...         # Run detection, return list
    def postprocess(self, output, scale, padding): ...     # Parse + NMS + scale back

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    track_id: Optional[int] = None
```

**Used by:** perception/, predictive_maintenance/, multitask/, robustness/, optimization/

### 2. `TrackingPipeline` — `core/tracking_pipeline.py`
**Purpose:** Wrap ByteTrack. Take detections across frames, return stable IDs.

```python
class TrackingPipeline:
    def __init__(self, track_thresh: float = 0.5,
                 match_thresh: float = 0.8, frame_rate: int = 30):
        ...
    def update(self, detections: List[Detection]) -> List[Detection]:
        # Adds stable track_id to each detection. Maintains ID across frames.
        ...
```

**Used by:** integration/ (robot pick simulator uses the track_id)

### 3. `MCPTriageSurface` — `core/triage_surface.py`
**Purpose:** FastMCP server scaffold. Subclass to add domain-specific tools.

```python
class MCPTriageSurface:
    def __init__(self, name: str, alert_source: 'AlertSource'):
        ...
    def tool(self, func): ...                  # @tool decorator
    def resource(self, uri_pattern: str): ...  # @resource decorator
    def run(self, transport: str = "stdio"): ...

class AlertSource(Protocol):
    def get_recent(self, limit: int) -> List[Alert]: ...
    def classify(self, alert_id: str) -> str: ...
    def escalate(self, alert_id: str, reason: str) -> None: ...
    def get_health(self) -> SystemHealth: ...
    def log_resolution(self, alert_id: str, action: str) -> None: ...
```

**Used by:** triage/ (the L1 agent implementation — 5 tools, CLI driver)

### 4. `DriftMonitor` — `core/drift_monitor.py`
**Purpose:** Drift detection + health check. Watch production signals, fire alerts.

```python
class DriftMonitor:
    def __init__(self, baseline_window: int = 1000,
                 drift_threshold: float = 0.05):
        ...
    def update(self, signal: ProductionSignal) -> None: ...
    def check_drift(self) -> Optional[DriftAlert]: ...   # KS test on per-class confidence
    def get_health(self) -> SystemHealth: ...            # Throughput, latency, errors

@dataclass
class ProductionSignal:
    class_id: int
    confidence: float
    inference_time_ms: float
    timestamp: float

@dataclass
class DriftAlert:
    drift_type: str   # "confidence", "class_count", "latency"
    severity: str     # "info", "warn", "critical"
    details: dict
```

**Used by:** monitoring/ (drift detection + retraining trigger)

---

## Why this design (the senior-engineering pitch)

- **Layered** — each layer can be tested independently. The framework abstractions have unit tests; the modules have integration tests. This is the bar for production code, not notebook demos.
- **Reusable** — the abstractions are domain-agnostic. A different team could use `DetectionPipeline` for medical imaging, autonomous driving, or quality inspection by just swapping the model.
- **Explicit patterns** — the JD asks for "tool-surface architecture" and "production ML lifecycle". The 4 abstractions make these patterns explicit, not implicit. The recruiter can *see* the engineering thinking, not just the code.
- **Senior vs junior** — the difference between "I built a thing" and "I built a thing that other people can build on top of" is abstraction. This design shows the latter.

---

## The 8 modules (Layer 2) — what each uses from Layer 1

| Module | Layer 1 abstractions used | JD bullet |
|---|---|---|
| `perception/` | DetectionPipeline | 1. Real-time detection |
| `predictive_maintenance/` | DetectionPipeline + custom signals | 2. Beyond sorting (differentiator) |
| `multitask/` (classifier + anomaly + time_series) | DetectionPipeline outputs | 3. Object detection, classification, anomaly, time-series |
| `integration/` (robot_pick_simulator) | DetectionPipeline + TrackingPipeline | 4. RecycleOS + robots |
| `robustness/` | DetectionPipeline + augmentations | 5. Chaotic environments |
| `monitoring/` | DriftMonitor | 6. Drift + retraining |
| `optimization/` | DetectionPipeline + TensorRT | 7. Edge inference |
| `triage/` | MCPTriageSurface | 8. L1 ROC triage agent |

---

## The end-to-end example (Layer 3)

`examples/conveyor_demo.py` wires all 4 abstractions into one runnable demo:

```python
# Pseudocode — actual file written in Step 16
detection = DetectionPipeline(model_path='models/best.onnx')
tracking = TrackingPipeline(frame_rate=30)
drift = DriftMonitor()
triage = MCPTriageSurface(name='l1-triage', alert_source=conveyor_alert_source)

for frame in video_source:
    detections = detection.infer(frame)
    tracked = tracking.update(detections)
    for d in tracked:
        drift.update(ProductionSignal(d.class_id, d.confidence,
                                      time_ms, timestamp))
    conveyor_alert_source.add(tracked)

# In parallel: triage.run() — MCP server accepts tool calls
# CLI driver: triage.classify(...) → triage.escalate(...) or log_resolution(...)
```

The recruiter runs this one command and sees the whole framework in action.

---

## Cross-references
- `SPEC.md` — the technical contract (frozen)
- `JOB_DESCRIPTION_MAPPING.md` — the JD → module index
- `BUILD_LEARN_PREP_PLAN.md` — the 3-layer overlay (Build + Learn + Interview Prep)
