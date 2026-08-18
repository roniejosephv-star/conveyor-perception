# Upgrade Paths — how to keep this stack current

**Purpose:** Every dependency in this project has a known upgrade path.
When a new major version drops, this doc tells you (a) when to upgrade,
(b) what the breaking changes are, and (c) how to verify the upgrade
without breaking the project.

**Last updated:** 2026-08-19
**Verified against:** Python 3.12, macOS Darwin 24.x

---

## §1. Ultralytics (YOLO26, ByteTrack)

| Version | Released | Status | Notes |
|---|---|---|---|
| 8.4.121 | Aug 2026 | **Current** | YOLO26 is the default model. ByteTrack via `.track(tracker="bytetrack.yaml")`. |
| 9.0.x | (likely Q4 2026) | Watch | New major version, expected to have breaking API changes. |

**How to upgrade:**
```bash
pip install --upgrade ultralytics
# Re-export existing models
yolo export model=models/best.pt format=onnx imgsz=640
# Run tests
pytest tests/ -v
```

**Verify after upgrade:**
- [ ] `pytest tests/test_detection_pipeline.py` passes
- [ ] `python -c "from ultralytics import YOLO; m = YOLO('yolo26s.pt'); print(m.names)"` works
- [ ] YOLO26 still uses NMS-free output (`(1, 300, 6)` shape)
- [ ] ByteTrack tracker still works (`model.track(source=..., tracker='bytetrack.yaml')`)

**YOLO model version comparison** (COCO 640px):
- YOLO26s: 48.6 mAP, 2.5ms T4 TRT10, **NMS-free** (current)
- YOLO11s: 47.0 mAP, 2.5ms T4 TRT10, NMS required
- YOLOv8s: 44.9 mAP, 2.66ms T4 TRT10, NMS required

**Why we picked YOLO26:** +1.6 mAP over YOLO11s at the same speed, NMS-free
simplifies deployment. The DetectionPipeline abstraction auto-detects
NMS-free vs NMS-required output, so swapping models is a 1-line change
in the model_path argument.

---

## §2. OpenCV (`opencv-python`)

| Version | Released | Status | Notes |
|---|---|---|---|
| 4.13.0.92 | Aug 2026 | **Pinned** | Last 4.x before 5.0 major jump. |
| 5.0.0.93 | Aug 2026 | **Watch** | Major version. ABI changes likely. Test before adopting. |

**The 4.x → 5.0 question:** OpenCV 5.0 just dropped. We pin to 4.13.x
for stability because:
1. ultralytics 8.4.x has not been fully tested with OpenCV 5.0
2. YOLO26's NMS-free output handling depends on dnn module behavior
3. Most community tutorials + Docker images are still on 4.x

**How to test 5.0 (when you're ready):**
```bash
# In a venv, not global
python -m venv .venv-test-5.0
source .venv-test-5.0/bin/activate
pip install opencv-python==5.0.0.93
# Run all tests
pytest tests/ -v
# If everything passes, update requirements.txt
```

**Breaking changes to watch for in 5.0:**
- `cv2.dnn.readNet()` signature (probably unchanged, but verify)
- `cv2.dnn.NMSBoxes()` return type (may be numpy array instead of list)
- `cv2.dnn.blobFromImage()` parameter validation (may be stricter)

---

## §3. NumPy

| Version | Status | Notes |
|---|---|---|
| 1.26.x | **Pinned** | Last 1.x. Stable, all packages tested. |
| 2.0+ | Watch | Some breaking changes around array printing, copy semantics. |

**Why 1.26 and not 2.x:** ultralytics + opencv have had intermittent
issues with NumPy 2.0+. The 1.26.x line is the safe choice until the
ecosystem fully catches up.

**How to test 2.x:**
```bash
pip install --upgrade "numpy>=2.0,<2.3"
pytest tests/ -v
# Watch for: array copy warnings, deprecated np.float_, np.int_
```

---

## §4. FastMCP

| Version | Released | Status | Notes |
|---|---|---|---|
| 3.4.7 | Aug 2026 | **Current** | 3.x is the new API. Different from 2.x. |

**The 2.x → 3.x question:** FastMCP 3.x has breaking changes vs 2.x.
We're on 3.x from the start. If you ever need to revert to 2.x:

```python
# 2.x: from mcp.server.fastmcp import FastMCP
# 3.x: from fastmcp import FastMCP
```

The decorator API is similar but the import path differs.

**How to upgrade within 3.x:**
```bash
pip install --upgrade fastmcp
# Run the integration test
python -m conveyor_perception.triage.server &
python -m conveyor_perception.triage.cli
# Verify all 5 tools work
```

---

## §5. Pydantic

| Version | Status | Notes |
|---|---|---|
| 2.13.4 | **Current** | v2 is the right choice. v1 is end-of-life. |

**We use v2 features:** `model_dump()`, `model_dump_json()`, `Field(...)`.
These are stable in 2.x.

**How to upgrade:**
```bash
pip install --upgrade pydantic
pytest tests/ -v
```

---

## §6. Python

| Version | Status | Notes |
|---|---|---|
| 3.11 | Supported | LTS-style support. |
| 3.12 | **Pinned** | Default for this project. Stable, all wheels available. |
| 3.13 | Tested | Works, but some packages may need recompiled wheels. |
| 3.14 | Bleeding edge | Not all packages have wheels. Avoid for production. |

**Why 3.12:** The sweet spot. All packages we use have wheels, 3.12 is fast,
and it's the current stable target for scientific Python.

**How to install Python 3.12 on Mac:**
```bash
# Using pyenv (recommended)
brew install pyenv
pyenv install 3.12
pyenv global 3.12
python --version  # should be 3.12.x
```

---

## §7. ROS 2

| Distro | Released | Status | Notes |
|---|---|---|---|
| Jazzy Jalisco | May 2024 | **Current LTS** | Supported until May 2029. |
| Kilted Kaiju | May 2025 | Non-LTS | Skip for production. |
| Humble Hawksbill | May 2022 | EOL May 2027 | Don't start new projects. |

**We use Jazzy because:** it's the current LTS with the longest support
window. ROS 2 on Mac is only available via Docker (`osrf/ros:jazzy-desktop`).

**How to upgrade within Jazzy:**
```bash
# Pull the latest patch image
docker pull osrf/ros:jazzy-desktop
# Rebuild the containers
cd docker && docker compose build --pull
```

**Jazzy → Kilted migration (when Kilted gets LTS, ~May 2027):**
- Update `docker/*.Dockerfile` to use `osrf/ros:kilted-desktop`
- Update `docker-compose.yml` image references
- Rebuild + retest all ROS 2 nodes

---

## §8. Docker

| Version | Status | Notes |
|---|---|---|
| Docker Desktop 4.x | **Current** | Free for personal use. |
| Colima 0.6+ | Alternative | Open source, lighter, Mac-native. |
| OrbStack 1.x | Alternative | Fastest on Mac, $8/mo for commercial. |

**We use Docker Desktop by default** (most familiar). Colima and OrbStack
are drop-in alternatives for users who want lighter/faster.

---

## §9. Colab runtime

| Version | Status | Notes |
|---|---|---|
| Python 3.11 | **Default on Colab** | Works. |
| T4 GPU | **Default** | 16GB VRAM, 2.5ms YOLO26s inference. |
| A100 | Optional upgrade | 4x faster than T4, but limited hours. |

**How to verify the Colab demo works:**
1. Open the notebook from GitHub
2. Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Confirm the training converges, the inference runs, the JSON summary is produced

---

## §10. When to upgrade (decision rubric)

| Signal | Action |
|---|---|
| Patch version bump (e.g., 8.4.121 → 8.4.122) | Upgrade freely, run tests. |
| Minor version bump (e.g., 8.4.x → 8.5.x) | Read changelog. Test in a venv. Upgrade within a week if no breaking changes. |
| Major version bump (e.g., 4.x → 5.x) | Read migration guide. Test in a venv for at least a week. Check the broader ecosystem (ultralytics, opencv) for compatibility before adopting. |
| New Ultralytics YOLO version (YOLO27, YOLO28, ...) | Update the `DetectionPipeline` model_path. Re-verify the output format matches what the pipeline expects. Run the 3-way benchmark in `optimization/`. |
| ROS 2 distro end-of-life (Jazzy EOL May 2029) | Plan migration 6 months before EOL. Update Docker base image. |
| CUDA deprecation (e.g., CUDA 12 → 13) | Update Docker base images. Test on Colab T4 first. |

---

## §11. The "always know how to upgrade" rule

For every new dep you add:
1. Pin to the current latest
2. Add a row to this doc with the upgrade path
3. Add a test that exercises the dep (so the test fails if the API breaks)
4. Note the major version's breaking changes (if any)

This is the difference between a prototype and a maintainable system. The
first user of this code might be a recruiter in 6 months, or your future
self in 2 years. The upgrade path is the difference between "this code
still works" and "this code is a museum piece."
