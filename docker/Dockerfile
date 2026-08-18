# Dockerfile for the conveyor-perception ROC service
#
# This is the production image for the Remote Operations Center (ROC) Ubuntu
# machines. It bundles:
#   - Python 3.12 + the pinned deps from requirements.txt
#   - The YOLO26 model + ONNX export (TensorRT optional, see scripts/export_tensorrt.py)
#   - The conveyor-perception package
#   - The ROS 2 environment (Humble or Jazzy, picked at build time)
#
# Build:
#   docker build -t conveyor-perception:latest -f docker/Dockerfile .
#
# Run (with a mounted data + models directory):
#   docker run -it --rm \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     -p 8080:8080 \
#     conveyor-perception:latest
#
# Multi-container orchestration: see docker-compose.yml
#
# Why a Dockerfile (vs Conda / pyenv / system Python):
#   - Production ROC machines are uniform Ubuntu 22.04; one image, many hosts
#   - Image is the deployment unit; CI builds, registry stores, ROC pulls
#   - Layered cache speeds up CI by skipping deps when only code changes
#
# Why python:3.12-slim (not -alpine, not full ubuntu):
#   - Slim has glibc, which is what OpenCV DNN needs (alpine uses musl)
#   - 3.12 matches the dev box; package wheels all available
#   - Slim saves ~700MB vs full ubuntu

FROM python:3.12-slim AS base

# System deps for OpenCV, supervision, etc.
# - libgl1 / libglib2.0-0: OpenCV runtime
# - libsm6 / libxext6 / libxrender1: OpenCV image I/O
# - ffmpeg: video codec support
# - git: pip install from git URLs (e.g., some Ultralytics deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer for fast rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the conveyor-perception package itself
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Copy scripts and examples (entry points)
COPY scripts ./scripts
COPY examples ./examples

# Copy the trained model + ONNX export. These are typically mounted at
# runtime in production, but we copy them for the demo image.
# (Uncomment if you want to bake the model into the image.)
# COPY models ./models

# Data is mounted, not copied (datasets are large + change often).
# Run with: -v $(pwd)/data:/app/data

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default: run the CLI driver
# Override with: docker run conveyor-perception:latest python -m conveyor_perception.app.conveyor --source ...
ENTRYPOINT ["python"]
CMD ["-m", "conveyor_perception.app.conveyor", "--source", "data/sample/bus.jpg", "--max-frames", "100"]
