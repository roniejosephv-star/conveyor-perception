"""Multitask pipeline layer.

Runs detection + tracking + drift + triage on every frame, sharing a single
input. The architecture decouples each concern so any component can be
swapped for a production-grade version (e.g., Roboflow-trained detector,
Kafka-backed alert queue) without touching the others.
"""

from conveyor_perception.multitask.pipeline import FrameResult, MultitaskPipeline

__all__ = ["FrameResult", "MultitaskPipeline"]
