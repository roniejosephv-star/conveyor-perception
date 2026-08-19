"""Image augmentations for robustness testing.

MRF (Material Recovery Facility) conditions are hostile to computer vision:
- Variable lighting (overhead fluorescents, shadows from moving objects)
- Reflections on shiny metal/glass
- Motion blur from fast conveyor
- Occlusion (objects overlap)
- Sensor noise (cheap industrial cameras)
- Lens contamination (dust, oil)
- Color shift (white balance drift over time)

This module provides a library of augmentations that simulate these conditions.
The RobustnessTestSuite (in the same package) runs the detector on each
augmented variant and reports detection success + accuracy degradation.

Why augmentations, not just a robustness benchmark:
- Augmentations let us see *which* conditions break the model
- We can fix the breakage (retrain with that augmentation)
- The augmentation library is also useful for training-time data augmentation
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------- Augmentation primitives ----------


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Multiply pixel values by `factor`. 1.0 = no change."""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Multiply (pixel - 128) by `factor`, add 128. 1.0 = no change."""
    return np.clip((image.astype(np.float32) - 128) * factor + 128, 0, 255).astype(np.uint8)


def add_gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    """Add Gaussian noise with given standard deviation."""
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper(image: np.ndarray, amount: float) -> np.ndarray:
    """Salt-and-pepper noise: random pixels set to 0 or 255.

    Args:
        amount: Fraction of pixels to corrupt, in [0, 1]. 0.05 = 5%.
    """
    out = image.copy()
    h, w = image.shape[:2]
    num_salt = int(h * w * amount / 2)
    num_pepper = int(h * w * amount / 2)
    # Salt
    coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt))
    out[coords] = 255
    # Pepper
    coords = (np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper))
    out[coords] = 0
    return out


def gaussian_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    """Gaussian blur with a (ksize, ksize) kernel. ksize must be odd."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def motion_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Simulate motion blur along the horizontal axis (conveyor direction)."""
    if kernel_size < 1:
        return image
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)


def random_occlusion(image: np.ndarray, n_patches: int, patch_size: int) -> np.ndarray:
    """Add `n_patches` random black rectangles (simulates objects occluding)."""
    out = image.copy()
    h, w = image.shape[:2]
    for _ in range(n_patches):
        x = np.random.randint(0, max(1, w - patch_size))
        y = np.random.randint(0, max(1, h - patch_size))
        out[y:y + patch_size, x:x + patch_size] = 0
    return out


def lens_blur(image: np.ndarray, strength: int) -> np.ndarray:
    """Simulate lens contamination with a circular blur.

    Strength 1-15 is typical. We use cv2.bilateralFilter which preserves edges.
    """
    if strength < 1:
        return image
    return cv2.bilateralFilter(image, strength * 2, 75, 75)


def color_jitter(image: np.ndarray, hue_shift: int) -> np.ndarray:
    """Shift the hue by `hue_shift` degrees (0-180 in OpenCV HSV space)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_glare(image: np.ndarray, intensity: float, radius: int) -> np.ndarray:
    """Add a circular bright spot (simulates a glare from a light reflection)."""
    out = image.copy()
    h, w = image.shape[:2]
    # Random position for the glare center
    cx = np.random.randint(radius, w - radius)
    cy = np.random.randint(radius, h - radius)
    # Create a radial gradient mask
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.clip(1.0 - dist / radius, 0, 1) * intensity * 255
    out = np.clip(out.astype(np.float32) + mask[:, :, None], 0, 255).astype(np.uint8)
    return out


def jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    """Simulate JPEG compression artifacts at given quality (1-100)."""
    quality = max(1, min(100, int(quality)))
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image
    return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)


# ---------- Augmentation registry ----------


@dataclass
class AugmentationSpec:
    """A single named augmentation: function + args + description."""

    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    description: str
    args_summary: str = ""  # human-readable params, e.g. "factor=0.5"


# The canonical MRF-condition augmentations. Each is realistic + parametrized.
# The "moderate" severity is the default; "severe" pushes to breaking point.
MRF_AUGMENTATIONS: list[AugmentationSpec] = [
    AugmentationSpec(
        name="low_light",
        fn=lambda img: adjust_brightness(img, 0.4),
        description="40% of normal brightness (warehouse at night / power dip)",
        args_summary="brightness=0.4",
    ),
    AugmentationSpec(
        name="bright_light",
        fn=lambda img: adjust_brightness(img, 1.6),
        description="160% brightness (overhead fluorescent at full power)",
        args_summary="brightness=1.6",
    ),
    AugmentationSpec(
        name="low_contrast",
        fn=lambda img: adjust_contrast(img, 0.5),
        description="50% contrast (dust on lens, fog)",
        args_summary="contrast=0.5",
    ),
    AugmentationSpec(
        name="high_contrast",
        fn=lambda img: adjust_contrast(img, 1.8),
        description="180% contrast (direct sun on shiny metal)",
        args_summary="contrast=1.8",
    ),
    AugmentationSpec(
        name="gaussian_noise",
        fn=lambda img: add_gaussian_noise(img, 20.0),
        description="Sensor noise at high ISO (std=20)",
        args_summary="std=20",
    ),
    AugmentationSpec(
        name="salt_pepper",
        fn=lambda img: add_salt_pepper(img, 0.03),
        description="3% salt-and-pepper noise (transmission errors)",
        args_summary="amount=0.03",
    ),
    AugmentationSpec(
        name="gaussian_blur",
        fn=lambda img: gaussian_blur(img, 5),
        description="Slight out-of-focus (kernel size 5)",
        args_summary="ksize=5",
    ),
    AugmentationSpec(
        name="motion_blur",
        fn=lambda img: motion_blur(img, 9),
        description="Conveyor at 60 FPS (kernel size 9)",
        args_summary="kernel=9",
    ),
    AugmentationSpec(
        name="occlusion",
        fn=lambda img: random_occlusion(img, 3, 50),
        description="3 random black patches (overlapping objects)",
        args_summary="patches=3, size=50",
    ),
    AugmentationSpec(
        name="lens_blur",
        fn=lambda img: lens_blur(img, 3),
        description="Slight oil/dust on lens (bilateral filter strength 3)",
        args_summary="strength=3",
    ),
    AugmentationSpec(
        name="color_shift",
        fn=lambda img: color_jitter(img, 20),
        description="White balance drift (+20 hue)",
        args_summary="hue=20",
    ),
    AugmentationSpec(
        name="glare",
        fn=lambda img: add_glare(img, 0.5, 80),
        description="Light reflection / glare (intensity 0.5, radius 80)",
        args_summary="intensity=0.5, radius=80",
    ),
    AugmentationSpec(
        name="jpeg_artifacts",
        fn=lambda img: jpeg_compress(img, 30),
        description="Heavy JPEG compression (quality 30)",
        args_summary="quality=30",
    ),
]


def get_augmentation(name: str) -> AugmentationSpec:
    """Look up a named augmentation. Raises KeyError if not found."""
    for spec in MRF_AUGMENTATIONS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown augmentation: {name}. Available: {[s.name for s in MRF_AUGMENTATIONS]}")
