"""Robustness layer.

A library of MRF-condition augmentations + a test suite that runs the
detector on each and reports detection success rate + accuracy
degradation. The "chaotic environments" requirement from the JD.
"""

from conveyor_perception.robustness.augmentations import (
    MRF_AUGMENTATIONS,
    AugmentationSpec,
    adjust_brightness,
    adjust_contrast,
    add_gaussian_noise,
    add_salt_pepper,
    color_jitter,
    gaussian_blur,
    get_augmentation,
    jpeg_compress,
    lens_blur,
    motion_blur,
    random_occlusion,
)
from conveyor_perception.robustness.test_suite import (
    AugmentationResult,
    RobustnessReport,
    RobustnessTestSuite,
)

__all__ = [
    # Augmentations
    "MRF_AUGMENTATIONS",
    "AugmentationSpec",
    "adjust_brightness",
    "adjust_contrast",
    "add_gaussian_noise",
    "add_salt_pepper",
    "color_jitter",
    "gaussian_blur",
    "get_augmentation",
    "jpeg_compress",
    "lens_blur",
    "motion_blur",
    "random_occlusion",
    # Test suite
    "AugmentationResult",
    "RobustnessReport",
    "RobustnessTestSuite",
]
