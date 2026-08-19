"""Tests for the robustness module."""

from __future__ import annotations

import numpy as np
import pytest

from conveyor_perception.robustness.augmentations import (
    MRF_AUGMENTATIONS,
    add_gaussian_noise,
    add_salt_pepper,
    adjust_brightness,
    adjust_contrast,
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


@pytest.fixture
def test_image():
    """A test image with enough variety to exercise the augmentations."""
    np.random.seed(42)
    return (np.random.rand(480, 640, 3) * 255).astype(np.uint8)


class TestAugmentations:
    def test_adjust_brightness_darkens(self, test_image):
        result = adjust_brightness(test_image, 0.5)
        # The mean should be roughly half
        assert result.shape == test_image.shape
        assert result.mean() < test_image.mean()

    def test_adjust_brightness_brightens(self, test_image):
        result = adjust_brightness(test_image, 1.5)
        assert result.mean() > test_image.mean()

    def test_adjust_contrast_increases_range(self, test_image):
        result = adjust_contrast(test_image, 2.0)
        # Std should be larger (more contrast = more spread)
        assert result.std() > test_image.std()

    def test_add_gaussian_noise_changes_pixels(self, test_image):
        np.random.seed(0)
        result = add_gaussian_noise(test_image, 10.0)
        # Some pixels should be different
        diff = (result != test_image).sum()
        assert diff > 0

    def test_add_salt_pepper_corrupts_pixels(self, test_image):
        np.random.seed(0)
        result = add_salt_pepper(test_image, 0.05)
        # 5% of pixels should be 0 or 255
        corrupted = ((result == 0) | (result == 255)).sum()
        total = result.size
        ratio = corrupted / total
        # Should be approximately 5% (allow some tolerance)
        assert 0.02 < ratio < 0.08

    def test_gaussian_blur_smooths(self, test_image):
        result = gaussian_blur(test_image, 5)
        assert result.shape == test_image.shape
        # The std should drop (smoothing reduces variance)
        assert result.std() <= test_image.std()

    def test_motion_blur_horizontal(self, test_image):
        result = motion_blur(test_image, 9)
        assert result.shape == test_image.shape

    def test_random_occlusion_adds_black_patches(self, test_image):
        np.random.seed(0)
        result = random_occlusion(test_image, n_patches=2, patch_size=50)
        # At least one black region should exist
        assert (result == 0).any()

    def test_lens_blur_smooths(self, test_image):
        result = lens_blur(test_image, 3)
        assert result.shape == test_image.shape

    def test_color_jitter_changes_hue(self, test_image):
        result = color_jitter(test_image, 20)
        assert result.shape == test_image.shape
        # Hue changed; result should differ
        assert not np.array_equal(result, test_image)

    def test_jpeg_compress_preserves_shape(self, test_image):
        result = jpeg_compress(test_image, 50)
        assert result.shape == test_image.shape

    def test_all_augmentations_preserve_shape(self, test_image):
        for spec in MRF_AUGMENTATIONS:
            result = spec.fn(test_image)
            assert result.shape == test_image.shape, f"{spec.name} changed shape"

    def test_all_augmentations_change_image(self, test_image):
        for spec in MRF_AUGMENTATIONS:
            result = spec.fn(test_image)
            # The image should change in at least some pixels
            assert not np.array_equal(result, test_image), f"{spec.name} did not change image"

    def test_get_augmentation(self):
        spec = get_augmentation("low_light")
        assert spec.name == "low_light"
        assert "brightness" in spec.args_summary.lower() or "low" in spec.description.lower()

    def test_get_unknown_augmentation_raises(self):
        with pytest.raises(KeyError):
            get_augmentation("doesnt-exist")

    def test_all_augmentations_have_name_and_description(self):
        for spec in MRF_AUGMENTATIONS:
            assert spec.name
            assert spec.description
            assert spec.fn is not None


class TestRobustnessReport:
    def test_to_dict_includes_summary(self):
        baseline = AugmentationResult(
            name="baseline", description="", args_summary="—",
            detection_count=5, mean_confidence=0.9, inference_ms=10.0,
        )
        r1 = AugmentationResult(
            name="low_light", description="", args_summary="",
            detection_count=3, mean_confidence=0.7, inference_ms=11.0,
        )
        report = RobustnessReport(
            baseline=baseline, results=[r1],
            broken_count=0, degraded_count=1, ok_count=0,
        )
        d = report.to_dict()
        assert "baseline" in d
        assert "results" in d
        assert d["summary"]["total"] == 1
        assert d["summary"]["degraded"] == 1

    def test_to_markdown_includes_table(self):
        baseline = AugmentationResult(
            name="baseline", description="", args_summary="—",
            detection_count=5, mean_confidence=0.9, inference_ms=10.0,
        )
        report = RobustnessReport(
            baseline=baseline, results=[],
            broken_count=0, degraded_count=0, ok_count=0,
        )
        md = report.to_markdown()
        assert "# Robustness Test Report" in md
        assert "BASELINE" in md
        assert "Summary" in md

    def test_to_dict_rounds_floats(self):
        baseline = AugmentationResult(
            name="b", description="", args_summary="—",
            detection_count=5, mean_confidence=0.91234, inference_ms=12.345,
        )
        report = RobustnessReport(
            baseline=baseline, results=[],
            broken_count=0, degraded_count=0, ok_count=0,
        )
        d = report.to_dict()
        assert d["baseline"]["mean_confidence"] == 0.912
        assert d["baseline"]["inference_ms"] == 12.345


class TestRobustnessTestSuite:
    def test_runs_all_default_augmentations(self, test_image):
        class FakeDetector:
            def detect(self, frame):
                # Always return 5 detections
                class FakeD:
                    confidence = 0.9

                return [FakeD() for _ in range(5)]

        suite = RobustnessTestSuite(FakeDetector(), test_image)
        report = suite.run()
        # We should have one row per augmentation
        assert len(report.results) == len(MRF_AUGMENTATIONS)
        # Baseline is 5
        assert report.baseline.detection_count == 5
        # All results should have non-broken status (since fake detector always returns 5)
        for r in report.results:
            assert r.detection_count == 5
            assert not r.broken

    def test_broken_flag_set_when_detector_returns_zero(self, test_image):
        class FakeDetector:
            def __init__(self):
                self.call_count = 0

            def detect(self, frame):
                self.call_count += 1
                if self.call_count == 1:
                    # Baseline: 5 detections
                    class FakeD:
                        confidence = 0.9

                    return [FakeD() for _ in range(5)]
                # All augmented: 0 detections
                return []

        suite = RobustnessTestSuite(FakeDetector(), test_image)
        report = suite.run()
        # All augmented results should be broken
        assert report.broken_count == len(MRF_AUGMENTATIONS)
        for r in report.results:
            assert r.broken

    def test_degraded_flag_set_when_partial_drop(self, test_image):
        class FakeDetector:
            def __init__(self):
                self.call_count = 0

            def detect(self, frame):
                self.call_count += 1
                if self.call_count == 1:
                    return [_FakeD(0.9) for _ in range(10)]
                return [_FakeD(0.7) for _ in range(7)]  # 70% = degraded

        class _FakeD:
            def __init__(self, conf):
                self.confidence = conf

        # Re-build the detector with the right _FakeD
        class FakeDetector2:
            def __init__(self):
                self.call_count = 0

            def detect(self, frame):
                self.call_count += 1
                if self.call_count == 1:
                    return [_FakeD2(0.9) for _ in range(10)]
                return [_FakeD2(0.7) for _ in range(7)]

        class _FakeD2:
            def __init__(self, conf):
                self.confidence = conf

        suite = RobustnessTestSuite(FakeDetector2(), test_image)
        report = suite.run()
        # Each result: 7/10 = 0.7 ratio → degraded (not broken)
        assert report.degraded_count == len(MRF_AUGMENTATIONS)
        assert report.broken_count == 0

    def test_runs_per_augmentation_averages(self, test_image):
        class FakeDetector:
            def __init__(self):
                self.call_count = 0

            def detect(self, frame):
                self.call_count += 1
                # Vary count to test averaging
                if self.call_count % 2 == 1:
                    return [_FakeD2(0.9) for _ in range(5)]
                return [_FakeD2(0.8) for _ in range(3)]

        class _FakeD2:
            def __init__(self, conf):
                self.confidence = conf

        suite = RobustnessTestSuite(FakeDetector(), test_image, runs_per_augmentation=2)
        report = suite.run()
        # For each aug, we run twice; mean is (5+3)/2 = 4
        for r in report.results:
            assert r.detection_count == 4

    def test_works_with_dict_output_detector(self, test_image):
        class FakeDetector:
            def detect(self, frame):
                return {"detections": [{"conf": 0.9}] * 3}

        suite = RobustnessTestSuite(FakeDetector(), test_image)
        report = suite.run()
        # Baseline: 3 detections
        assert report.baseline.detection_count == 3
        # Ratio is 1.0 (same as baseline)
        for r in report.results:
            assert r.detection_count == 3
            assert r.detection_ratio == 1.0

    def test_runs_subset_of_augmentations(self, test_image):
        specs = [
            MRF_AUGMENTATIONS[0],  # low_light
            MRF_AUGMENTATIONS[6],  # gaussian_blur
        ]
        class FakeDetector:
            def detect(self, frame):
                return [_FakeD3(0.9) for _ in range(5)]

        class _FakeD3:
            def __init__(self, conf):
                self.confidence = conf

        suite = RobustnessTestSuite(FakeDetector(), test_image)
        report = suite.run(augmentations=specs)
        # Only 2 augmentations
        assert len(report.results) == 2
        assert report.results[0].name == "low_light"
        assert report.results[1].name == "gaussian_blur"

    def test_handles_augmentation_exception_gracefully(self, test_image):
        """If an augmentation fn raises, the test suite logs and continues."""
        from conveyor_perception.robustness.augmentations import AugmentationSpec

        def bad_fn(img):
            raise RuntimeError("intentional failure")

        specs = [
            MRF_AUGMENTATIONS[0],
            AugmentationSpec(name="bad", fn=bad_fn, description="test", args_summary="—"),
            MRF_AUGMENTATIONS[6],
        ]

        class FakeDetector:
            def detect(self, frame):
                return []

        suite = RobustnessTestSuite(FakeDetector(), test_image)
        report = suite.run(augmentations=specs)
        # The bad one is skipped; we have 2 results
        assert len(report.results) == 2
        assert all(r.name != "bad" for r in report.results)
