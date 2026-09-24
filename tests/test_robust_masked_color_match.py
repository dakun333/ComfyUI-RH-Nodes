"""Tests for robust masked color matching."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_rh_nodes_robust_match_test"


def load_package():
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


def fixture():
    rng = np.random.default_rng(5)
    target = rng.uniform(0.15, 0.75, (96, 112, 3))
    matrix = np.array(
        [
            [0.97, 0.02, 0.01],
            [0.01, 1.02, -0.01],
            [0.01, -0.02, 1.03],
            [-0.01, 0.02, 0.01],
        ]
    )
    reference = target @ matrix[:3] + matrix[3]
    mask = np.zeros(target.shape[:2])
    mask[35:60, 45:70] = 1
    return reference, target, mask, matrix


class RobustMaskedColorMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = load_package()
        cls.core = sys.modules[
            f"{PACKAGE_NAME}.robust_masked_color_match.core"
        ]
        cls.match_class = package.NODE_CLASS_MAPPINGS["RCMRobustMaskedColorMatch"]
        cls.feather_class = package.NODE_CLASS_MAPPINGS["RCMCoreFeatherMask"]

    def tensors(self):
        reference, target, mask, _ = fixture()
        return tuple(
            torch.from_numpy(value).to(dtype=torch.float32)
            for value in (reference[None], target[None], mask[None])
        )

    def test_recovers_affine_and_corrects_excluded_region(self):
        reference, target, mask, matrix = fixture()
        changed_reference = reference.copy()
        changed_reference[mask > 0] = 1
        corrected, region, report = self.core.match_frame(
            changed_reference, target, mask, exclude_expand=4
        )
        np.testing.assert_allclose(corrected, reference, atol=1e-10)
        np.testing.assert_allclose(
            report["coefficients_rgb_rowvector"], matrix, atol=1e-10
        )
        self.assertTrue(np.all(region[mask > 0] == 0))

    def test_batch_broadcast_shape_and_report(self):
        reference, target, mask = self.tensors()
        target = target.repeat(2, 1, 1, 1)
        image, region, report = self.match_class().match(
            reference, target, mask, exclude_expand=4
        )
        self.assertEqual(tuple(image.shape), tuple(target.shape))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(tuple(region.shape), (2, 96, 112))
        self.assertEqual(len(json.loads(report)["frames"]), 2)
        torch.testing.assert_close(image[0], image[1])

    def test_outliers_fallback_and_strength(self):
        reference, target, mask, matrix = fixture()
        noisy = reference.copy()
        noisy[::9, ::3] = [0.8, 0.2, 0.7]
        _, _, report = self.core.match_frame(
            noisy, target, mask, exclude_expand=4
        )
        np.testing.assert_allclose(
            report["coefficients_rgb_rowvector"], matrix, atol=0.002
        )

        uniform_reference = np.full((50, 60, 3), [0.25, 0.4, 0.35])
        uniform_target = uniform_reference + 0.025
        corrected, _, report = self.core.match_frame(
            uniform_reference, uniform_target, np.zeros((50, 60))
        )
        self.assertEqual(report["fit_model"], "rgb_offset_fallback")
        np.testing.assert_allclose(corrected, uniform_reference, atol=1e-10)

        full, _, _ = self.core.match_frame(
            reference, target, mask, exclude_expand=4
        )
        partial, _, _ = self.core.match_frame(
            reference, target, mask, strength=0.25, exclude_expand=4
        )
        np.testing.assert_allclose(
            partial, 0.75 * target + 0.25 * full, atol=1e-12
        )

    def test_empty_and_full_masks_are_actionable(self):
        reference, target, mask, _ = fixture()
        _, _, report = self.core.match_frame(reference, target, mask * 0)
        self.assertTrue(any("empty" in warning for warning in report["warnings"]))
        with self.assertRaisesRegex(ValueError, "White means EXCLUDE"):
            self.core.match_frame(reference, target, mask * 0 + 1)

    def test_distinct_batch_frames_are_independent(self):
        reference, target, mask = self.tensors()
        references = torch.cat([reference, reference + 0.02])
        images, _, _ = self.match_class().match(
            references, target.repeat(2, 1, 1, 1), mask, exclude_expand=4
        )
        torch.testing.assert_close(
            images[1] - images[0],
            torch.full_like(images[0], 0.02),
            atol=2e-6,
            rtol=0,
        )

    def test_invalid_alignment_and_mask_shapes_fail(self):
        reference, target, mask = self.tensors()
        with self.assertRaisesRegex(ValueError, "identical"):
            self.match_class().match(reference[:, :-1], target, mask)
        with self.assertRaises(ValueError):
            self.match_class().match(reference, target, mask[..., None])

    def test_feather_preserves_core_and_output_batch(self):
        reference, _, mask = self.tensors()
        (feather,) = self.feather_class().feather(
            mask[:, :-1], reference.repeat(2, 1, 1, 1), mask_resize="nearest"
        )
        self.assertEqual(tuple(feather.shape), (2, 96, 112))
        self.assertEqual(feather.dtype, torch.float32)
        self.assertTrue(torch.all(feather[:, 40:55, 50:65] == 1))


if __name__ == "__main__":
    unittest.main()
