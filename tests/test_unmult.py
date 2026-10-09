"""Tests for automatic and mask-guided UnMult nodes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_rh_nodes_unmult_test"


def load_package():
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec and spec.loader
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)
    return package


class UnMultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = load_package()
        cls.auto_class = package.NODE_CLASS_MAPPINGS["MekajikiUnMult"]
        cls.masked_class = package.NODE_CLASS_MAPPINGS["MekajikiMaskedUnMult"]
        cls.display_names = package.NODE_DISPLAY_NAME_MAPPINGS

    def test_legacy_ids_use_generic_display_names(self):
        self.assertEqual(
            self.display_names["MekajikiUnMult"], "UnMult (Auto Background)"
        )
        self.assertEqual(
            self.display_names["MekajikiMaskedUnMult"],
            "Mask-Guided UnMult (Uniform Background)",
        )

    def test_black_background_formula_and_batch(self):
        image = torch.tensor(
            [
                [[[0.1, 0.2, 0.4], [0.0, 0.0, 0.0]]],
                [[[0.3, 0.6, 0.1], [0.2, 0.1, 0.2]]],
            ],
            dtype=torch.float32,
        )
        rgba, alpha = self.auto_class().apply(image, auto_background=False)
        expected_alpha = image.amax(dim=-1)
        expected_rgb = torch.where(
            (expected_alpha > 0).unsqueeze(-1),
            image / expected_alpha.clamp_min(1.0e-8).unsqueeze(-1),
            torch.zeros_like(image),
        )
        torch.testing.assert_close(alpha, expected_alpha)
        torch.testing.assert_close(rgba[..., :3], expected_rgb)
        torch.testing.assert_close(rgba[..., 3], alpha)

    def test_one_two_three_and_four_channel_inputs(self):
        node = self.auto_class()
        for channels in (1, 2, 3, 4):
            image = torch.full((2, 6, 7, channels), 0.5)
            rgba, alpha = node.apply(image, auto_background=False)
            self.assertEqual(tuple(rgba.shape), (2, 6, 7, 4))
            self.assertEqual(tuple(alpha.shape), (2, 6, 7))
            self.assertEqual(rgba.dtype, torch.float32)

    def test_public_contract_preserves_existing_workflows(self):
        required = self.auto_class.INPUT_TYPES()["required"]
        self.assertEqual(
            list(required),
            ["image", "black_threshold", "dither", "auto_background"],
        )
        self.assertEqual(required["black_threshold"][1]["default"], 0)
        self.assertEqual(required["dither"][1]["default"], 0.0)
        self.assertIs(required["auto_background"][1]["default"], True)
        self.assertEqual(self.auto_class.RETURN_TYPES, ("IMAGE", "MASK"))
        self.assertEqual(
            self.auto_class.RETURN_NAMES, ("transparent_rgba", "alpha")
        )

        masked_required = self.masked_class.INPUT_TYPES()["required"]
        self.assertEqual(list(masked_required), ["image", "mask"])
        self.assertIs(
            self.masked_class.INPUT_TYPES()["optional"]["allow_alpha_adjustment"][1]["default"],
            False,
        )
        self.assertEqual(self.masked_class.RETURN_TYPES, ("IMAGE", "MASK"))
        self.assertEqual(
            self.masked_class.RETURN_NAMES, ("transparent_rgba", "alpha")
        )

    def test_auto_background_recovers_colored_matte_example(self):
        background = torch.tensor([0.2, 0.3, 0.4])
        foreground = torch.tensor([1.0, 0.5, 0.2])
        opacity = 0.6
        image = background.expand(1, 64, 64, 3).clone()
        image[:, 20:44, 20:44] = foreground * opacity + background * (
            1 - opacity
        )
        rgba, alpha = self.auto_class().apply(image)
        self.assertTrue(torch.all(alpha[:, :10, :10] == 0))
        self.assertGreater(float(alpha[:, 25:40, 25:40].mean()), 0.5)
        torch.testing.assert_close(
            rgba[:, 25:40, 25:40, 0],
            torch.ones_like(rgba[:, 25:40, 25:40, 0]),
            atol=1e-5,
            rtol=0,
        )

    def test_threshold_dither_and_nonfinite_inputs_are_safe(self):
        image = torch.full((1, 8, 8, 3), 0.5)
        plain, plain_alpha = self.auto_class().apply(
            image, black_threshold=0, dither=0, auto_background=False
        )
        first, alpha = self.auto_class().apply(
            image, black_threshold=0, dither=1, auto_background=False
        )
        second, _ = self.auto_class().apply(
            image, black_threshold=0, dither=1, auto_background=False
        )
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first[..., :3], plain[..., :3]))
        torch.testing.assert_close(alpha, plain_alpha)

        cleared, cleared_alpha = self.auto_class().apply(
            image, black_threshold=255, dither=1, auto_background=False
        )
        self.assertTrue(torch.all(cleared == 0))
        self.assertTrue(torch.all(cleared_alpha == 0))

        invalid = image.clone()
        invalid[0, 0, 0] = torch.tensor([float("nan"), float("inf"), -1])
        rgba, alpha = self.auto_class().apply(invalid, auto_background=False)
        self.assertTrue(torch.isfinite(rgba).all())
        self.assertTrue(torch.isfinite(alpha).all())
        self.assertGreaterEqual(float(rgba.min()), 0)
        self.assertLessEqual(float(rgba.max()), 1)

    def test_empty_and_invalid_channels_fail(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.auto_class().apply(torch.empty((0, 8, 8, 3)))
        with self.assertRaisesRegex(ValueError, "1, 2, 3, or 4"):
            self.auto_class().apply(torch.empty((1, 8, 8, 5)))

    def test_mask_guided_uniform_background_round_trip(self):
        background = torch.tensor([0.1, 0.25, 0.4])
        foreground = torch.tensor([0.9, 0.6, 0.2])
        alpha = torch.zeros((1, 64, 64))
        alpha[:, 20:44, 20:44] = 0.5
        composite = foreground * alpha[..., None] + background * (
            1 - alpha[..., None]
        )
        rgba, returned_alpha = self.masked_class().apply(composite, alpha)
        torch.testing.assert_close(returned_alpha, alpha)
        torch.testing.assert_close(
            rgba[:, 20:44, 20:44, :3],
            foreground.expand(1, 24, 24, 3),
            atol=2e-5,
            rtol=0,
        )
        self.assertTrue(torch.all(rgba[returned_alpha == 0] == 0))

    def test_mask_broadcast_resize_and_mismatch(self):
        image = torch.full((2, 32, 40, 3), 0.5)
        mask = torch.zeros((1, 16, 20))
        mask[:, 4:12, 5:15] = 1
        rgba, alpha = self.masked_class().apply(image, mask)
        self.assertEqual(tuple(rgba.shape), (2, 32, 40, 4))
        self.assertEqual(tuple(alpha.shape), (2, 32, 40))
        with self.assertRaisesRegex(ValueError, "batches must match"):
            self.masked_class().apply(image, mask.repeat(3, 1, 1))

    def test_channel_first_width_one_mask_is_not_transposed(self):
        image = torch.zeros((1, 5, 1, 3))
        mask = torch.linspace(0, 1, 5).reshape(1, 1, 5, 1)
        _, alpha = self.masked_class().apply(image, mask)
        torch.testing.assert_close(alpha, mask[:, 0])

    def test_mask_guided_no_background_seed_passes_visible_rgb(self):
        image = torch.rand((1, 16, 16, 3))
        mask = torch.ones((1, 16, 16))
        rgba, _ = self.masked_class().apply(image, mask)
        torch.testing.assert_close(rgba[..., :3], image)

    def test_alpha_adjustment_is_bounded_and_output_alpha_matches(self):
        image = torch.zeros((1, 64, 64, 3))
        image[:, 20:44, 20:44] = torch.tensor([0.8, 0.4, 0.1])
        mask = torch.zeros((1, 64, 64))
        mask[:, 20:44, 20:44] = 0.2
        default, alpha = self.masked_class().apply(image, mask)
        explicit, _ = self.masked_class().apply(image, mask, False)
        self.assertTrue(torch.equal(default, explicit))
        self.assertTrue(torch.equal(alpha, mask))
        adjusted, adjusted_alpha = self.masked_class().apply(image, mask, True)
        self.assertGreater(float((adjusted_alpha - mask).max()), 0)
        self.assertTrue(torch.all(adjusted_alpha >= mask))
        self.assertLessEqual(float((adjusted_alpha - mask).max()), 43 / 255 + 1e-7)
        self.assertTrue(torch.equal(adjusted[..., 3], adjusted_alpha))
        self.assertTrue(torch.isfinite(adjusted).all())
        self.assertGreaterEqual(float(adjusted.min()), 0)
        self.assertLessEqual(float(adjusted.max()), 1)

    def test_gamut_overflow_preserves_nonnegative_channel_ratios(self):
        image = torch.zeros((1, 64, 64, 3))
        image[:, 20:44, 20:44] = torch.tensor([0.8, 0.4, 0.1])
        mask = torch.zeros((1, 64, 64))
        mask[:, 20:44, 20:44] = 0.2
        rgba, _ = self.masked_class().apply(image, mask)
        torch.testing.assert_close(rgba[0, 30, 30, :3], torch.tensor([1.0, 0.5, 0.125]))


if __name__ == "__main__":
    unittest.main()
