"""Tests for original-pixel restoration and task-specific masking."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_rh_nodes_pixel_restore_test"


def install_comfy_stubs():
    if importlib.util.find_spec("comfy") is not None:
        return
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_management.throw_exception_if_processing_interrupted = lambda: None
    utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            self.total = total

        def update(self, amount):
            return amount

    utils.ProgressBar = ProgressBar
    sys.modules.update(
        {
            "comfy": comfy,
            "comfy.model_management": model_management,
            "comfy.utils": utils,
        }
    )


def load_package():
    install_comfy_stubs()
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


class OriginalPixelRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = load_package()
        cls.restore_class = cls.package.NODE_CLASS_MAPPINGS[
            "OPR_RestoreOriginalPixels"
        ]

    def scene(self, size=64):
        y, x = np.mgrid[:size, :size]
        image = np.stack([x + 55, y + 60, (x + y) // 2 + 70], -1)
        return torch.from_numpy(image.astype(np.float32) / 255)[None]

    def test_identity_and_batch_broadcast(self):
        original = self.scene().repeat(2, 1, 1, 1)
        result = self.restore_class().restore(
            original, original[:1], method="A", capture_steps=False
        )
        self.assertTrue(torch.equal(result[0], original))
        self.assertEqual(tuple(result[0].shape), tuple(original.shape))
        self.assertEqual(float(result[1].sum()), 0)

    def test_default_method_b_runs_graph_cut(self):
        original = self.scene()
        edited = original.clone()
        edited[:, 20:44, 20:44] = 0.9
        result = self.restore_class().restore(
            original,
            edited,
            high=1.0,
            low=0.5,
            flow_limit=0,
            capture_steps=False,
        )
        stats = json.loads(result[5])[0]
        self.assertEqual(stats["method"], "B")
        self.assertEqual(tuple(result[0].shape), tuple(original.shape))
        self.assertTrue(torch.isfinite(result[0]).all())
        self.assertGreater(int(result[2].sum()), 0)
        self.assertTrue(
            torch.equal(result[0][~result[1].bool()], original[~result[1].bool()])
        )

    def test_manual_mask_preserves_outside_tensor_values(self):
        original = self.scene() + 0.00037
        edited = original.clone()
        edited[:, 20:35, 20:35] = 0.25
        mask = torch.zeros((1, 64, 64))
        mask[:, 20:35, 20:35] = 1
        result = self.restore_class().restore(
            original,
            edited,
            method="A",
            edit_mask=mask,
            padding=0,
            feather=0,
        )
        self.assertTrue(
            torch.equal(result[0][~result[1].bool()], original[~result[1].bool()])
        )

    def test_invalid_mask_range_fails(self):
        original = self.scene()
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            self.restore_class().restore(
                original,
                original,
                method="A",
                edit_mask=torch.full((1, 64, 64), 2.0),
            )

    def test_steps_render_nine_panels(self):
        original = self.scene()
        result = self.restore_class().restore(original, original, method="A")
        panels, summary = self.package.NODE_CLASS_MAPPINGS[
            "OPR_DiagnosticSteps"
        ]().render(result[6])
        self.assertEqual(tuple(panels.shape), (9, 520, 1200, 3))
        self.assertEqual(len(json.loads(summary)["steps"]), 9)

    def test_large_mask_batch_and_restricted_composite(self):
        edited = self.scene(96)
        original = edited.repeat(2, 1, 1, 1)
        original[:, 30:65, 35:65] = torch.tensor([0.8, 0.3, 0.2])
        masks = self.package.NODE_CLASS_MAPPINGS["OPR_LargeObjectMask"]().detect(
            original,
            edited,
            alignment="none",
            close_radius=2,
            padding=2,
            feather=4,
            exclude_expand=6,
        )
        self.assertEqual(len(masks), 8)
        self.assertEqual(tuple(masks[0].shape), (2, 96, 96))
        self.assertEqual(tuple(masks[4].shape), tuple(original.shape))
        self.assertGreater(int(masks[0].sum()), 0)
        output, support = self.package.NODE_CLASS_MAPPINGS[
            "OPR_RestrictedComposite"
        ]().compose(original.double(), masks[4], masks[2])
        self.assertEqual(output.dtype, torch.float64)
        self.assertTrue(
            torch.equal(output[~support.bool()], original.double()[~support.bool()])
        )

    def test_large_mask_include_and_protect_priority(self):
        original = self.scene(96)
        include = torch.zeros((1, 96, 96))
        include[:, 30:60, 30:60] = 1
        protect = torch.zeros_like(include)
        protect[:, 40:50, 40:50] = 1
        result = self.package.NODE_CLASS_MAPPINGS["OPR_LargeObjectMask"]().detect(
            original,
            original,
            alignment="none",
            include_mask=include,
            protect_mask=protect,
        )
        self.assertTrue(torch.all(result[0][(include > 0) & (protect == 0)] == 1))
        self.assertTrue(torch.all(result[2][protect > 0] == 0))

    def test_precision_core_round_trips_16_bit_rgb(self):
        precision = sys.modules[
            f"{PACKAGE_NAME}.original_pixel_restore.core.precision"
        ]
        values = np.arange(4096, dtype=np.uint16).reshape(64, 64) * 16
        rgb = np.stack([values, values[::-1], values[:, ::-1]], axis=-1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.png"
            precision.save_png(path, rgb.astype(np.float32) / 257)
            decoded, profile = precision.load_rgb(path)
        np.testing.assert_array_equal(
            np.rint(decoded * 257).astype(np.uint16), rgb
        )
        self.assertEqual(profile["bit_depth"], 16)


if __name__ == "__main__":
    unittest.main()
