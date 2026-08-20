"""Dependency-light smoke tests for the node collection."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_rh_nodes_test"


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


class NodeCollectionSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = load_package()
        import torch

        cls.torch = torch

    def test_registers_all_node_families(self):
        self.assertEqual(
            set(self.package.NODE_CLASS_MAPPINGS),
            {
                "CCROcclusionColorRestore",
                "CCROcclusionColorRestoreAdvanced",
                "CCROcclusionColorRestoreAdvancedV08",
                "CCROcclusionColorRestoreAdvancedV1",
                "ComicOutlineDetect",
                "BBoxMaskToReferenceImage",
            },
        )

    def test_comic_outline_preserves_batch_shape(self):
        torch = self.torch
        image = torch.ones((2, 96, 96, 3), dtype=torch.float32)
        image[:, 20:76, 20:76] = torch.tensor((0.15, 0.45, 0.85))
        result, = self.package.NODE_CLASS_MAPPINGS["ComicOutlineDetect"]().detect(
            **{
                "图像": image,
                "line_width": 3,
                "edge_percentile": 90.0,
                "alpha_thr": 24,
                "bg_threshold": 18.0,
                "invert": False,
                "add_outer_contour": True,
                "min_edge_area": -1,
            }
        )
        self.assertEqual(tuple(result.shape), tuple(image.shape))
        self.assertTrue(torch.isfinite(result).all())
        self.assertLess(float(result.min()), 0.1)

    def test_reference_colour_restore_keeps_image_shape(self):
        torch = self.torch
        reference = torch.zeros((1, 96, 96, 3), dtype=torch.float32)
        reference[:, 24:72, 24:72] = torch.tensor((0.25, 0.55, 0.85))
        corrected, *_ = self.package.NODE_CLASS_MAPPINGS[
            "CCROcclusionColorRestore"
        ]().restore(reference, reference, False)
        self.assertEqual(tuple(corrected.shape), tuple(reference.shape))
        self.assertTrue(torch.isfinite(corrected).all())

    def test_v1_returns_correction_edge_heatmap(self):
        torch = self.torch
        reference = torch.zeros((1, 96, 96, 3), dtype=torch.float32)
        reference[:, 24:72, 24:72] = torch.tensor((0.25, 0.55, 0.85))
        node_class = self.package.NODE_CLASS_MAPPINGS[
            "CCROcclusionColorRestoreAdvancedV1"
        ]
        inputs = node_class.INPUT_TYPES()["required"]
        values = {}
        for name, value in inputs.items():
            if name in {"ai_image", "reference_image"}:
                values[name] = reference
            else:
                values[name] = value[1]["default"]
        outputs = node_class().restore(**values)
        self.assertEqual(len(outputs), 8)
        self.assertEqual(tuple(outputs[-1].shape), tuple(reference.shape))
        self.assertTrue(torch.isfinite(outputs[-1]).all())



    def test_bbox_mask_reference_produces_aligned_outputs(self):
        torch = self.torch
        # 128x128 image with a white square mask region
        image = torch.full((1, 128, 128, 3), 0.5, dtype=torch.float32)
        mask = torch.zeros((1, 128, 128, 3), dtype=torch.float32)
        mask[:, 32:96, 32:96, :] = 1.0
        node = self.package.NODE_CLASS_MAPPINGS["BBoxMaskToReferenceImage"]()
        ref, crop = node.make_reference(
            image=image,
            bbox_mask=mask,
            mask_threshold=0.5,
            target_area=1024 * 1024,
            allow_upscale=True,
        )
        # reference_image is BHWC with 3 channels, H and W divisible by 16
        self.assertEqual(ref.ndim, 4)
        self.assertEqual(ref.shape[0], 1)
        self.assertEqual(ref.shape[3], 3)
        self.assertEqual(ref.shape[1] % 16, 0)
        self.assertEqual(ref.shape[2] % 16, 0)
        # bbox_crop is BHWC with 3 channels, H and W divisible by 16
        self.assertEqual(crop.ndim, 4)
        self.assertEqual(crop.shape[0], 1)
        self.assertEqual(crop.shape[3], 3)
        self.assertEqual(crop.shape[1] % 16, 0)
        self.assertEqual(crop.shape[2] % 16, 0)
        self.assertTrue(torch.isfinite(ref).all())
        self.assertTrue(torch.isfinite(crop).all())
        # crop values must be within [0, 1]
        self.assertGreaterEqual(float(crop.min()), 0.0)
        self.assertLessEqual(float(crop.max()), 1.0)

if __name__ == "__main__":
    unittest.main()
