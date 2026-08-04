"""Dependency-light smoke tests for the node collection."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_dakun333_nodes_test"


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
                "ComicOutlineDetect",
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


if __name__ == "__main__":
    unittest.main()
