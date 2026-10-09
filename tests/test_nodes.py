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
                "BBoxRestoreCropToCanvas",
                "RCMRobustMaskedColorMatch",
                "RCMCoreFeatherMask",
                "OPR_LargeObjectMask",
                "OPR_RestrictedComposite",
                "OPR_RestoreOriginalPixels",
                "OPR_LoadImageICC",
                "OPR_SaveImageICC",
                "OPR_SaveImagePrecision",
                "OPR_DiagnosticSteps",
                "MekajikiUnMult",
                "MekajikiMaskedUnMult",
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



    def _make_bbox_inputs(self):
        torch = self.torch
        image = torch.full((1, 128, 128, 3), 0.5, dtype=torch.float32)
        mask = torch.zeros((1, 128, 128, 3), dtype=torch.float32)
        mask[:, 32:96, 32:96, :] = 1.0
        return image, mask

    def test_bbox_mask_reference_stretch_outputs(self):
        torch = self.torch
        image, mask = self._make_bbox_inputs()
        node = self.package.NODE_CLASS_MAPPINGS["BBoxMaskToReferenceImage"]()
        ref, crop, crop_info, crop_info_json, stretch_info = node.make_reference(
            image=image, bbox_mask=mask, mask_threshold=0.5,
            target_area=1024 * 1024, allow_upscale=True, alignment_mode="stretch",
        )
        self.assertEqual(ref.ndim, 4)
        self.assertEqual(ref.shape[1] % 16, 0)
        self.assertEqual(ref.shape[2] % 16, 0)
        self.assertEqual(crop.ndim, 4)
        self.assertEqual(crop.shape[1] % 16, 0)
        self.assertEqual(crop.shape[2] % 16, 0)
        self.assertTrue(torch.isfinite(ref).all())
        self.assertIsInstance(crop_info, dict)
        self.assertEqual(crop_info["version"], 1)
        self.assertIsInstance(crop_info_json, str)
        self.assertEqual(stretch_info["alignment_mode"], "stretch")

    def test_bbox_mask_reference_pad_gray_outputs(self):
        torch = self.torch
        image, mask = self._make_bbox_inputs()
        node = self.package.NODE_CLASS_MAPPINGS["BBoxMaskToReferenceImage"]()
        ref, crop, crop_info, crop_info_json, stretch_info = node.make_reference(
            image=image, bbox_mask=mask, mask_threshold=0.5,
            target_area=1024 * 1024, allow_upscale=True, alignment_mode="pad_gray",
        )
        self.assertEqual(ref.shape[1] % 16, 0)
        self.assertEqual(ref.shape[2] % 16, 0)
        self.assertEqual(stretch_info["alignment_mode"], "pad_gray")
        self.assertEqual(stretch_info["padding_ltrb"][2], ref.shape[2] - stretch_info["logical_size_wh"][0])

    def test_bbox_restore_crop_to_canvas(self):
        torch = self.torch
        image, mask = self._make_bbox_inputs()
        ref_node = self.package.NODE_CLASS_MAPPINGS["BBoxMaskToReferenceImage"]()
        ref, crop, crop_info, crop_info_json, stretch_info = ref_node.make_reference(
            image=image, bbox_mask=mask, mask_threshold=0.5,
            target_area=1024 * 1024, allow_upscale=True, alignment_mode="stretch",
        )
        restore_node = self.package.NODE_CLASS_MAPPINGS["BBoxRestoreCropToCanvas"]()
        restored, alpha, working, working_alpha = restore_node.restore(
            crop=crop, crop_info=crop_info, background="white", custom_color="#000000",
            stretch_info=stretch_info,
        )
        self.assertEqual(restored.ndim, 4)
        self.assertEqual(restored.shape[3], 4)
        self.assertEqual(working.ndim, 4)
        self.assertEqual(working.shape[1] % 16, 0)
        self.assertEqual(working.shape[2] % 16, 0)
        self.assertTrue(torch.isfinite(restored).all())
        self.assertTrue(torch.isfinite(alpha).all())

if __name__ == "__main__":
    unittest.main()
