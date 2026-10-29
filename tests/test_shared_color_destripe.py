"""Behavioral regression for broadband vertical artifact removal."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rh_destripe_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
NODE = PACKAGE.NODE_CLASS_MAPPINGS["SharedColorBroadbandDestripe"]


def fixture(bright=False):
    y, x = np.mgrid[:320, :800]
    color = np.array([.30, -.45, .84]); color /= np.linalg.norm(color)
    values = (252 if bright else 130) + (1.7 if bright else 3.5) * np.cos(
        2*np.pi*x/32 + .2*np.sin(y/43)
    )[:, :, None]*color
    return torch.from_numpy(np.rint(values).astype(np.float32)/255)[None]


class DestripeTests(unittest.TestCase):
    def test_vertical_wave_reduction_and_batch_independence(self):
        image = fixture()
        batch = torch.cat([image, image.flip(2)])
        result, text = NODE().process(batch, auto_scale=True)
        report = json.loads(text)
        self.assertEqual(len(report["frames"]), 2)
        for i in range(2):
            one, _ = NODE().process(batch[i:i+1], auto_scale=True)
            self.assertTrue(torch.equal(result[i:i+1], one))
            self.assertEqual(report["frames"][i]["status"], "processed_not_quality_approved")
            self.assertLess(float(result[i].std()), float(batch[i].std()))

    def test_near_white_evidence_opt_out(self):
        image = fixture(True)
        cleaned, text = NODE().process(image, auto_scale=True)
        legacy, _ = NODE().process(image, auto_scale=True, include_near_white=False)
        self.assertFalse(torch.equal(cleaned, image))
        self.assertTrue(torch.equal(legacy, image))
        self.assertTrue(json.loads(text)["parameters"]["include_near_white"])

    def test_zero_gains_and_no_evidence_preserve_fractional_input(self):
        image = torch.full((2, 80, 120, 4), .51234)
        for kwargs in ({"auto_scale": True}, {"gain_first": 0, "gain_second": 0}):
            result, _ = NODE().process(image, **kwargs)
            self.assertTrue(torch.equal(result, image))

    def test_rgba_nonmutation_and_default_off_api_auto(self):
        image = fixture()
        alpha = torch.rand((*image.shape[:-1], 1))
        rgba = torch.cat([image, alpha], -1); before = rgba.clone()
        result, text = NODE().process(rgba, output_mode="float32（保留处理精度）")
        self.assertTrue(torch.equal(result[..., 3:], alpha))
        self.assertTrue(torch.equal(rgba, before))
        self.assertEqual(json.loads(text)["frames"][0]["scale_mode"], "manual")
        self.assertTrue(NODE.INPUT_TYPES()["optional"]["auto_scale"][1]["default"])

    def test_invalid_inputs(self):
        for image in (torch.zeros(0, 8, 8, 3), torch.full((1, 8, 8, 3), float("nan")), torch.ones(1, 8, 8, 3)*1.01):
            with self.assertRaises(ValueError): NODE().process(image)
        with self.assertRaises(ValueError): NODE().process(fixture(), period_min=80, period_max=22)


if __name__ == "__main__":
    unittest.main()
