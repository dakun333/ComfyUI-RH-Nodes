"""Transparent PNG interoperability and alpha-aware numerical regressions."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_shared_color_destripe import NODE, SPEC, fixture
import sys

NODES = sys.modules[SPEC.name + '.shared_color_destripe.nodes']


def rgba_fixture(opacity=1.):
    y, x = np.mgrid[:192, :320]
    color = np.array([.35, -.5, .8]); color /= np.linalg.norm(color)
    rgb = np.rint(150 + 3.5*np.cos(2*np.pi*x/12 + .2*np.sin(y/25))[:, :, None]*color).astype(np.float32)/255
    alpha = np.zeros((192, 320), np.float32)
    alpha[24:168, 24:296] = opacity
    alpha[20:24, 24:296] = opacity*.3
    alpha[168:172, 24:296] = opacity*.3
    rgb[alpha == 0] = 0
    return torch.from_numpy(np.concatenate([rgb, alpha[:, :, None]], axis=2))[None]


class TransparentTests(unittest.TestCase):
    def test_mask_appended_without_output_changes(self):
        self.assertEqual(NODE.RETURN_TYPES, ('IMAGE', 'STRING'))
        self.assertEqual(list(NODE.INPUT_TYPES()['optional']), ['auto_scale', 'include_near_white', 'mask'])
        self.assertEqual(NODE.INPUT_TYPES()['optional']['mask'][0], 'MASK')

    def test_rgb_mask_matches_rgba_and_reduces_waves(self):
        x = rgba_fixture()
        direct, text = NODE().process(x, auto_scale=True)
        separate, report = NODE().process(x[..., :3], mask=1-x[..., 3], auto_scale=True)
        self.assertTrue(torch.equal(direct, separate))
        self.assertTrue(torch.equal(direct[..., 3], x[..., 3]))
        row = json.loads(text)['frames'][0]
        self.assertEqual(row['status'], 'processed_not_quality_approved')
        self.assertGreaterEqual(row['support_tiles'], 6)
        self.assertGreaterEqual(row['scale_estimation']['spectral_tile_count'], 6)
        self.assertLess(float(direct[0, 40:150, 50:270, :3].std()), float(x[0, 40:150, 50:270, :3].std()))
        self.assertEqual(json.loads(report)['frames'][0]['alpha_source'], 'mask')

    def test_hidden_rgb_independence_and_preservation(self):
        x = rgba_fixture(); hidden = x[..., 3] == 0; alternate = x.clone()
        alternate[..., :3][hidden] = torch.rand((int(hidden.sum()), 3), generator=torch.Generator().manual_seed(122))
        for mode in (NODES.DIRECTED, NODES.FLOAT):
            a, ta = NODE().process(x, auto_scale=True, output_mode=mode)
            b, tb = NODE().process(alternate, auto_scale=True, output_mode=mode)
            self.assertTrue(torch.equal(a[..., :3][~hidden], b[..., :3][~hidden]))
            self.assertTrue(torch.equal(b[..., :3][hidden], alternate[..., :3][hidden]))
            for key in ('effective_period_min', 'estimated_color_direction'):
                self.assertEqual(json.loads(ta)['frames'][0][key], json.loads(tb)['frames'][0][key])

    def test_opaque_inputs_retain_rgb_algorithm(self):
        x = fixture()
        for auto in (False, True):
            rgb, _ = NODE().process(x, auto_scale=auto)
            mask, _ = NODE().process(x, mask=torch.zeros(x.shape[:3]), auto_scale=auto)
            rgba, _ = NODE().process(torch.cat([x, torch.ones((*x.shape[:3], 1))], -1), auto_scale=auto)
            self.assertTrue(torch.equal(rgb, mask[..., :3]))
            self.assertTrue(torch.equal(rgb, rgba[..., :3]))
            self.assertTrue(torch.all(mask[..., 3] == 1))

    def test_loadimage_empty_mask_sentinel(self):
        x = fixture(); rgb, _ = NODE().process(x)
        out, _ = NODE().process(x, mask=torch.zeros(1, 64, 64))
        self.assertTrue(torch.equal(out[..., :3], rgb))
        self.assertTrue(torch.all(out[..., 3] == 1))

    def test_half_opacity_evidence_and_faint_shadow(self):
        x = rgba_fixture(.5); x[:, 10:16, :, 3] = .02; x[:, 10:16, :, :3] = .12345
        for mode in (NODES.DIRECTED, NODES.FLOAT):
            y, text = NODE().process(x, auto_scale=True, output_mode=mode)
            self.assertEqual(json.loads(text)['frames'][0]['status'], 'processed_not_quality_approved')
            self.assertTrue(torch.equal(y[..., 3], x[..., 3]))
            self.assertTrue(torch.equal(y[:, 10:16], x[:, 10:16]))

    def test_no_evidence_and_zero_gains_preserve_original(self):
        transparent = torch.rand(1, 120, 160, 4); transparent[..., 3] = 0
        uniform = rgba_fixture(); uniform[..., :3] = .63
        tiny = torch.rand(1, 12, 16, 4)
        for x in (transparent, uniform, tiny):
            y, text = NODE().process(x, auto_scale=True)
            self.assertTrue(torch.equal(y, x))
            self.assertNotEqual(json.loads(text)['frames'][0]['status'], 'processed_not_quality_approved')
        x = rgba_fixture(); x[..., :3] += .17/255
        with patch.object(NODES, 'estimate_scale_alpha', side_effect=AssertionError('must bypass')):
            y, text = NODE().process(x[..., :3], mask=1-x[..., 3], auto_scale=True, gain_first=0, gain_second=0)
        self.assertTrue(torch.equal(y, x))

    def test_mask_broadcast_hw_and_mixed_batch_independence(self):
        x = rgba_fixture(); batch = torch.cat([x[..., :3], x[..., :3]+.1/255])
        mask = 1-x[..., 3]; before = mask.clone(); original = batch.clone()
        outputs, _ = NODE().process(batch, mask=mask[0], auto_scale=True)
        for index in range(2):
            one, _ = NODE().process(batch[index:index+1], mask=mask, auto_scale=True)
            self.assertTrue(torch.equal(one, outputs[index:index+1]))
        self.assertTrue(torch.equal(mask, before)); self.assertTrue(torch.equal(batch, original))
        opaque = x.clone(); opaque[..., 3] = 1
        mixed = torch.cat([x, opaque])
        outputs, text = NODE().process(mixed, auto_scale=True)
        self.assertEqual([r['alpha_aware'] for r in json.loads(text)['frames']], [True, False])
        for index in range(2):
            one, _ = NODE().process(mixed[index:index+1], auto_scale=True)
            self.assertTrue(torch.equal(one, outputs[index:index+1]))

    def test_invalid_or_conflicting_masks(self):
        x = rgba_fixture()
        masks = [torch.zeros(1, 60, 70), torch.zeros(2, 192, 320), torch.ones(1, 64, 64),
                 torch.zeros(1, 192, 320, 1), torch.full((1, 192, 320), float('nan')),
                 torch.full((1, 192, 320), 1.1), torch.zeros(1, 192, 320, dtype=torch.uint8),
                 torch.zeros(0, 192, 320), 'wrong']
        for mask in masks:
            with self.subTest(mask_type=type(mask).__name__):
                with self.assertRaises(ValueError): NODE().process(x[..., :3], mask=mask)
        with self.assertRaises(ValueError): NODE().process(x, mask=torch.zeros(x.shape[:3]))
        y, _ = NODE().process(x, mask=1-x[..., 3], auto_scale=True)
        self.assertTrue(torch.equal(y[..., 3], x[..., 3]))

    def test_byte_alpha_roundtrip_and_high_precision_mask(self):
        alpha = (torch.arange(256, dtype=torch.float32)/255)[None, None].expand(1, 80, 256).clone()
        rgb = torch.full((1, 80, 256, 3), .5)
        output, text = NODE().process(rgb, mask=1-alpha, auto_scale=True)
        np.testing.assert_array_equal((output[0, ..., 3].numpy()*255).astype(np.uint8),
                                      np.broadcast_to(np.arange(256, dtype=np.uint8), (80, 256)))
        self.assertTrue(json.loads(text)['frames'][0]['mask_8bit_roundtrip_corrected'])
        mask = torch.full((1, 80, 256), 1-.512345)
        output, text = NODE().process(rgb, mask=mask)
        self.assertTrue(torch.equal(output[..., 3], 1-mask))
        self.assertFalse(json.loads(text)['frames'][0]['mask_8bit_roundtrip_corrected'])

    def test_noncontiguous_float_input(self):
        x = rgba_fixture().transpose(1, 2); x[..., :3] += .17/255
        y, text = NODE().process(x, auto_scale=True, output_mode=NODES.FLOAT)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.equal(y[..., 3], x[..., 3]))
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(json.loads(text)['frames'][0]['input_had_subcode_precision'])
