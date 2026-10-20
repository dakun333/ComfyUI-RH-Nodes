import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / 'seedream_color_reuse'
spec = importlib.util.spec_from_file_location(
    'seedream_v64_test_plugin', ROOT / '__init__.py',
    submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from seedream_v64_test_plugin.nodes import (
    NODE_CLASS_MAPPINGS, SeedreamLayerColorReuseV64,
    SeedreamV64LoadAlignedLayers)


def composite_top_to_bottom(stack):
    rgb = np.zeros_like(stack[0, ..., :3])
    alpha = np.zeros_like(stack[0, ..., 3:4])
    for image in stack[::-1]:
        a = image[..., 3:4]
        rgb = image[..., :3] * a + rgb * (1 - a)
        alpha = a + alpha * (1 - a)
    return np.concatenate((rgb, alpha), axis=-1).astype(np.float32)


def scene():
    height, width = 64, 80
    background = np.zeros((height, width, 4), np.float32)
    background[..., :3] = [.10, .16, .24]
    background[..., 3] = 1
    receiver = np.zeros_like(background)
    receiver[6:58, 7:73, :3] = [.34, .18, .48]
    receiver[6:58, 7:73, 3] = 1
    top = np.zeros_like(background)
    top[20:46, 24:59, :3] = [.88, .66, .14]
    top[20:46, 24:59, 3] = 1
    stack = np.stack((top, receiver, background))
    reference = composite_top_to_bottom(stack)
    reference[..., :3] = np.clip(reference[..., :3] + [.035, .02, .01], 0, 1)
    return torch.from_numpy(reference[None]), torch.from_numpy(stack)


class V64NodeTests(unittest.TestCase):
    def test_registered_in_separate_plugin(self):
        self.assertEqual(set(NODE_CLASS_MAPPINGS), {'SeedreamLayerColorReuseV64', 'SeedreamV64LoadAlignedLayers', 'SeedreamLayerColorReuseV64Fast'})

    def test_calibrated_contour_patch_defaults_and_source_hashes(self):
        import hashlib
        from seedream_v64_test_plugin.engine import Config
        config = Config()
        self.assertEqual(config.frequency_calibrated_baseline, 1)
        self.assertEqual(config.ownership_local_seams, 1)
        self.assertEqual(config.protect_contact_shadow_tails, 1)
        self.assertEqual(config.frequency_soft_protection, 1)
        self.assertEqual(config.ownership_flat_receivers, 1)
        self.assertEqual(config.reuse_boundary_continuity, 1)
        self.assertEqual(config.chromatic_contour_ink, 1)
        self.assertEqual(Config(chromatic_contour_ink=0).chromatic_contour_ink, 0)
        self.assertEqual(Config(reuse_boundary_continuity=0).reuse_boundary_continuity, 0)
        self.assertEqual(config.ownership_local_max_texture, 8.)
        self.assertEqual(Config(ownership_local_seams=0).ownership_local_seams, 0)
        provenance = json.loads((ROOT / 'core_provenance.json').read_text(encoding='utf8'))
        self.assertEqual(provenance['algorithm_version'], '6.4.5-chromatic-contour-continuity')
        for name, digest in provenance['files_sha256'].items():
            # Compare normalized text snapshots; Git normalizes Windows CRLF.
            raw = (ROOT / 'engine' / name).read_bytes()
            normalized = raw.replace(b'\r\n', b'\n').rstrip() + b'\n'
            self.assertEqual(hashlib.sha256(normalized).hexdigest(), digest)

    def test_output_contract_order_and_no_mutation(self):
        reference, images = scene()
        before_reference, before_images = reference.clone(), images.clone()
        output = SeedreamLayerColorReuseV64().correct_and_reuse(
            reference, images, normalize_alpha_levels=False)
        self.assertEqual(len(output), 8)
        self.assertEqual(output[0].shape, (3, 64, 80, 4))
        self.assertEqual(output[1].shape, (1, 64, 80, 3))
        self.assertEqual(output[2].shape, (1, 64, 80))
        self.assertEqual(output[3].shape, (2, 64, 80, 3))
        self.assertEqual(output[7].shape, (1, 64, 80, 4))
        self.assertTrue(torch.equal(reference, before_reference))
        self.assertTrue(torch.equal(images, before_images))
        self.assertTrue(torch.equal(output[0][..., 3], images[..., 3]))
        self.assertGreater(float(output[2].sum()), 0)
        self.assertTrue(all(np.isfinite(value) for value in output[4:7]))
        self.assertAlmostEqual(output[6], output[4] - output[5], places=7)

    def test_composite_port_is_appended_and_matches_corrected_layers(self):
        reference, images = scene()
        node = SeedreamLayerColorReuseV64()
        self.assertEqual(node.RETURN_TYPES,
                         ('IMAGE', 'IMAGE', 'MASK', 'IMAGE', 'FLOAT', 'FLOAT', 'FLOAT', 'IMAGE'))
        self.assertEqual(node.RETURN_NAMES[:7],
                         ('images', 'reuse_image', 'reuse_mask', 'difference_heatmaps',
                          'mae_before', 'mae_after', 'mae_reduction'))
        self.assertEqual(node.RETURN_NAMES[7], '拼接结果')
        output = node.correct_and_reuse(reference, images, normalize_alpha_levels=False)
        merged = output[7]
        self.assertEqual(merged.shape, (1, 64, 80, 4))
        self.assertEqual(merged.dtype, torch.float32)
        expected = composite_top_to_bottom(output[0].numpy())
        np.testing.assert_array_equal(merged[0].numpy(), expected)
        self.assertTrue(torch.all(merged[..., 3] == 1))
        mask = output[2][0].numpy() > .5
        actual = np.rint(merged[0, ..., :3].numpy() * 255).astype(np.uint8)
        source = np.rint(reference[0, ..., :3].numpy() * 255).astype(np.uint8)
        np.testing.assert_array_equal(actual[mask], source[mask])
        # The new output has no green/red diagnostic overlay.
        self.assertFalse(torch.equal(merged[..., :3], output[1]))

    def test_composite_keeps_soft_alpha_and_ignores_invisible_rgb(self):
        reference, images = scene()
        # Retain an opaque region for metrics, but expose soft-alpha/transparent
        # canvas regions so a hard-coded opaque or premultiplied output fails.
        images[2, :4, :, 3] = 0
        images[2, :4, :, :3] = torch.tensor([1., 0., 1.])
        images[1, :2, 10:20, :3] = torch.tensor([.2, .8, .4])
        images[1, :2, 10:20, 3] = .5
        images[0, :2, 15:20, :3] = torch.tensor([.8, .2, .6])
        images[0, :2, 15:20, 3] = .25
        output = SeedreamLayerColorReuseV64().correct_and_reuse(
            reference, images, normalize_alpha_levels=False)
        premultiplied = composite_top_to_bottom(output[0].numpy())
        expected_rgb = np.divide(premultiplied[..., :3], premultiplied[..., 3:4],
                                 out=np.zeros_like(premultiplied[..., :3]),
                                 where=premultiplied[..., 3:4] > 0)
        merged = output[7][0].numpy()
        np.testing.assert_array_equal(merged[..., 3], premultiplied[..., 3])
        np.testing.assert_array_equal(merged[..., :3], expected_rgb)
        self.assertTrue((merged[2:4] == 0).all())
        self.assertAlmostEqual(float(merged[0, 12, 3]), 128 / 255, places=6)
        self.assertAlmostEqual(float(merged[0, 17, 3]),
                               64 / 255 + (128 / 255) * (1 - 64 / 255), places=6)
        self.assertGreater(float(merged[0, 12, 1]), float(premultiplied[0, 12, 1]))

    def test_composite_output_does_not_modify_existing_outputs(self):
        from unittest.mock import patch
        reference, images = scene()
        node = SeedreamLayerColorReuseV64()
        # Replacing only the new output must leave all existing output data
        # unchanged; this test has no dependency on a local backup directory.
        with patch('seedream_v64_test_plugin.nodes._composite_rgba',
                   return_value=np.zeros((1, 64, 80, 4), np.float32)):
            without_merge = node.correct_and_reuse(reference, images)
        with_merge = node.correct_and_reuse(reference, images)
        self.assertEqual(len(with_merge), 8)
        for index in range(4):
            self.assertTrue(torch.equal(without_merge[index], with_merge[index]))
        np.testing.assert_array_equal(without_merge[4:7], with_merge[4:7])
        self.assertFalse(torch.equal(without_merge[7], with_merge[7]))

    def test_bottom_to_top_option_preserves_caller_order(self):
        reference, top_to_bottom = scene()
        node = SeedreamLayerColorReuseV64()
        a = node.correct_and_reuse(
            reference, top_to_bottom, normalize_alpha_levels=False)
        bottom_to_top = top_to_bottom.flip(0)
        b = node.correct_and_reuse(
            reference, bottom_to_top, layer_order='bottom_to_top',
            normalize_alpha_levels=False)
        self.assertTrue(torch.equal(a[0], b[0].flip(0)))
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertTrue(torch.equal(a[2], b[2]))
        self.assertTrue(torch.equal(a[3], b[3]))
        np.testing.assert_allclose(a[4:7], b[4:7], rtol=0, atol=0)
        self.assertTrue(torch.equal(a[7], b[7]))

    def test_rgb_plus_masks_matches_rgba(self):
        reference, images = scene()
        node = SeedreamLayerColorReuseV64()
        rgba = node.correct_and_reuse(
            reference, images, normalize_alpha_levels=False)
        rgb_mask = node.correct_and_reuse(
            reference, images[..., :3], image_masks=1 - images[..., 3],
            normalize_alpha_levels=False)
        for index in range(4):
            self.assertTrue(torch.equal(rgba[index], rgb_mask[index]))
        np.testing.assert_allclose(rgba[4:7], rgb_mask[4:7], rtol=0, atol=0)
        self.assertTrue(torch.equal(rgba[7], rgb_mask[7]))

    def test_reference_is_resized_to_layer_canvas(self):
        reference, images = scene()
        large = reference.repeat_interleave(2, 1).repeat_interleave(2, 2)
        output = SeedreamLayerColorReuseV64().correct_and_reuse(
            large, images, normalize_alpha_levels=False)
        self.assertEqual(output[0].shape, images.shape)
        self.assertEqual(output[3].shape, (2, 64, 80, 3))

    def test_alpha_levels_are_explicit_and_disableable(self):
        reference, images = scene()
        images[0, 5, 5, :3] = .5
        images[0, 5, 5, 3] = 2 / 255
        node = SeedreamLayerColorReuseV64()
        normalized = node.correct_and_reuse(reference, images)
        preserved = node.correct_and_reuse(
            reference, images, normalize_alpha_levels=False)
        self.assertEqual(float(normalized[0][0, 5, 5, 3]), 0)
        self.assertAlmostEqual(float(preserved[0][0, 5, 5, 3]), 2 / 255, places=7)


    def test_aligned_layer_loader_defaults_to_top_first(self):
        import tempfile
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            for index in (1, 2, 10):
                Image.new('RGBA', (12, 8), (index, 20, 30, 128)).save(
                    Path(folder) / f'layer_{index}.png')
            images, masks, order = SeedreamV64LoadAlignedLayers().load(folder)
            names = [Path(item).name for item in __import__('json').loads(order)]
            self.assertEqual(names, ['layer_10.png', 'layer_2.png', 'layer_1.png'])
            self.assertEqual(images.shape, (3, 8, 12, 4))
            self.assertTrue(torch.allclose(masks, 1 - images[..., 3]))

    def test_invalid_reference_and_missing_alpha_are_rejected(self):
        reference, images = scene()
        node = SeedreamLayerColorReuseV64()
        with self.assertRaisesRegex(ValueError, 'Exactly one reference'):
            node.correct_and_reuse(reference.repeat(2, 1, 1, 1), images)
        with self.assertRaisesRegex(ValueError, 'no alpha'):
            node.correct_and_reuse(reference, images[..., :3])


    def test_example_workflow_and_api_prompt_are_wired(self):
        workflow = json.loads((ROOT / 'examples' / 'workflow.json').read_text(encoding='utf-8'))
        api = json.loads((ROOT / 'examples' / 'api_prompt.json').read_text(encoding='utf-8'))
        node_ids = {node['id'] for node in workflow['nodes']}
        link_ids = {link[0] for link in workflow['links']}
        self.assertEqual(workflow['last_node_id'], max(node_ids))
        self.assertEqual(workflow['last_link_id'], max(link_ids))
        self.assertEqual(len(node_ids), len(workflow['nodes']))
        self.assertEqual(len(link_ids), len(workflow['links']))
        self.assertEqual(api['3']['class_type'], 'SeedreamLayerColorReuseV64')
        self.assertEqual(api['3']['inputs']['reference'], ['1', 0])
        self.assertEqual(api['3']['inputs']['reference_mask'], ['1', 1])
        self.assertEqual(api['3']['inputs']['images'], ['2', 0])
        self.assertEqual(api['3']['inputs']['layer_order'], 'top_to_bottom')
        self.assertEqual(api['2']['inputs']['file_order'], 'descending')
        self.assertEqual(api['4']['inputs']['images'], ['3', 0])
        self.assertEqual(api['5']['inputs']['images'], ['3', 1])
        self.assertEqual(api['6']['inputs']['mask'], ['3', 2])
        self.assertEqual(api['8']['inputs']['images'], ['3', 3])
        self.assertEqual(api['12']['inputs']['images'], ['3', 7])
        node = next(n for n in workflow['nodes'] if n['id'] == 3)
        self.assertEqual(len(node['outputs']), 8)
        self.assertEqual(node['outputs'][7]['name'], '拼接结果')
        self.assertIn([12, 3, 7, 12, 0, 'IMAGE'], workflow['links'])

if __name__ == '__main__':
    unittest.main()
