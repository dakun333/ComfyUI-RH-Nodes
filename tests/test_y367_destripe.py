"""Y367 contracts and real frozen-kernel CPU regressions."""
import importlib
import json
import unittest
from unittest.mock import patch

import cv2
import numba
import numpy as np
import torch

from test_nodes import load_package

PACKAGE = load_package()
NODES = importlib.import_module(PACKAGE.__name__ + '.y367_destripe.guarded_nodes')
ENGINE = importlib.import_module(PACKAGE.__name__ + '.y367_destripe.guarded_engine')


def fixture():
    y, x = np.mgrid[:160, :192]
    direction = np.array([.3, -.6, .8]); direction /= np.linalg.norm(direction)
    rgb = np.rint(150 + 3.5*np.cos(2*np.pi*48*x/192)[:, :, None]*direction).astype(np.float32)/255
    return torch.from_numpy(rgb)[None]


class Y367Tests(unittest.TestCase):
    def test_defaults_outputs_and_fingerprint(self):
        for node_id in ('Y367CF1Destripe', 'Y367CE4Destripe'):
            cls = PACKAGE.NODE_CLASS_MAPPINGS[node_id]
            self.assertEqual(cls.CATEGORY, 'RH-Nodes/restore')
            self.assertEqual(cls.RETURN_TYPES, ('IMAGE', 'IMAGE', 'MASK', 'STRING'))
            self.assertEqual(cls.INPUT_TYPES()['required']['orientation'][1]['default'], 'both')
            self.assertEqual(cls.INPUT_TYPES()['required']['output_mode'][1]['default'], 'actual8')
            self.assertEqual(len(cls.IS_CHANGED(fixture())), 64)
        self.assertNotIn('Y367FastDestripe', PACKAGE.NODE_CLASS_MAPPINGS)

    def test_real_kernels_preserve_alpha_threads_and_input(self):
        x = fixture(); alpha = torch.ones(x.shape[:3]); alpha[:, :8] = .4
        rgba = torch.cat([x, alpha[..., None]], -1)
        before = rgba.clone(); cv_threads = cv2.getNumThreads(); nb_threads = numba.get_num_threads()
        for node_id in ('Y367CF1Destripe', 'Y367CE4Destripe'):
            output, diagnostic, mask, text = PACKAGE.NODE_CLASS_MAPPINGS[node_id]().process(rgba, orientation='vertical')
            self.assertTrue(torch.equal(output[..., 3], alpha))
            self.assertTrue(torch.equal(output[:, :8], rgba[:, :8]))
            self.assertTrue(torch.equal(mask, 1-alpha))
            self.assertEqual(diagnostic.shape, x.shape)
            self.assertTrue(torch.isfinite(output).all())
            row = json.loads(text)['frames'][0]
            self.assertEqual(row['logical_refreshes_per_axis'], 5)
            self.assertFalse(row['approximate_early_stop'])
            self.assertLessEqual(row['max_actual8_change'], 8)
        self.assertTrue(torch.equal(rgba, before))
        self.assertEqual(cv2.getNumThreads(), cv_threads)
        self.assertEqual(numba.get_num_threads(), nb_threads)

    def test_mask_broadcast_and_diagnostic_actual_output(self):
        x = fixture(); batch = torch.cat([x, x])
        mask = torch.zeros(1, *x.shape[1:3]); mask[:, :8] = .6
        def bypass(rgb, alpha, orientation, variant):
            target = rgb.copy(); target[alpha >= .95] -= 1
            return target, target.astype(np.uint8), {}
        with patch.object(NODES, 'run_guarded_frame', side_effect=bypass):
            out, diagnostic, returned, text = NODES.Y367CF1Destripe().process(batch, mask=mask)
        self.assertEqual(out.shape, (2, 160, 192, 4))
        self.assertTrue(torch.equal(returned[0], returned[1]))
        self.assertTrue(torch.equal(out[0], out[1]))
        self.assertTrue(torch.all(diagnostic[:, :8] == 128/255))
        self.assertTrue(torch.all(diagnostic[:, 8:] == 144/255))

    def test_invalid_float_geometry_and_conflicting_alpha(self):
        for kwargs in ({'image': fixture()+.17/255}, {'image': torch.zeros(1, 64, 64, 3)},
                       {'image': fixture(), 'orientation': 'diagonal'},
                       {'image': fixture(), 'output_mode': '8bit'},
                       {'image': fixture(), 'mask': torch.ones(1, 20, 20)}):
            with self.assertRaises(ValueError): NODES.Y367CF1Destripe().process(**kwargs)
        x = torch.cat([fixture(), torch.ones(1, 160, 192, 1)], -1)
        with self.assertRaises(ValueError):
            NODES.Y367CF1Destripe().process(x, mask=torch.ones(1, 160, 192))

    def test_exception_restores_cpu_budget(self):
        previous_cv = cv2.getNumThreads(); previous_nb = numba.get_num_threads()
        previous_hot = ENGINE.envelope._hot._THREAD_COUNT
        with patch.object(ENGINE.contour_edge, 'process', side_effect=RuntimeError('cancelled')):
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                ENGINE.run_guarded_frame((fixture()[0].numpy()*255).round(), np.ones((160, 192), np.float32), 'vertical', variant='CF1')
        self.assertEqual(cv2.getNumThreads(), previous_cv)
        self.assertEqual(numba.get_num_threads(), previous_nb)
        self.assertEqual(ENGINE.envelope._hot._THREAD_COUNT, previous_hot)
