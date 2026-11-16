"""Matting bridge contracts, frozen light math, and isolated worker outputs."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from test_nodes import load_package


class MattingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = load_package()
        cls.ck = importlib.import_module(cls.package.__name__ + '.corridorkey2.nodes')
        cls.light = cls.package.NODE_CLASS_MAPPINGS['PureLightMatting']

    def setUp(self):
        self.mm = types.ModuleType('comfy.model_management')
        for name in ('throw_exception_if_processing_interrupted', 'unload_all_models', 'soft_empty_cache'):
            setattr(self.mm, name, Mock())
        self.mm.get_torch_device = Mock(return_value=torch.device('cpu'))
        utils = types.ModuleType('comfy.utils')
        utils.ProgressBar = Mock()
        comfy = types.ModuleType('comfy'); comfy.model_management = self.mm; comfy.utils = utils
        graph = types.ModuleType('comfy_execution.graph_utils')
        class Blocker:
            def __init__(self, message): self.message = message
        graph.ExecutionBlocker = Blocker
        # patch.dict(sys.modules) restores the entire dictionary on exit, also
        # removing unrelated modules lazily imported by torch.testing. Their
        # C++ operator registrations survive, so reimporting them fails.
        missing = object()
        for name, module in {'comfy': comfy, 'comfy.model_management': self.mm,
                             'comfy.utils': utils, 'comfy_execution.graph_utils': graph}.items():
            previous = sys.modules.get(name, missing)
            def restore(name=name, previous=previous):
                if previous is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
            self.addCleanup(restore)
            sys.modules[name] = module

    def test_schema_defaults_and_registration(self):
        for node_id in ('CorridorKey2Matting', 'PureLightMatting'):
            cls = self.package.NODE_CLASS_MAPPINGS[node_id]
            self.assertEqual(cls.CATEGORY, 'RH-Nodes/matting')
            self.assertEqual(cls.RETURN_TYPES, ('IMAGE', 'MASK', 'IMAGE', 'IMAGE', 'STRING', 'VIDEO'))
            self.assertEqual(cls.RETURN_NAMES, ('foreground_rgb', 'alpha', 'rgba', 'checker_preview', 'report', 'video'))
            self.assertEqual(cls.INPUT_TYPES()['optional']['output_mode'][1]['default'], 'images')
        self.assertEqual(self.ck.CorridorKey2Matting.INPUT_TYPES()['required']['despill_strength'][1]['default'], 0)

    def test_light_straight_reconstruction_and_alpha(self):
        image = torch.tensor([[[[.04, .2, .08, .5], [0, 0, 0, 1]]]])
        fg, alpha, rgba, preview, report, inactive = self.light().matte(image, auto_background=False, processing_device='cpu')
        torch.testing.assert_close(fg[0, 0, 0], torch.tensor([.2, 1, .4]))
        torch.testing.assert_close(alpha, torch.tensor([[[.1, 0]]]))
        self.assertTrue(torch.equal(rgba[..., 3], alpha))
        self.assertEqual(preview.shape, (1, 1, 2, 3))
        self.assertFalse(json.loads(report)['weights_required'])
        self.mm.unload_all_models.assert_not_called()

    def test_light_batch_nonmutation_dither_and_no_global_rng_changes(self):
        x = torch.rand(2, 24, 32, 3, generator=torch.Generator().manual_seed(12))
        before = x.clone(); rng = torch.get_rng_state().clone()
        batch = self.light().matte(x, dither=.4, processing_device='cpu')
        for i in range(2):
            one = self.light().matte(x[i:i+1], dither=.4, processing_device='cpu')
            for j in range(4): self.assertTrue(torch.equal(batch[j][i:i+1], one[j]))
        self.assertTrue(torch.equal(x, before)); self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_light_black_and_invalid_inputs(self):
        result = self.light().matte(torch.zeros(2, 8, 8, 3), processing_device='cpu')
        self.assertEqual(int(torch.count_nonzero(result[2])), 0)
        for kwargs in ({}, {'images': torch.zeros(1, 4, 4, 3), 'video': object()},
                       {'images': torch.zeros(1, 4, 4, 3), 'output_mode': 'both'},
                       {'images': torch.zeros(1, 4, 4, 3), 'black_threshold': 256}):
            with self.assertRaises((ValueError, TypeError)): self.light().matte(**kwargs)

    def test_corridor_worker_bridge_preserves_shape_hint_and_color(self):
        images = torch.full((2, 4, 6, 3), .5); original = images.clone()
        hint = torch.full((1, 4, 6), .25)
        seen = {}
        def worker(command, **kwargs):
            directory = Path(command[-1]).parent
            seen['command'] = command
            seen['hint'] = np.load(directory / 'hints.npy')
            seen['job'] = json.loads((directory / 'job.json').read_text())
            linear = np.full((2, 4, 6, 4), .18, np.float32); linear[..., 3] = .4
            np.save(directory / 'linear_rgba.npy', linear)
            (directory / 'report.json').write_text(json.dumps({'frames': 2}))
            return types.SimpleNamespace(poll=lambda: 0, returncode=0)
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.ck, 'ROOT', Path(tmp)), patch.object(self.ck.subprocess, 'Popen', side_effect=worker):
            outputs = self.ck.CorridorKey2Matting().matte(images, hint_mask=hint, invert_hint_mask=True)
        self.assertEqual(len(outputs), 6)
        self.assertEqual(outputs[2].shape, (2, 4, 6, 4))
        torch.testing.assert_close(outputs[1], torch.full((2, 4, 6), .4))
        expected = float(importlib.import_module(self.package.__name__ + '.corridorkey2.color').linear_to_srgb(np.array(.18, np.float32)))
        torch.testing.assert_close(outputs[0], torch.full((2, 4, 6, 3), expected))
        np.testing.assert_array_equal(seen['hint'], .75)
        self.assertIn('-I', seen['command'])
        self.assertEqual(seen['job']['settings']['despill_strength'], 0)
        self.assertTrue(torch.equal(images, original))

    def test_corridor_invalid_hints_before_worker_launch(self):
        with patch.object(self.ck.subprocess, 'Popen') as worker:
            with self.assertRaises(ValueError):
                self.ck.CorridorKey2Matting().matte(torch.zeros(2, 8, 8, 3), hint_mask=torch.zeros(3, 8, 8))
            worker.assert_not_called()

    def test_import_does_not_start_installer_or_load_models(self):
        with patch('subprocess.Popen', side_effect=AssertionError('worker on import')), patch('subprocess.run', side_effect=AssertionError('installer on import')):
            package = load_package()
        self.assertIn('CorridorKey2Matting', package.NODE_CLASS_MAPPINGS)
        self.assertIn('PureLightMatting', package.NODE_CLASS_MAPPINGS)

    def test_bootstrap_ready_skips_pip_and_missing_uses_isolated_worker(self):
        import os
        bootstrap = importlib.import_module(self.package.__name__ + '.corridorkey2.dependency_bootstrap')
        with patch.dict(os.environ, {'CORRIDORKEY2_AUTO_INSTALL_DEPS': '1'}), patch.object(bootstrap, 'quick_private_version_check', return_value=True), patch.object(bootstrap.subprocess, 'run') as run:
            bootstrap.ensure_worker_dependencies()
            run.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'CORRIDORKEY2_AUTO_INSTALL_DEPS': '1'}), patch.object(bootstrap.subprocess, 'run') as run:
            bootstrap.ensure_worker_dependencies(tmp)
            command = run.call_args.args[0]
            self.assertEqual(command[:3], [sys.executable, '-I', '-u'])
            self.assertIn('--if-needed', command)
            self.assertNotIn('PYTHONPATH', run.call_args.kwargs['env'])


class MattingModuleIsolationTests(unittest.TestCase):
    def test_cleanup_preserves_lazy_imports_and_restores_only_comfy_stubs(self):
        name = '_rh_matting_lazy_import_regression'
        original_comfy = sys.modules.get('comfy')
        had_comfy = 'comfy' in sys.modules
        fixture = MattingTests('test_schema_defaults_and_registration')
        fixture.setUp()
        lazy_module = types.ModuleType(name)
        try:
            sys.modules[name] = lazy_module
            # Exercise the actual lazy DTensor imports used in failing CI.
            torch.testing.assert_close(torch.ones(1), torch.ones(1))
            fixture.doCleanups()
            self.assertIs(sys.modules.get(name), lazy_module)
            self.assertEqual('comfy' in sys.modules, had_comfy)
            if had_comfy:
                self.assertIs(sys.modules['comfy'], original_comfy)
            torch.testing.assert_close(torch.ones(1), torch.ones(1))
        finally:
            fixture.doCleanups()
            sys.modules.pop(name, None)
