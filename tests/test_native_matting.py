"""Source-only checkpoint contracts and VRMBG temporal/color/mapped-storage checks."""
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def module_entries(entries):
    """Restore only substituted modules, preserving unrelated lazy torch imports."""
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in entries}
    sys.modules.update(entries)
    try:
        yield
    finally:
        for name, old in previous.items():
            if old is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def package_context(name, directory):
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / directory)]
    sys.modules[name] = package
    return package


package_context('rh_native_contract_tests', 'corridorkey2_native')
CK = importlib.import_module('rh_native_contract_tests.checkpoint_io')
ALPHA = importlib.import_module('rh_native_contract_tests.alpha')
HOST = importlib.import_module('rh_native_contract_tests.host_dependencies')


class NativeCheckpointTests(unittest.TestCase):
    def checkpoint(self, path):
        values = {key: torch.ones(1) for key in ('stage1.weight', 'stage2.weight',
                  'foundations.moge.weight', 'foundations.cradio.weight', 'foundations.rvm.weight')}
        for name in ('moge', 'cradio', 'rvm'):
            for stat in ('mean', 'std'):
                values[f'foundations.{name}_{stat}'] = torch.ones(1)
        metadata = {'format': CK.BUNDLE_FORMAT, 'format_version': '1', 'tensor_count': str(len(values)),
                    'model_config': '{}', 'foundation_config': json.dumps({'moge': {}, 'cradio': {}, 'rvm': {}})}
        save_file(values, str(path), metadata=metadata)
        return values, metadata

    def test_existing_combined_file_never_reads_or_downloads_original_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'combined.safetensors'
            self.checkpoint(file)
            with patch.object(CK, 'ensure_assets', side_effect=AssertionError('original assets accessed')), \
                 patch.object(CK, 'convert_checkpoint', side_effect=AssertionError('unnecessary conversion')):
                self.assertEqual(CK.ensure_checkpoint(file), [])

    def test_wrong_checkpoint_and_missing_foundation_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'combined.safetensors'
            values, metadata = self.checkpoint(file)
            metadata['format'] = 'sd'
            save_file(values, str(file), metadata=metadata)
            with self.assertRaisesRegex(ValueError, 'SD/Flux'): CK.read_checkpoint_metadata(file)
            metadata['format'] = CK.BUNDLE_FORMAT
            del values['foundations.rvm_std']
            metadata['tensor_count'] = str(len(values))
            save_file(values, str(file), metadata=metadata)
            with self.assertRaisesRegex(ValueError, 'rvm std'): CK.read_checkpoint_metadata(file)

    def test_missing_file_prepares_original_assets_before_conversion(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(CK, 'ensure_assets', return_value=['original']) as assets, patch.object(CK, 'convert_checkpoint') as convert:
            file = Path(temp) / 'combined.safetensors'
            self.assertEqual(CK.ensure_checkpoint(file), ['original'])
            assets.assert_called_once()
            self.assertEqual(convert.call_args.args[1], file.resolve())

    def test_cli_default_handles_nested_rh_module(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'folder_paths.py').touch()
            module = root / 'custom_nodes/RH/corridorkey2_native'; module.mkdir(parents=True)
            with patch.object(CK, 'ROOT', module):
                self.assertEqual(CK.default_checkpoint_path(), root / 'models/checkpoints' / CK.BUNDLE_FILENAME)

    def test_alpha_default_identity_soft_shrink_and_order(self):
        alpha = np.full((21, 23), .6, np.float32); alpha[8:13, 9:14] = .2
        self.assertIs(ALPHA.adjust_alpha(alpha), alpha)
        expected = ALPHA.adjust_alpha(ALPHA.adjust_alpha(alpha, 1), 0, .7)
        np.testing.assert_array_equal(ALPHA.adjust_alpha(alpha, 1, .7), expected)
        np.testing.assert_array_equal(alpha[8:13, 9:14], np.full((5, 5), .2, np.float32))

    def test_existing_utils3d_is_never_replaced(self):
        before = list(sys.path); existing = object()
        with patch.object(HOST.importlib.util, 'find_spec', return_value=object()), patch.object(HOST.importlib, 'import_module', return_value=existing):
            self.assertIs(HOST.load_utils3d(), existing)
        self.assertEqual(sys.path, before)
        with patch.object(HOST.importlib.util, 'find_spec', return_value=object()), patch.object(HOST.importlib, 'import_module', side_effect=ImportError('host API')):
            with self.assertRaisesRegex(RuntimeError, 'host API'): HOST.load_utils3d()
        self.assertEqual(sys.path, before)


class VRMBGWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        paths = types.ModuleType('folder_paths'); paths.models_dir = str(self.work / 'models')
        paths.add_model_folder_path = Mock(); paths.get_folder_paths = Mock(return_value=[paths.models_dir])
        paths.get_temp_directory = lambda: str(self.work)
        mm = types.ModuleType('comfy.model_management')
        for name in ('load_models_gpu', 'unload_model_and_clones', 'soft_empty_cache', 'throw_exception_if_processing_interrupted'):
            setattr(mm, name, Mock())
        mm.get_torch_device = lambda: torch.device('cpu')
        utils = types.ModuleType('comfy.utils'); utils.ProgressBar = Mock()
        patcher = types.ModuleType('comfy.model_patcher'); patcher.ModelPatcher = Mock()
        comfy = types.ModuleType('comfy'); comfy.model_management = mm
        entries = {'folder_paths': paths, 'comfy': comfy, 'comfy.model_management': mm,
                   'comfy.utils': utils, 'comfy.model_patcher': patcher}
        self.mm = mm
        self.context = module_entries(entries); self.context.__enter__(); self.addCleanup(self.context.__exit__, None, None, None)
        self.name = 'rh_vrmbg_wrapper_tests'
        package_context(self.name, 'vrmbg3_v6')
        self.nodes = importlib.import_module(self.name + '.nodes')
        self.color = importlib.import_module(self.name + '.foreground_color')
        self.memory = importlib.import_module(self.name + '.batch_memory')
        # These modules hold references to their initial host API objects.
        self.nodes.mm = mm; self.nodes.folder_paths = paths; self.memory.folder_paths = paths

    def model(self):
        recorded = []
        class Network(torch.nn.Module):
            def forward(self, paired):
                recorded.append(paired.clone())
                return [paired[:, :1] * .25 + paired[:, 3:4] * .2]
        network = Network()
        model = types.SimpleNamespace(lock=threading.RLock(), ensure_loaded=lambda: types.SimpleNamespace(model=network), model_info={})
        return model, recorded

    def test_temporal_history_resets_and_alpha_cleanup_does_not_feed_back(self):
        image = torch.rand(2, 64, 64, 3, generator=torch.Generator().manual_seed(16))
        original = image.clone(); model, raw_inputs = self.model()
        first = self.nodes.VRMBG3V6Matting().infer(model, image, precision='fp32')
        model, cleanup_inputs = self.model()
        cleaned = self.nodes.VRMBG3V6Matting().infer(model, image, precision='fp32', alpha_shrink=1, alpha_feather=.8)
        self.assertTrue(torch.equal(raw_inputs[0], cleanup_inputs[0]))
        self.assertTrue(torch.equal(raw_inputs[1], cleanup_inputs[1]))
        self.assertEqual(int(torch.count_nonzero(raw_inputs[0][:, 3:])), 0)
        self.assertGreater(int(torch.count_nonzero(raw_inputs[1][:, 3:])), 0)
        self.assertTrue(torch.equal(cleaned[0][..., :3], image))
        self.assertTrue(torch.equal(cleaned[0][..., 3], cleaned[1]))
        self.assertFalse(torch.equal(first[1], cleaned[1]))
        self.assertTrue(torch.equal(image, original))
        model, fresh = self.model()
        self.nodes.VRMBG3V6Matting().infer(model, image, precision='fp32', temporal=False)
        self.assertEqual(int(torch.count_nonzero(fresh[1][:, 3:])), 0)

    def test_multilevel_preserves_alpha_and_exact_opaque_hidden_rgb_rules(self):
        rgb = np.full((16, 16, 3), .5, np.float32)
        alpha = np.ones((16, 16), np.float32); alpha[:4] = 0; alpha[4:8] = .5
        before = alpha.copy()
        result = self.color.recover_foreground_rgb(rgb, alpha)
        np.testing.assert_array_equal(alpha, before)
        np.testing.assert_array_equal(result[alpha == 1], rgb[alpha == 1])
        self.assertEqual(int(np.count_nonzero(result[alpha == 0])), 0)
        self.assertTrue(np.isfinite(result).all())

    def test_mapped_storage_survives_tensor_view_then_reclaims_file(self):
        tensor = self.memory.mapped_tensor((2, 64, 64, 4)); tensor.fill_(.3)
        files = list((self.work / 'vrmbg3_v6_batches').glob('*.f32'))
        self.assertEqual(len(files), 1)
        view = tensor[..., 3]; del tensor; gc.collect()
        self.assertTrue(files[0].exists()); self.assertTrue(torch.all(view == .3))
        del view; gc.collect()
        self.assertFalse(files[0].exists())

    def test_pixel_budget_and_missing_weights_do_not_fallback(self):
        plan = self.nodes.resolution_plan(1920, 1080)
        self.assertLessEqual(plan['width'] * plan['height'], 1048576)
        with self.assertRaisesRegex(FileNotFoundError, 'not distributed'):
            self.nodes.VRMBG3V6Loader().load()
        self.assertEqual(self.nodes.VRMBG3V6Loader.INPUT_TYPES()['required']['model_name'][1]['default'], self.nodes.DEFAULT_MODEL_NAME)

    def test_model_ids_and_foreground_color_defaults(self):
        self.assertEqual(len(self.nodes.NODE_CLASS_MAPPINGS), 5)
        for cls in self.nodes.NODE_CLASS_MAPPINGS.values():
            self.assertEqual(cls.CATEGORY, 'RH-Nodes/matting')
        settings = self.nodes.VRMBG3V6Matting.INPUT_TYPES()['optional']
        self.assertFalse(settings['enable_alpha_unmult'][1]['default'])
        self.assertEqual(settings['color_recovery_method'][1]['default'], 'multilevel')

    def test_pinned_bria_architecture_byte_hashes(self):
        folder = ROOT / 'vrmbg3_v6/vendor/bria_vrmbg3'
        for name, expected in self.nodes.SOURCE_SHA.items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), expected, name)
