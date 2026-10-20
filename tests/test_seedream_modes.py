"""UI/API compatibility for default-off color-only and distinct fast nodes."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / 'seedream_color_reuse'
spec=importlib.util.spec_from_file_location('seedream_modes_test_plugin',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
pkg=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pkg;spec.loader.exec_module(pkg)
from seedream_modes_test_plugin import nodes


def scene():
    raw=np.zeros((3,80,96,4),np.float32)
    raw[2,:,:,:3]=[.18,.28,.38];raw[2,:,:,3]=1
    raw[1,8:72,10:86,:3]=[.4,.25,.5];raw[1,8:72,10:86,3]=1
    raw[1,8,10:86,3]=128/255
    raw[0,24:52,28:64,:3]=[.82,.65,.13];raw[0,24:52,28:64,3]=1
    ref=nodes._composite_rgba(raw,'top_to_bottom');ref[...,:3]=np.clip(ref[...,:3]+[.04,.02,.01],0,1)
    return torch.from_numpy(ref),torch.from_numpy(raw)


class ModesNodeTests(unittest.TestCase):
    def test_switch_is_optional_appended_default_off_and_outputs_unchanged(self):
        schema=nodes.SeedreamLayerColorReuseV64.INPUT_TYPES()
        self.assertEqual(list(schema['required']),['reference','images','layer_order','normalize_alpha_levels','heatmap_max','mask_convention'])
        self.assertEqual(list(schema['optional']),['image_masks','reference_mask','color_only'])
        self.assertFalse(schema['optional']['color_only'][1]['default'])
        self.assertEqual(nodes.SeedreamLayerColorReuseV64Fast.RETURN_NAMES,nodes.SeedreamLayerColorReuseV64.RETURN_NAMES)
        self.assertIs(nodes.NODE_CLASS_MAPPINGS['SeedreamLayerColorReuseV64Fast'],nodes.SeedreamLayerColorReuseV64Fast)

    def test_active_reuse_seam_shared_standard_fast_engine(self):
        from seedream_modes_test_plugin.engine import core
        from seedream_modes_test_plugin.engine.fast import _runtime
        yy, xx = np.indices((96,112))
        raw = np.zeros((2,96,112,4),np.float32)
        raw[1,...,:3] = 30/255;raw[1,...,3]=1
        raw[0,...,:3] = np.array([180,140,110],np.float32)/255;raw[0,...,3]=1
        ref = np.rint(raw[0]*255).astype(np.uint8)
        ref[xx < yy+5,:3] -= np.array([8,16,12],np.uint8)
        ref[(xx >= yy+5)&(xx < yy+8),:3] -= np.array([60,60,60],np.uint8)
        ref = torch.from_numpy(ref.astype(np.float32)[None]/255)
        images = torch.from_numpy(raw)
        identity = lambda image: (image.copy(),np.ones(image.shape[:2],np.float32))
        reports = []
        outputs = []
        for cls in (nodes.SeedreamLayerColorReuseV64,nodes.SeedreamLayerColorReuseV64Fast):
            node = cls();entry=node._run_engine
            def capture(*args,**kwargs):
                result=entry(*args,**kwargs);reports.append(result[1]);return result
            node._run_engine=capture
            with patch.object(core.ColorField,'apply',side_effect=identity), \
                 patch.object(_runtime.ColorField,'apply',side_effect=identity):
                outputs.append(node.correct_and_reuse(ref,images,normalize_alpha_levels=False))
        self.assertGreater(reports[0]['reuse_boundary_continuity']['changed_pixels'],10)
        self.assertEqual(reports[0],{k:v for k,v in reports[1].items() if not k.startswith('execution_')})
        for a,b in zip(*outputs):
            if isinstance(a,torch.Tensor): self.assertTrue(torch.equal(a,b))
            else:self.assertEqual(a,b)
        self.assertTrue(torch.equal(outputs[0][0][...,3],images[...,3]))
        exact=outputs[0][2][0]>.5
        self.assertTrue(torch.equal(outputs[0][7][0,...,:3][exact],ref[0,...,:3][exact]))

    def test_missing_false_and_list_false_match_all_eight_outputs(self):
        ref,images=scene();node=nodes.SeedreamLayerColorReuseV64()
        results=[node.correct_and_reuse(ref,images,**kw) for kw in [{},{'color_only':False},{'color_only':[False]}]]
        for a,b,c in zip(*results):
            if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b) and torch.equal(a,c))
            else:self.assertEqual(a,b);self.assertEqual(a,c)

    def test_only_switch_list_and_no_input_mutation(self):
        ref,images=scene();original=images.clone();original_ref=ref.clone()
        result=nodes.SeedreamLayerColorReuseV64().correct_and_reuse([ref],[images],color_only=[True],normalize_alpha_levels=[False])
        self.assertEqual(len(result),8);self.assertEqual(float(result[2].sum()),0)
        np.testing.assert_array_equal(result[0][...,3],images[...,3].numpy())
        self.assertTrue(torch.equal(original,images));self.assertTrue(torch.equal(original_ref,ref))
        self.assertTrue(torch.equal(result[7],torch.from_numpy(nodes._composite_rgba(result[0].numpy(),'top_to_bottom'))))

    def test_switch_invalid_values_not_silently_cast(self):
        ref,images=scene()
        for value in ['false',1,None,[True,False]]:
            with self.assertRaises(ValueError):nodes.SeedreamLayerColorReuseV64().correct_and_reuse(ref,images,color_only=value)

    def test_only_uses_no_reference_pixel_projection(self):
        ref,images=scene()
        with patch.object(nodes,'calibrate',wraps=nodes.calibrate) as engine:
            result=nodes.SeedreamLayerColorReuseV64().correct_and_reuse(ref,images,color_only=True)
        self.assertEqual(engine.call_args.args[2].color_only,1)
        self.assertFalse(result[2].any())

    def test_fast_masks_list_orders_transparent_canvas_and_only(self):
        ref,images=scene()
        for order in ['top_to_bottom','bottom_to_top']:
            for only in [False,True]:
                stack=images if order=='top_to_bottom' else torch.flip(images,[0])
                kwargs={'layer_order':[order],'normalize_alpha_levels':[False], 'color_only':[only],
                        'image_masks':[1-stack[...,3]],'reference_mask':[1-ref[...,3]]}
                standard=nodes.SeedreamLayerColorReuseV64().correct_and_reuse([ref],[stack[0:1],stack[1:]],**kwargs)
                fast=nodes.SeedreamLayerColorReuseV64Fast().correct_and_reuse([ref],[stack[0:1],stack[1:]],**kwargs)
                for a,b in zip(standard,fast):
                    if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
                    else:self.assertEqual(a,b)
        # Transparent output keeps zero undefined straight RGB and exact alpha.
        stack=images.clone();stack[:,:,0:4,3]=0
        for cls in [nodes.SeedreamLayerColorReuseV64,nodes.SeedreamLayerColorReuseV64Fast]:
            out=cls().correct_and_reuse(ref,stack,normalize_alpha_levels=False)
            self.assertFalse(out[7][0,:,0:4].any())

    def test_new_fast_examples_and_only_api_wiring(self):
        import json
        api=json.loads((ROOT/'examples/api_prompt_fast.json').read_text(encoding='utf8'))
        workflow=json.loads((ROOT/'examples/workflow_fast.json').read_text(encoding='utf8'))
        only=json.loads((ROOT/'examples/api_prompt_color_only.json').read_text(encoding='utf8'))
        self.assertEqual(api['3']['class_type'],'SeedreamLayerColorReuseV64Fast')
        self.assertFalse(api['3']['inputs']['color_only'])
        self.assertEqual(api['12']['inputs']['images'],['3',7])
        self.assertTrue(only['3']['inputs']['color_only'])
        node=next(n for n in workflow['nodes'] if n['id']==3)
        self.assertEqual(node['type'],'SeedreamLayerColorReuseV64Fast')
        self.assertEqual(node['properties']['Node name for S&R'],node['type'])
        self.assertEqual(node['widgets_values'],['top_to_bottom',True,40.,'comfy_transparency',False])
        self.assertEqual(node['outputs'][7]['name'],nodes.SeedreamLayerColorReuseV64.RETURN_NAMES[7])
        self.assertIn([12,3,7,12,0,'IMAGE'],workflow['links'])

    def test_fast_layer_stages_and_reductions_do_not_change_standard_globals(self):
        import numpy as np
        from seedream_modes_test_plugin.engine import core
        from seedream_modes_test_plugin.engine.fast import calibrate_fast, _runtime
        self.assertIs(core.np,np)
        self.assertIsNot(_runtime.np,np)
        self.assertIsNot(core._map_layer_stage,_runtime._map_layer_stage)
        ref,images=scene()
        node=nodes.SeedreamLayerColorReuseV64Fast()
        captured={}
        def run(*args,**kwargs):
            result=calibrate_fast(*args,**kwargs)
            captured.update(result[1])
            return result
        node._run_engine=run
        actual=node.correct_and_reuse(ref,images)
        expected=nodes.SeedreamLayerColorReuseV64().correct_and_reuse(ref,images)
        self.assertEqual(captured['execution_layer_workers'],3)
        self.assertIn('map_initial_layer',captured['execution_stage_seconds'])
        for a,b in zip(actual,expected):
            if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
            else:self.assertEqual(a,b)
        self.assertIs(core.np,np)

    def test_old_examples_without_new_inputs_still_compatible(self):
        import json
        data=json.loads((ROOT/'examples/api_prompt.json').read_text(encoding='utf8'))
        prompt=data.get('prompt',data)
        for item in prompt.values():
            if isinstance(item,dict) and item.get('class_type')=='SeedreamLayerColorReuseV64':
                self.assertNotIn('color_only',item['inputs'])
        self.assertEqual(nodes.SeedreamLayerColorReuseV64.correct_and_reuse.__defaults__[-1],False)


if __name__=='__main__':unittest.main()
