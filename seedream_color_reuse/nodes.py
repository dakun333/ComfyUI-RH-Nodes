"""ComfyUI node for V6.4 double layer calibration and exact pixel reuse."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .engine import Config, Layer, calibrate, composite


def _single(value, name):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f'{name}: expected one setting/reference, got {len(value)}')
        return value[0]
    return value


def _batch(value, name, *, mask=False):
    parts = value if isinstance(value, (list, tuple)) else [value]
    arrays = []
    for item in parts:
        if not isinstance(item, torch.Tensor):
            raise ValueError(f'{name} must contain tensors')
        array = item.detach().to(device='cpu', dtype=torch.float32).numpy()
        if mask and array.ndim == 2:
            array = array[None]
        expected_ndim = 3 if mask else 4
        if array.ndim != expected_ndim or not array.shape[0]:
            raise ValueError(f'{name}: invalid shape {array.shape}')
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError(f'{name}: expected finite values in [0,1]')
        arrays.append(array)
    if not arrays:
        raise ValueError(f'{name} is empty')
    if len({array.shape[1:] for array in arrays}) != 1:
        raise ValueError(f'{name}: all canvases must have identical dimensions')
    return np.concatenate(arrays, axis=0)


def _rgba(images, masks, name, convention='comfy_transparency'):
    array = _batch(images, name)
    if min(array.shape[1:3]) < 2 or array.shape[-1] not in (3, 4):
        raise ValueError(f'{name}: expected [N,H,W,3 or 4], H/W >= 2')
    if masks is not None:
        mask = _batch(masks, name + '_masks', mask=True)
        if mask.shape != array.shape[:3]:
            raise ValueError(f'{name}: one same-size mask per image is required')
        alpha = 1 - mask if convention == 'comfy_transparency' else mask
        if array.shape[-1] == 4 and not np.allclose(array[..., 3], alpha, atol=1e-6):
            raise ValueError(f'{name}: RGBA alpha disagrees with connected MASK')
    elif array.shape[-1] == 4:
        alpha = array[..., 3]
    elif name == 'reference':
        alpha = np.ones(array.shape[:3], np.float32)
    else:
        raise ValueError('RGB layer images have no alpha; connect image_masks or supply RGBA images')
    return np.concatenate((array[..., :3], alpha[..., None]), axis=-1).astype(np.float32)


def _resize_rgba(image, width, height):
    """Premultiplied resize so transparent RGB cannot create fringes."""
    if image.shape[:2] == (height, width):
        return image.copy()
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
    alpha = tensor[:, 3:4]
    premultiplied = torch.cat((tensor[:, :3] * alpha, alpha), dim=1)
    downsample = width < image.shape[1] and height < image.shape[0]
    if downsample:
        resized = F.interpolate(premultiplied, size=(height, width), mode='area')
    else:
        resized = F.interpolate(premultiplied, size=(height, width), mode='bilinear', align_corners=False)
    alpha = resized[:, 3:4].clamp(0, 1)
    rgb = torch.where(alpha > 1e-7, resized[:, :3] / alpha.clamp_min(1e-7), 0)
    return torch.cat((rgb.clamp(0, 1), alpha), dim=1)[0].permute(1, 2, 0).numpy()


def _engine_layers(source, order):
    """Convert aligned full canvases into the engine's bottom-to-top local layers."""
    count, height, width = source.shape[:3]
    indices = list(range(count))
    if order == 'top_to_bottom':
        indices.reverse()
    layers = []
    for input_index in indices:
        image = source[input_index]
        alpha_full = np.rint(np.clip(image[..., 3], 0, 1) * 255).astype(np.uint8)
        yy, xx = np.nonzero(alpha_full > 0)
        if len(xx):
            x0, x1 = int(xx.min()), int(xx.max()) + 1
            y0, y1 = int(yy.min()), int(yy.max()) + 1
        else:
            x0, y0, x1, y1 = 0, 0, width, height
        layers.append(Layer(
            name=f'image_{input_index:04d}.png',
            rgb=(np.clip(image[y0:y1, x0:x1, :3], 0, 1) * 255).astype(np.float32),
            alpha=alpha_full[y0:y1, x0:x1],
            box=(x0, y0, x1, y1),
            metadata={'input_index': input_index},
        ))
    return layers


def _expand_layers(corrected, source):
    result = source.copy()
    seen = set()
    for layer in corrected:
        input_index = int(layer.metadata['input_index'])
        if input_index in seen or not 0 <= input_index < len(source):
            raise RuntimeError('Engine returned an invalid layer index')
        seen.add(input_index)
        local = np.concatenate((
            np.clip(layer.rgb / 255, 0, 1),
            layer.alpha[..., None].astype(np.float32) / 255,
        ), axis=-1)
        result[input_index][layer.region] = local
    if seen != set(range(len(source))):
        raise RuntimeError('Engine changed the layer count')
    return result.astype(np.float32)



def _composite_rgba(layers, order):
    """Merge the returned full-canvas layers, preserving straight RGBA alpha.

    Composite the actual output batch, not the quantized diagnostic RGB or the
    reference. Invisible RGB never contributes, and a transparent canvas is
    unpremultiplied exactly once for PNG/IMAGE consumers.
    """
    stack = layers[::-1] if order == 'top_to_bottom' else layers
    rgb = np.zeros_like(layers[0, ..., :3])
    alpha = np.zeros_like(layers[0, ..., 3:4])
    for image in stack:
        a = image[..., 3:4]
        rgb = image[..., :3] * a + rgb * (1 - a)
        alpha = a + alpha * (1 - a)
    straight = np.divide(rgb, alpha, out=np.zeros_like(rgb), where=alpha > 0)
    return np.concatenate((np.clip(straight, 0, 1), alpha), axis=-1)[None].astype(np.float32)


def _difference_heatmap(error, valid, maximum):
    """Blue -> cyan -> yellow -> red, with invalid pixels black."""
    value = np.clip(error.astype(np.float32) / maximum, 0, 1)
    stops = np.array([0, .25, .5, .75, 1], np.float32)
    colors = np.array([
        [0.02, 0.03, 0.18],
        [0.05, 0.25, 1.00],
        [0.00, 0.95, 0.95],
        [1.00, 0.92, 0.05],
        [0.95, 0.02, 0.02],
    ], np.float32)
    heat = np.stack([np.interp(value, stops, colors[:, channel]) for channel in range(3)], axis=-1)
    heat[~valid] = 0
    return heat.astype(np.float32)


class SeedreamLayerColorReuseV64:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'reference': ('IMAGE', {
                'tooltip': 'One original/reference image. Pixel reuse cannot be decided without it.'}),
            'images': ('IMAGE', {
                'tooltip': 'Aligned RGBA layer batch. Default order: top layer to bottom layer.'}),
            'layer_order': (['top_to_bottom', 'bottom_to_top'],),
            'normalize_alpha_levels': ('BOOLEAN', {
                'default': True,
                'tooltip': 'Apply V6.4 Alpha Levels 3-252 once. Disable to preserve input alpha byte-for-byte.'}),
            'heatmap_max': ('FLOAT', {
                'default': 40.0, 'min': 1.0, 'max': 255.0, 'step': 1.0,
                'tooltip': 'Shared max-channel RGB-error scale for both heatmap frames.'}),
            'mask_convention': (['comfy_transparency', 'alpha_opacity'],),
        }, 'optional': {
            'image_masks': ('MASK',),
            'reference_mask': ('MASK',),
            'color_only': ('BOOLEAN', {
                'default': False, 'label_on': '仅校色：开启', 'label_off': '仅校色：关闭',
                'tooltip': '关闭保持现有流程。开启只学习并应用各层颜色映射；不复用参考像素，不做边缘反解或空间残差修复。Alpha Levels 开关仍独立生效。'}),
        }}

    INPUT_IS_LIST = True
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'MASK', 'IMAGE', 'FLOAT', 'FLOAT', 'FLOAT', 'IMAGE')
    RETURN_NAMES = ('images', 'reuse_image', 'reuse_mask', 'difference_heatmaps',
                    'mae_before', 'mae_after', 'mae_reduction', '拼接结果')
    FUNCTION = 'correct_and_reuse'
    CATEGORY = 'RHNodes/seedream'
    DESCRIPTION = ('Double calibration: whole-layer color fit, verified exact reuse, then a '
                   'low-texture-gated second fit only on non-reused pixels. The heatmap output '
                   'contains two frames in order: BEFORE and AFTER. The final IMAGE output '
                   'merges corrected layers in their original order into one straight-RGBA canvas.')

    def _run_engine(self, *args, **kwargs):
        return calibrate(*args, **kwargs)

    def _merge_output(self, corrected, source, order):
        return _composite_rgba(source, order)

    def correct_and_reuse(self, reference, images, layer_order='top_to_bottom',
                          normalize_alpha_levels=True, heatmap_max=40.0,
                          mask_convention='comfy_transparency', image_masks=None,
                          reference_mask=None, color_only=False):
        only = _single(color_only, 'color_only')
        if not isinstance(only, (bool, np.bool_)):
            raise ValueError('color_only must be a BOOLEAN')
        order = _single(layer_order, 'layer_order')
        normalize = bool(_single(normalize_alpha_levels, 'normalize_alpha_levels'))
        heat_max = float(_single(heatmap_max, 'heatmap_max'))
        convention = _single(mask_convention, 'mask_convention')
        if order not in ('top_to_bottom', 'bottom_to_top'):
            raise ValueError('Invalid layer_order')
        if convention not in ('comfy_transparency', 'alpha_opacity'):
            raise ValueError('Invalid mask_convention')
        if not np.isfinite(heat_max) or not 0 < heat_max <= 255:
            raise ValueError('heatmap_max must be in (0,255]')

        refs = _rgba(reference, reference_mask, 'reference', convention)
        if len(refs) != 1:
            raise ValueError('Exactly one reference image is required')
        source = _rgba(images, image_masks, 'images', convention)
        height, width = source.shape[1:3]
        ref = _resize_rgba(refs[0], width, height)

        raw_layers = _engine_layers(source, order)
        model_guide, _ = composite(raw_layers, (height, width))
        reference_u8 = np.rint(np.clip(ref, 0, 1) * 255).astype(np.uint8)
        config = Config(alpha_levels=int(normalize), alpha_black=3, alpha_white=252)
        # Leave the default Config construction/path unchanged for old APIs.
        if only:
            config.color_only = 1
        corrected, report, diagnostics = self._run_engine(
            raw_layers, reference_u8, config, model_guide=model_guide)
        corrected_rgba = _expand_layers(corrected, source)

        exact = diagnostics['exact'].astype(bool)
        conflicts = diagnostics['conflicts'].astype(bool) & ~exact
        after_rgb = diagnostics['after'].astype(np.float32) / 255
        reference_rgb = reference_u8[..., :3].astype(np.float32) / 255
        reuse_image = after_rgb * .12
        reuse_image[exact] = (.78 * reference_rgb[exact]
                              + .22 * np.array([.08, 1., .18], np.float32))
        reuse_image[conflicts] = np.array([1., .05, .05], np.float32)

        valid = ((reference_u8[..., 3] == 255)
                 & (diagnostics['alpha'] >= 1 - 1e-6))
        before_mae = report['before']['mae']
        after_mae = report['after']['mae']
        if not valid.any() or before_mae is None or after_mae is None:
            raise ValueError('No common opaque pixels are available for metrics')
        before_error = np.max(np.abs(
            diagnostics['before'].astype(np.float32) - reference_u8[..., :3]), axis=-1)
        after_error = np.max(np.abs(
            diagnostics['after'].astype(np.float32) - reference_u8[..., :3]), axis=-1)
        heatmaps = np.stack((
            _difference_heatmap(before_error, valid, heat_max),
            _difference_heatmap(after_error, valid, heat_max),
        ))

        before_mae = float(before_mae)
        after_mae = float(after_mae)
        tensor = lambda array: torch.from_numpy(np.ascontiguousarray(array))
        return (
            tensor(corrected_rgba),
            tensor(reuse_image[None]),
            tensor(exact.astype(np.float32)[None]),
            tensor(heatmaps),
            before_mae,
            after_mae,
            float(before_mae - after_mae),
            tensor(self._merge_output(corrected, corrected_rgba, order)),
        )


class SeedreamLayerColorReuseV64Fast(SeedreamLayerColorReuseV64):
    """Complete standard pipeline with exact CPU execution optimizations."""
    DESCRIPTION = ('与标准节点相同的校色、复用和保护策略；使用 RGB 分通道并行滤波、'
                   '合成缓冲复用和重复计算消除提速，不缩图、不降低保护质量。输出端口相同。')

    def _run_engine(self, *args, **kwargs):
        from .engine.fast import calibrate_fast
        return calibrate_fast(*args, **kwargs)

    def _merge_output(self, corrected, source, order):
        # Only process original alpha bounding boxes. No invisible RGB outside
        # these boxes contributes, and source-over layer order is unchanged.
        height, width = source.shape[1:3]
        rgb = np.zeros((height, width, 3), np.float32)
        alpha = np.zeros((height, width, 1), np.float32)
        scratch = np.empty_like(rgb)
        for layer in corrected:  # Engine order is always bottom-to-top.
            s = layer.region
            image = source[int(layer.metadata['input_index'])][s]
            a = image[..., 3:4]
            inverse = 1-a
            np.multiply(image[..., :3], a, out=scratch[s])
            np.multiply(rgb[s], inverse, out=rgb[s])
            np.add(scratch[s], rgb[s], out=rgb[s])
            np.multiply(alpha[s], inverse, out=alpha[s])
            np.add(a, alpha[s], out=alpha[s])
        straight = np.divide(rgb, alpha, out=np.zeros_like(rgb), where=alpha > 0)
        return np.concatenate((np.clip(straight, 0, 1), alpha), axis=-1)[None].astype(np.float32)


class SeedreamV64LoadAlignedLayers:
    """Load a naturally sorted batch of already aligned RGBA PNG layers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'folder': ('STRING', {
                'default': 'seedream_v64_examples/01_perfect_combo/layers',
                'tooltip': 'Folder containing full-canvas, already aligned RGBA PNG layers.'}),
            'pattern': ('STRING', {'default': 'layer_*.png'}),
            'file_order': (['descending', 'ascending'],),
        }}

    RETURN_TYPES = ('IMAGE', 'MASK', 'STRING')
    RETURN_NAMES = ('images', 'masks', 'file_order_json')
    FUNCTION = 'load'
    CATEGORY = 'RHNodes/seedream'
    DESCRIPTION = ('Loads full-canvas RGBA PNG layers only; it does not read placement metadata '
                   'or perform geometric alignment. Descending layer_XX order is top-to-bottom.')

    @staticmethod
    def _files(folder, pattern):
        directory = Path(folder).expanduser()
        if not directory.is_absolute():
            import folder_paths
            directory = Path(folder_paths.get_input_directory()) / directory
        if not directory.is_dir():
            raise ValueError(f'Layer folder does not exist: {directory}')
        if '/' in pattern or '\\' in pattern or '..' in pattern:
            raise ValueError('pattern must match files in the selected folder only')
        natural = lambda path: [
            (0, int(part)) if part.isdigit() else (1, part.lower())
            for part in re.split(r'(\d+)', path.name)
        ]
        files = sorted((path for path in directory.glob(pattern)
                        if path.is_file() and path.suffix.lower() == '.png'), key=natural)
        if not files:
            raise ValueError('No matching PNG layers')
        return files

    @classmethod
    def IS_CHANGED(cls, folder, pattern, file_order='descending'):
        digest = hashlib.sha256(file_order.encode())
        for path in cls._files(folder, pattern):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def load(self, folder, pattern='layer_*.png', file_order='descending'):
        if file_order not in ('ascending', 'descending'):
            raise ValueError('Invalid file_order')
        paths = self._files(folder, pattern)
        if file_order == 'descending':
            paths.reverse()
        arrays = []
        for path in paths:
            with Image.open(path) as image:
                arrays.append(np.asarray(image.convert('RGBA'), dtype=np.float32) / 255)
        if len({array.shape for array in arrays}) != 1:
            raise ValueError('All layers must already share one canvas size')
        batch = np.stack(arrays).astype(np.float32)
        return (
            torch.from_numpy(batch),
            torch.from_numpy(1 - batch[..., 3]),
            json.dumps([str(path) for path in paths], ensure_ascii=False),
        )


NODE_CLASS_MAPPINGS = {
    'SeedreamLayerColorReuseV64': SeedreamLayerColorReuseV64,
    'SeedreamV64LoadAlignedLayers': SeedreamV64LoadAlignedLayers,
    'SeedreamLayerColorReuseV64Fast': SeedreamLayerColorReuseV64Fast,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'SeedreamLayerColorReuseV64': 'Seedream 分层双次校色与像素复用 V6.4',
    'SeedreamV64LoadAlignedLayers': 'Seedream V6.4 读取已对齐图层',
    'SeedreamLayerColorReuseV64Fast': 'Seedream 分层双次校色与像素复用 V6.4 Fast',
}
