"""ComfyUI frame-batch wrapper for pure glow/effect UnMult, without AI models."""
from __future__ import annotations

import json
import math
from numbers import Integral, Real
import time

import numpy as np
import torch

from .light_unmult import LightEffectsUnmultKernel
from .video_io import MattingOutputBuffer, resolve_media, validate_output_mode


def checker_preview_source(foreground, alpha, square=32):
    """Alpha-over in the SAME numeric space as the legacy AE-derived kernel.

    Do not use CorridorKey's linear-light checker: that changes source-space
    UnMult's black-background reconstruction and changes weak-glow appearance.
    """
    h, w = alpha.shape
    yy, xx = np.ogrid[:h, :w]
    grid = np.where(((yy // square + xx // square) % 2) == 0,
                    0.35, 0.65).astype(np.float32)
    return np.clip(foreground * alpha[..., None]
                   + grid[..., None] * (1.0 - alpha[..., None]), 0.0, 1.0)


def validate_light_inputs(images, black_threshold, dither, auto_background, processing_device):
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a ComfyUI IMAGE tensor")
    if images.ndim != 4 or any(int(x) < 1 for x in images.shape[:3]):
        raise ValueError("images must be a nonempty [N,H,W,C] frame batch")
    if images.shape[-1] not in (1, 2, 3, 4):
        raise ValueError("images must have 1, 2, 3 or 4 channels")
    if not torch.is_floating_point(images):
        raise TypeError("ComfyUI IMAGE values must be floating point in [0,1]")
    if isinstance(black_threshold, bool) or not isinstance(black_threshold, Integral) or not 0 <= black_threshold <= 255:
        raise ValueError("black_threshold must be an integer in [0,255]")
    if isinstance(dither, bool) or not isinstance(dither, Real) or not math.isfinite(dither) or not 0 <= dither <= 1:
        raise ValueError("dither must be finite and in [0,1]")
    if not isinstance(auto_background, bool):
        raise TypeError("auto_background must be a boolean")
    if processing_device not in ("auto", "cpu"):
        raise ValueError("processing_device must be 'auto' or 'cpu'")


class PureLightMatting:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "black_threshold": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1,
                "tooltip": "Alpha 黑点连续重映射，非二值裁切。0 不清理；255 全透明。过大会抹掉弱光。"}),
            "dither": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "固定、可复现的 RGB 亚色阶抖动。0 关闭；仅在保存 8bit 时可能有帮助，不消除压缩块。"}),
            "auto_background": ("BOOLEAN", {"default": True,
                "tooltip": "逐帧从四边估计恒定底色并反算前景 RGB。近黑底抑制暗噪声；白底补偿是估计。纯黑底可关闭以固定 B=0、避免底色估计跳变。"}),
            "processing_device": (["auto", "cpu"], {"default": "auto",
                "tooltip": "auto 使用 ComfyUI 当前计算设备；cpu 不占 GPU。逐帧处理、输出 CPU 张量，不加载或卸载任何模型。"}),
        }, "optional": {
            "images": ("IMAGE", {"tooltip": "纯光效IMAGE帧批次，与 video 二选一；无需Mask。"}),
            "video": ("VIDEO", {"tooltip": "直接连接加载视频；仅纯光效，不用于人物/实体。自动使用源帧率/音频。"}),
            "output_mode": (["images", "video"], {"default": "images", "tooltip": "只生成选定输出。video 保留Alpha，不生成RGB/Mask/棋盘批次副本。"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.01, "tooltip": "仅IMAGE→VIDEO时的帧率；VIDEO输入自动采用源帧率。"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "STRING", "VIDEO")
    RETURN_NAMES = ("foreground_rgb", "alpha", "rgba", "checker_preview", "report", "video")
    FUNCTION = "matte"
    CATEGORY = "RH-Nodes/matting"
    DESCRIPTION = (
        "仅用于纯光效视频/图像：AE UnMult 衍生的 Auto Q 颜色反算 + Alpha。"
        "无需权重、无需 Mask；不是 CorridorKey AI 推理，不适用于实体或人物。"
        "RGB/RGBA 是 straight、未预乘；Alpha 1=前景，0=透明。"
        "沿用原算法的输入数值色域（通常为 sRGB 编码值），预览也在该色域合成。"
        "逐帧独立，不含时序模型/稳定；边缘被光效占满时自动底色可能误估。"
        "IMAGE/VIDEO输入二选一；images/video输出模式互斥，VIDEO携带Alpha及源帧率/音频。"
    )

    def matte(self, images=None, black_threshold=0, dither=0.0,
              auto_background=True, processing_device="auto", video=None,
              output_mode="images", fps=24.0):
        from comfy import model_management as mm
        from comfy.utils import ProgressBar

        validate_output_mode(output_mode)
        mm.throw_exception_if_processing_interrupted()
        images, frame_rate, audio, metadata, input_kind = resolve_media(images, video, fps, preserve_input_alpha=True)
        validate_light_inputs(images, black_threshold, dither, auto_background, processing_device)
        mm.throw_exception_if_processing_interrupted()
        device = mm.get_torch_device() if processing_device == "auto" else torch.device("cpu")
        n, h, w, channels = map(int, images.shape)
        outputs = MattingOutputBuffer(output_mode, n, h, w, frame_rate, audio, metadata)
        progress = ProgressBar(n)
        kernel = LightEffectsUnmultKernel()
        started = time.perf_counter()
        transparent_pixels = semitransparent_pixels = 0
        # One frame at a time bounds transient device memory. Never alter torch's
        # global thread settings or evict models from the running ComfyUI queue.
        with torch.inference_mode():
            for i in range(n):
                mm.throw_exception_if_processing_interrupted()
                frame = images[i:i + 1].detach().to(device=device, dtype=torch.float32)
                result, _ = kernel.apply(frame, black_threshold=black_threshold,
                                         dither=dither, auto_background=auto_background)
                result_cpu = result[0].to(device="cpu")
                a = result_cpu[..., 3].numpy()
                outputs.write(i, result_cpu[..., :3], result_cpu[..., 3],
                              lambda: checker_preview_source(result_cpu[..., :3].numpy(), a))
                transparent_pixels += int(np.count_nonzero(a == 0.0))
                semitransparent_pixels += int(np.count_nonzero((a > 0.0) & (a < 1.0)))
                # Keep no full-frame device tensor alive between iterations.
                del frame, result, _, result_cpu, a
                mm.throw_exception_if_processing_interrupted()
                progress.update_absolute(i + 1)
        report = {
            "algorithm": "Local AE-UnMult-derived Auto Q (frozen 2026-10-15 math)",
            "scope": "Pure light effects only; not semantic/solid-object matting",
            "frames": n, "width": w, "height": h, "input_channels": channels,
            "input_kind": input_kind,
            "device": str(device), "seconds": round(time.perf_counter() - started, 4),
            "settings": {"black_threshold": int(black_threshold), "dither": float(dither),
                         "auto_background": auto_background, "processing_device": processing_device},
            "weights_required": False, "mask_required": False,
            "temporal_model": False, "frame_order_preserved": True,
            "working_space": "input numeric RGB space (normally sRGB-coded ComfyUI IMAGE)",
            "alpha_convention": "straight RGB; alpha 1=foreground, 0=transparent",
            "checker_compositing": "source-space F*A + B*(1-A), NOT linear-light",
            "transparent_pixel_fraction": transparent_pixels / (n * h * w),
            "semitransparent_pixel_fraction": semitransparent_pixels / (n * h * w),
            "warnings": [
                "Use only pure glow/light artwork on a spatially uniform background.",
                "Auto background is independent per frame; changing/occluded borders may cause flicker.",
                "Single-background RGB and alpha are not uniquely identifiable; white gain is heuristic.",
                "An H.264 checker-preview video is not a transparent video.",
            ],
        }
        return outputs.finish(report)
