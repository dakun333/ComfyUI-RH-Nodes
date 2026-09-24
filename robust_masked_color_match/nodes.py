"""ComfyUI adapters for robust masked color matching."""

from __future__ import annotations

import json
import logging

import numpy as np
import torch

from .core import core_feather, match_frame, prepare_mask


def _images(value, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 4
        or value.shape[-1] != 3
        or any(size == 0 for size in value.shape)
    ):
        raise ValueError(f"{name}: expected nonempty ComfyUI IMAGE [B,H,W,3]")
    return value


def _masks(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError("mask must be a torch Tensor")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or any(size == 0 for size in value.shape):
        raise ValueError("mask must be nonempty [B,H,W] or [H,W]")
    return value


def _frame(value, index):
    return (
        value[0 if value.shape[0] == 1 else index]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
    )


class RobustMaskedColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": (
                    "IMAGE",
                    {"tooltip": "原始参考图 / original image defining desired colors."},
                ),
                "target": (
                    "IMAGE",
                    {"tooltip": "待校色的完整 AI 编辑图 / edited image to correct."},
                ),
                "exclude_mask": (
                    "MASK",
                    {"tooltip": "白=排除编辑物体，黑=共享背景；不是最终羽化遮罩。"},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "exclude_expand": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "排除区外扩像素；外部已扩张时设为 0。",
                    },
                ),
                "mask_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.001, "max": 1.0, "step": 0.01},
                ),
                "mask_resize": (
                    ["error", "nearest"],
                    {
                        "default": "error",
                        "tooltip": "nearest 只修正尺寸，不负责几何对齐。",
                    },
                ),
                "max_samples": (
                    "INT",
                    {"default": 100000, "min": 100, "max": 1000000, "step": 1000},
                ),
                "iterations": (
                    "INT",
                    {"default": 12, "min": 1, "max": 100},
                ),
                "outlier_tolerance": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.01,
                        "max": 255.0,
                        "step": 0.1,
                        "tooltip": "鲁棒残差阈值，以 0-255 RGB 级数计。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("corrected_image", "fit_region_not_blend_mask", "report")
    FUNCTION = "match"
    CATEGORY = "image/color/robust_match"
    DESCRIPTION = (
        "用遮罩外的对应背景像素拟合鲁棒 RGB 仿射变换，再校正整张 target。"
        "白色排除区不参与统计，但仍接受校色。reference 与 target 必须同尺寸、"
        "同位置；节点不做自动图像配准。"
    )

    def match(
        self,
        reference,
        target,
        exclude_mask,
        strength=1.0,
        exclude_expand=32,
        mask_threshold=0.5,
        mask_resize="error",
        max_samples=100000,
        iterations=12,
        outlier_tolerance=3.0,
    ):
        reference = _images(reference, "reference")
        target = _images(target, "target")
        mask = _masks(exclude_mask)
        batch = target.shape[0]
        if reference.shape[1:] != target.shape[1:]:
            raise ValueError(
                "reference and target must have identical H,W,3 and corresponding "
                "content; resize/register explicitly"
            )
        if reference.shape[0] not in (1, batch) or mask.shape[0] not in (1, batch):
            raise ValueError(
                "reference/mask batch must be 1 (broadcast) or equal to target batch"
            )
        outputs, regions, reports = [], [], []
        for index in range(batch):
            result, region, report = match_frame(
                _frame(reference, index),
                _frame(target, index),
                _frame(mask, index),
                strength=strength,
                exclude_expand=exclude_expand,
                mask_threshold=mask_threshold,
                mask_resize=mask_resize,
                max_samples=max_samples,
                iterations=iterations,
                outlier_tolerance=outlier_tolerance,
            )
            outputs.append(torch.from_numpy(result.astype(np.float32)))
            regions.append(torch.from_numpy(region))
            reports.append({"batch_index": index, **report})
            for warning in report["warnings"]:
                logging.warning("[Robust Masked Color Match] %s", warning)
        report_text = json.dumps(
            {"frames": reports}, ensure_ascii=False, indent=2, allow_nan=False
        )
        return (
            torch.stack(outputs).to(target.device),
            torch.stack(regions).to(target.device),
            report_text,
        )


class CoreFeatherMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "物体白、背景黑的原始遮罩。"}),
                "size_reference": (
                    "IMAGE",
                    {"tooltip": "只使用此图的尺寸和批次，不读取颜色。"},
                ),
                "core_expand": (
                    "INT",
                    {"default": 24, "min": 0, "max": 4096},
                ),
                "feather_width": (
                    "FLOAT",
                    {
                        "default": 96.0,
                        "min": 0.1,
                        "max": 4096.0,
                        "step": 1.0,
                        "tooltip": "向外过渡总宽度，不是 Gaussian sigma。",
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.001, "max": 1.0, "step": 0.01},
                ),
                "mask_resize": (
                    ["error", "nearest"],
                    {"default": "error"},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("blend_mask",)
    FUNCTION = "feather"
    CATEGORY = "mask/robust_match"
    DESCRIPTION = (
        "物体及安全边距保持严格 1，只在外侧用五次 smoothstep 羽化到 0。"
    )

    def feather(
        self,
        mask,
        size_reference,
        core_expand=24,
        feather_width=96.0,
        threshold=0.5,
        mask_resize="error",
    ):
        image = _images(size_reference, "size_reference")
        mask = _masks(mask)
        batch, height, width, _ = image.shape
        if mask.shape[0] not in (1, batch):
            raise ValueError("mask batch must be 1 or equal to size_reference batch")
        outputs = []
        for index in range(batch):
            prepared, _ = prepare_mask(
                _frame(mask, index), height, width, mask_resize
            )
            outputs.append(
                torch.from_numpy(
                    core_feather(
                        prepared, core_expand, feather_width, threshold
                    ).astype(np.float32)
                )
            )
        return (torch.stack(outputs).to(image.device),)


NODE_CLASS_MAPPINGS = {
    "RCMRobustMaskedColorMatch": RobustMaskedColorMatch,
    "RCMCoreFeatherMask": CoreFeatherMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RCMRobustMaskedColorMatch": "Robust Masked Color Match / 鲁棒遮罩校色",
    "RCMCoreFeatherMask": "Core-Preserving Feather Mask / 保核外羽化",
}
