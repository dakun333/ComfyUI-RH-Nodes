"""ComfyUI adapters for robust masked color matching."""

from __future__ import annotations

import json
import logging

import numpy as np
import torch

from .core import core_feather, match_frame, prepare_mask


ALPHA_THRESHOLD = 0.999


def _images(value, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 4
        or value.shape[-1] not in (3, 4)
        or any(size == 0 for size in value.shape)
    ):
        got = (
            tuple(int(size) for size in value.shape)
            if isinstance(value, torch.Tensor)
            else type(value).__name__
        )
        raise ValueError(
            f"{name}: expected nonempty ComfyUI IMAGE [B,H,W,3] or [B,H,W,4], "
            f"got {got}"
        )
    return value


def _alpha_planes(value):
    """Effective alpha [B,H,W] float64 in [0,1], or None if absent/opaque."""
    if value.shape[-1] != 4:
        return None
    alpha = value[..., 3].detach().to(device="cpu", dtype=torch.float64).numpy()
    if bool((alpha >= ALPHA_THRESHOLD).all()):
        return None
    return alpha


def _exclusion_source(mask_provided, alpha_excluded):
    if mask_provided:
        return "mask+alpha" if alpha_excluded else "mask"
    return "alpha" if alpha_excluded else "none"


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
                    {
                        "tooltip": "原始参考图，RGB 或 RGBA；透明 alpha 不参与拟合。"
                    },
                ),
                "target": (
                    "IMAGE",
                    {
                        "tooltip": "待校色的完整 AI 编辑图 / edited image to correct；"
                        "输出保留它的 alpha。"
                    },
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
            },
            "optional": {
                "exclude_mask": (
                    "MASK",
                    {
                        "tooltip": "白=排除编辑物体，黑=共享背景；可省略。"
                        "RGBA 输入的透明 alpha 自动并入排除区。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("corrected_image", "fit_region_not_blend_mask", "report")
    FUNCTION = "match"
    CATEGORY = "RH-Nodes/color"
    DESCRIPTION = (
        "用排除区之外的对应背景像素拟合鲁棒 RGB 仿射变换，再校正整张 target。"
        "白色排除区不参与统计，但仍接受校色；输入带透明 alpha 时，"
        "alpha<0.999 的像素自动并入排除区。reference 与 target 必须同尺寸、"
        "同位置；节点不做自动图像配准。"
    )

    def match(
        self,
        reference,
        target,
        exclude_mask=None,
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
        batch = target.shape[0]
        if reference.shape[1:3] != target.shape[1:3]:
            raise ValueError(
                "reference and target must have identical H,W and corresponding "
                "content; resize/register explicitly"
            )
        if reference.shape[0] not in (1, batch):
            raise ValueError(
                "reference batch must be 1 (broadcast) or equal to target batch"
            )
        mask_provided = exclude_mask is not None
        if mask_provided:
            mask = _masks(exclude_mask)
            if mask.shape[0] not in (1, batch):
                raise ValueError(
                    "mask batch must be 1 (broadcast) or equal to target batch"
                )
        else:
            mask = torch.zeros((1, *target.shape[1:3]), dtype=torch.float32)
        reference_alpha = _alpha_planes(reference)
        target_alpha = _alpha_planes(target)
        outputs, regions, reports = [], [], []
        for index in range(batch):
            target_frame = _frame(target, index)
            alpha_valid = None
            for planes in (reference_alpha, target_alpha):
                if planes is None:
                    continue
                plane = planes[0 if planes.shape[0] == 1 else index]
                valid = plane >= ALPHA_THRESHOLD
                alpha_valid = valid if alpha_valid is None else (alpha_valid & valid)
            result, region, report = match_frame(
                _frame(reference, index)[..., :3],
                target_frame[..., :3],
                _frame(mask, index),
                alpha_valid=alpha_valid,
                strength=strength,
                exclude_expand=exclude_expand,
                mask_threshold=mask_threshold,
                mask_resize=mask_resize,
                max_samples=max_samples,
                iterations=iterations,
                outlier_tolerance=outlier_tolerance,
            )
            frame = result
            if target_frame.shape[-1] == 4:
                frame = np.concatenate(
                    [result, target_frame[..., 3:4]], axis=-1
                )
            outputs.append(torch.from_numpy(frame.astype(np.float32)))
            regions.append(torch.from_numpy(region))
            report["exclusion_source"] = _exclusion_source(
                mask_provided, report["alpha_excluded_pixels"]
            )
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
    CATEGORY = "RH-Nodes/color"
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
