"""Task-specific detection and exact-outside-mask composition nodes."""

import json

import numpy as np
import torch

from .core.task_masks import LargeMaskConfig, large_object_mask
from .nodes import CATEGORY, check_image


def check_mask(mask, image, name):
    if (
        not isinstance(mask, torch.Tensor)
        or mask.ndim != 3
        or len(mask) not in (1, len(image))
        or tuple(mask.shape[1:]) != tuple(image.shape[1:3])
    ):
        raise ValueError(
            f"{name}: expected BxHxW matching original; batch 1 or original batch"
        )
    if (
        not torch.isfinite(mask).all()
        or mask.min() < 0
        or mask.max() > 1
    ):
        raise ValueError(f"{name}: finite values in [0,1] required")


class LargeObjectMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "edited": ("IMAGE",),
                "alignment": (
                    ["global", "none"],
                    {
                        "default": "global",
                        "tooltip": "只将编辑图全局对齐到原图网格；none 要求尺寸相同。",
                    },
                ),
                "high": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.1, "max": 50.0, "step": 0.1},
                ),
                "low": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.05, "max": 50.0, "step": 0.1},
                ),
                "min_area_fraction": (
                    "FLOAT",
                    {
                        "default": 0.001,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.0001,
                    },
                ),
                "min_radius": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 128.0, "step": 1.0},
                ),
                "close_radius": (
                    "INT",
                    {"default": 5, "min": 0, "max": 64},
                ),
                "fill_holes": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "填充封闭孔洞；请检查不应删除的封闭区域。",
                    },
                ),
                "padding": ("INT", {"default": 8, "min": 0, "max": 256}),
                "feather": (
                    "INT",
                    {
                        "default": 24,
                        "min": 0,
                        "max": 512,
                        "tooltip": "不透明 padding 外侧的五次羽化宽度，核心保持 1。",
                    },
                ),
                "exclude_expand": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 512,
                        "tooltip": "接 Robust 节点时，该节点的 exclude_expand 设为 0。",
                    },
                ),
            },
            "optional": {
                "include_mask": (
                    "MASK",
                    {"tooltip": "白色强制纳入编辑区，分辨率须与原图相同。"},
                ),
                "protect_mask": (
                    "MASK",
                    {"tooltip": "白色强制保护原图，优先级最高。"},
                ),
            },
        }

    RETURN_TYPES = (
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "IMAGE",
        "MASK",
        "MASK",
        "STRING",
    )
    RETURN_NAMES = (
        "edit_mask",
        "exclude_mask",
        "blend_mask",
        "support_mask",
        "aligned_edited",
        "difference_evidence",
        "review_mask",
        "diagnostics_json",
    )
    FUNCTION = "detect"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "生成大面积不透明物体的候选编辑、拟合排除和融合遮罩，不提供语义分割"
        "保证，也不执行最终校色、局部光流或 Poisson。"
    )

    def detect(
        self,
        original,
        edited,
        alignment="global",
        high=4.0,
        low=2.0,
        min_area_fraction=0.001,
        min_radius=4.0,
        close_radius=5,
        fill_holes=True,
        padding=8,
        feather=24,
        exclude_expand=32,
        include_mask=None,
        protect_mask=None,
    ):
        from comfy.model_management import throw_exception_if_processing_interrupted

        check_image(original, "original")
        check_image(edited, "edited")
        if len(edited) not in (1, len(original)):
            raise ValueError("edited batch must be 1 or original batch")
        for name, mask in (
            ("include_mask", include_mask),
            ("protect_mask", protect_mask),
        ):
            if mask is not None:
                check_mask(mask, original, name)
        config = LargeMaskConfig(
            high=high,
            low=low,
            min_area_fraction=min_area_fraction,
            min_radius=min_radius,
            close_radius=close_radius,
            fill_holes=fill_holes,
            padding=padding,
            feather=feather,
            exclude_expand=exclude_expand,
            alignment=alignment,
        )
        keys = [
            "edit_mask",
            "exclude_mask",
            "blend_mask",
            "support_mask",
            "aligned_edited",
            "evidence",
            "review_mask",
        ]
        columns = [[] for _ in keys]
        reports = []
        for index in range(len(original)):
            throw_exception_if_processing_interrupted()
            original_array = (
                original[index].detach().cpu().float().numpy() * 255.0
            )
            edited_array = (
                edited[min(index, len(edited) - 1)]
                .detach()
                .cpu()
                .float()
                .numpy()
                * 255.0
            )
            masks = {
                name: (
                    None
                    if mask is None
                    else mask[min(index, len(mask) - 1)]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
                for name, mask in (
                    ("include_mask", include_mask),
                    ("protect_mask", protect_mask),
                )
            }
            result = large_object_mask(
                original_array, edited_array, config, **masks
            )
            for column, key in zip(columns, keys):
                value = result[key].astype(np.float32)
                if key == "aligned_edited":
                    value = value / 255.0
                column.append(torch.from_numpy(value))
            reports.append({"batch_index": index, **result["stats"]})
        return tuple(torch.stack(column).to(original.device) for column in columns) + (
            json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False),
        )


class RestrictedComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "corrected_edited": ("IMAGE",),
                "blend_mask": ("MASK",),
            },
            "optional": {
                "protect_mask": (
                    "MASK",
                    {"tooltip": "白色覆盖 blend 并精确保留原图。"},
                )
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("restored", "support_mask")
    FUNCTION = "compose"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "只执行遮罩合成，不拟合、不配准、不模糊、不执行 Poisson；alpha=0 精确"
        "保留原图张量值，alpha=1 完全替换。"
    )

    def compose(
        self, original, corrected_edited, blend_mask, protect_mask=None
    ):
        check_image(original, "original")
        check_image(corrected_edited, "corrected_edited")
        if (
            corrected_edited.shape[1:] != original.shape[1:]
            or len(corrected_edited) not in (1, len(original))
        ):
            raise ValueError(
                "corrected_edited must be aligned, same size; batch 1 or original batch"
            )
        check_mask(blend_mask, original, "blend_mask")
        if protect_mask is not None:
            check_mask(protect_mask, original, "protect_mask")
        dtype = torch.float64 if original.dtype == torch.float64 else torch.float32
        original_values = original.to(dtype=dtype)
        generated = corrected_edited.to(
            device=original_values.device, dtype=dtype
        ).expand_as(original_values)
        alpha = (
            blend_mask.to(device=original_values.device, dtype=dtype)
            .expand(original_values.shape[:3])
            .clone()
        )
        if protect_mask is not None:
            alpha[
                protect_mask.to(original_values.device).expand(
                    original_values.shape[:3]
                )
                > 0.5
            ] = 0
        active = alpha > 0
        output = original_values.clone()
        mixed = original_values * (1 - alpha[..., None]) + generated * alpha[..., None]
        output[active] = mixed[active]
        output[alpha == 1] = generated[alpha == 1]
        return output, active.to(torch.float32)
