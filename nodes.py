"""ComfyUI nodes based on Reference Color Restore with local occlusion seams."""

from __future__ import annotations

from dataclasses import asdict

import cv2
import numpy as np
import torch

from .algorithm import RestorationReport, restore_reference_colors


CATEGORY = "image/color correction"


def _as_image_batch(image: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(
            f"{name} must be a ComfyUI IMAGE in BHWC layout, got {tuple(image.shape)}"
        )
    return image[..., :3].detach().float().cpu().clamp(0.0, 1.0)


def _resize_reference(reference: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = shape
    height, width = reference.shape[:2]
    ratio_error = abs((target_width / target_height) / (width / height) - 1.0)
    if ratio_error > 0.02:
        raise ValueError(
            f"Aspect ratios differ by {100 * ratio_error:.2f}%; refusing reference resize"
        )
    interpolation = cv2.INTER_AREA if height > target_height else cv2.INTER_CUBIC
    return cv2.resize(
        reference, (target_width, target_height), interpolation=interpolation
    )


def _format_report(index: int, report: RestorationReport, resized: bool) -> str:
    values = asdict(report)
    return "\n".join(
        [
            f"Batch item {index}",
            f"reference_resized: {str(resized).lower()}",
            f"background_rgb: {values['background_rgb']}",
            f"foreground: {100 * values['foreground_fraction']:.2f}%",
            f"safe_unchanged: {100 * values['unchanged_fraction']:.2f}%",
            f"unchanged_threshold: {values['threshold']:.2f}/255",
            f"safe_MAE_before_to_mapped: {values['before_mae']:.3f} -> "
            f"{values['mapped_mae']:.3f}/255",
            f"structurally_rescued_foreground: "
            f"{100 * values['structural_rescued_fraction']:.2f}%",
            f"exactly_restored_foreground: "
            f"{100 * values['hard_restored_fraction']:.2f}%",
            f"seam_compensated_foreground: "
            f"{100 * values['seam_compensated_fraction']:.2f}%",
            f"occlusion_core_foreground: "
            f"{100 * values['occlusion_core_fraction']:.2f}%",
            f"occlusion_zone_foreground: "
            f"{100 * values['occlusion_zone_fraction']:.2f}%",
            f"occlusion_seam_compensated_foreground: "
            f"{100 * values['occlusion_seam_compensated_fraction']:.2f}%",
        ]
    )


def _run_batch(
    ai_image: torch.Tensor,
    reference_image: torch.Tensor,
    *,
    resize_reference: bool,
    **options,
):
    edited_batch = _as_image_batch(ai_image, "ai_image")
    reference_batch = _as_image_batch(reference_image, "reference_image")
    edited_count = edited_batch.shape[0]
    reference_count = reference_batch.shape[0]
    if edited_count != reference_count and edited_count != 1 and reference_count != 1:
        raise ValueError(
            "Image batches must have equal sizes, or one input batch must contain one image; "
            f"got {edited_count} and {reference_count}"
        )

    corrected_items = []
    foreground_items = []
    unchanged_items = []
    occlusion_core_items = []
    occlusion_zone_items = []
    occlusion_compensated_items = []
    reports = []
    count = max(edited_count, reference_count)
    for index in range(count):
        edited = edited_batch[0 if edited_count == 1 else index].numpy().astype(
            np.float64, copy=False
        )
        reference = reference_batch[
            0 if reference_count == 1 else index
        ].numpy().astype(np.float64, copy=False)
        resized = False
        if edited.shape != reference.shape:
            if not resize_reference:
                raise ValueError(
                    f"Batch item {index}: image shapes differ: {edited.shape} vs "
                    f"{reference.shape}; enable resize_reference only when aligned"
                )
            reference = _resize_reference(reference, edited.shape[:2])
            resized = True
        try:
            (
                corrected,
                foreground,
                unchanged,
                report,
                occlusion_core,
                occlusion_zone,
                occlusion_compensated,
            ) = restore_reference_colors(
                edited,
                reference,
                return_occlusion_masks=True,
                **options,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Reference Color Restore (Occlusion) failed at batch item {index}: {exc}"
            ) from exc
        corrected_items.append(corrected.astype(np.float32))
        foreground_items.append(foreground.astype(np.float32))
        unchanged_items.append(unchanged.astype(np.float32))
        occlusion_core_items.append(occlusion_core.astype(np.float32))
        occlusion_zone_items.append(occlusion_zone.astype(np.float32))
        occlusion_compensated_items.append(occlusion_compensated.astype(np.float32))
        reports.append(_format_report(index, report, resized))

    return (
        torch.from_numpy(np.stack(corrected_items)),
        torch.from_numpy(np.stack(foreground_items)),
        torch.from_numpy(np.stack(unchanged_items)),
        torch.from_numpy(np.stack(occlusion_core_items)),
        torch.from_numpy(np.stack(occlusion_zone_items)),
        torch.from_numpy(np.stack(occlusion_compensated_items)),
        "\n\n".join(reports),
    )


RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK", "STRING")
RETURN_NAMES = (
    "corrected_image",
    "foreground_mask",
    "safe_unchanged_mask",
    "occlusion_core_mask",
    "occlusion_zone_mask",
    "occlusion_seam_mask",
    "report",
)


class ReferenceColorRestoreOcclusion:
    """Old Reference Color Restore defaults plus local 48px occlusion seam values."""

    DESCRIPTION = (
        "Reference Color Restore with the original global behavior preserved. "
        "Only the detected removed-object occlusion zone uses the separate "
        "occlusion blur and bridge width."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ai_image": (
                    "IMAGE",
                    {"tooltip": "AI extraction/edit on a solid background"},
                ),
                "reference_image": (
                    "IMAGE",
                    {"tooltip": "Aligned original/reference image"},
                ),
                "resize_reference": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Resize only when composition and aspect ratio are aligned",
                    },
                ),
            }
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "restore"
    CATEGORY = CATEGORY

    def restore(self, ai_image, reference_image, resize_reference):
        return _run_batch(
            ai_image,
            reference_image,
            resize_reference=resize_reference,
            max_samples=300_000,
            structural_rescue=True,
            hard_restore_high_confidence=True,
            seam_color_propagation=True,
            seam_max_distance=96.0,
            seam_decay=64.0,
            seam_residual_blur=12.0,
            seam_color_sigma=20.0,
            seam_bridge_width=12.0,
            occlusion_residual_blur=48.0,
            occlusion_bridge_width=48.0,
        )


class ReferenceColorRestoreOcclusionAdvanced:
    """Fully parameterized old node with two local occlusion seam controls."""

    DESCRIPTION = (
        "Advanced Reference Color Restore. seam_residual_blur and seam_bridge_width "
        "apply globally; occlusion_residual_blur and occlusion_bridge_width apply "
        "only around a conservatively detected removed-object occlusion. Set either "
        "occlusion value to 0 to disable the local pass."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ai_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "resize_reference": ("BOOLEAN", {"default": False}),
                "background_threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 80.0, "step": 0.1},
                ),
                "unchanged_threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 80.0, "step": 0.1},
                ),
                "boundary_guard": (
                    "INT",
                    {"default": 4, "min": 0, "max": 64, "step": 1},
                ),
                "transition_width": (
                    "FLOAT",
                    {"default": 6.0, "min": 1.0, "max": 128.0, "step": 0.5},
                ),
                "structural_rescue": ("BOOLEAN", {"default": True}),
                "structural_threshold": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "structural_max_shift": (
                    "INT",
                    {"default": 2, "min": 0, "max": 8, "step": 1},
                ),
                "exact_restore": ("BOOLEAN", {"default": True}),
                "exact_restore_threshold": (
                    "FLOAT",
                    {"default": 0.995, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "small_hole_no_feather_area": (
                    "INT",
                    {"default": 16, "min": 0, "max": 4096, "step": 1},
                ),
                "seam_color_propagation": ("BOOLEAN", {"default": True}),
                "seam_max_distance": (
                    "FLOAT",
                    {"default": 96.0, "min": 1.0, "max": 1024.0, "step": 1.0},
                ),
                "seam_decay": (
                    "FLOAT",
                    {"default": 64.0, "min": 1.0, "max": 1024.0, "step": 1.0},
                ),
                "seam_residual_blur": (
                    "FLOAT",
                    {"default": 12.0, "min": 0.1, "max": 128.0, "step": 0.5},
                ),
                "seam_color_sigma": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.1, "max": 128.0, "step": 0.5},
                ),
                "seam_bridge_width": (
                    "FLOAT",
                    {"default": 12.0, "min": 0.1, "max": 128.0, "step": 0.5},
                ),
                "occlusion_residual_blur": (
                    "FLOAT",
                    {
                        "default": 48.0,
                        "min": 0.0,
                        "max": 128.0,
                        "step": 0.5,
                        "tooltip": "Extra residual blur used only in the occlusion zone",
                    },
                ),
                "occlusion_bridge_width": (
                    "FLOAT",
                    {
                        "default": 48.0,
                        "min": 0.0,
                        "max": 128.0,
                        "step": 0.5,
                        "tooltip": "Extra bridge width used only in the occlusion zone",
                    },
                ),
                "max_samples": (
                    "INT",
                    {"default": 300000, "min": 100, "max": 2000000, "step": 1000},
                ),
            }
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "restore"
    CATEGORY = CATEGORY

    def restore(
        self,
        ai_image,
        reference_image,
        resize_reference,
        background_threshold,
        unchanged_threshold,
        boundary_guard,
        transition_width,
        structural_rescue,
        structural_threshold,
        structural_max_shift,
        exact_restore,
        exact_restore_threshold,
        small_hole_no_feather_area,
        seam_color_propagation,
        seam_max_distance,
        seam_decay,
        seam_residual_blur,
        seam_color_sigma,
        seam_bridge_width,
        occlusion_residual_blur,
        occlusion_bridge_width,
        max_samples,
    ):
        return _run_batch(
            ai_image,
            reference_image,
            resize_reference=resize_reference,
            max_samples=max_samples,
            background_threshold=(
                None if background_threshold <= 0 else background_threshold
            ),
            unchanged_threshold=(
                None if unchanged_threshold <= 0 else unchanged_threshold
            ),
            boundary_guard=boundary_guard,
            transition=transition_width,
            structural_rescue=structural_rescue,
            structural_threshold=structural_threshold,
            structural_max_shift=structural_max_shift,
            hard_restore_high_confidence=exact_restore,
            hard_restore_threshold=exact_restore_threshold,
            small_hole_no_feather_area=small_hole_no_feather_area,
            seam_color_propagation=seam_color_propagation,
            seam_max_distance=seam_max_distance,
            seam_decay=seam_decay,
            seam_residual_blur=seam_residual_blur,
            seam_color_sigma=seam_color_sigma,
            seam_bridge_width=seam_bridge_width,
            occlusion_residual_blur=occlusion_residual_blur,
            occlusion_bridge_width=occlusion_bridge_width,
        )


NODE_CLASS_MAPPINGS = {
    "CCROcclusionColorRestore": ReferenceColorRestoreOcclusion,
    "CCROcclusionColorRestoreAdvanced": ReferenceColorRestoreOcclusionAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCROcclusionColorRestore": "Reference Color Restore (Occlusion Seam)",
    "CCROcclusionColorRestoreAdvanced": (
        "Reference Color Restore (Occlusion Seam Advanced)"
    ),
}
