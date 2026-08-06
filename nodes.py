"""ComfyUI nodes based on Reference Color Restore with local occlusion seams."""

from __future__ import annotations

from dataclasses import asdict

import cv2
import numpy as np
import torch

from .algorithm import RestorationReport, restore_reference_colors


CATEGORY = "image/color correction"
SEAM_HEATMAP_CEILING_RGB_PER_PIXEL = 2.0


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


def _generated_correction_edge_heatmap(
    corrected: np.ndarray, base: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Render hard edges in the algorithm's correction field, not final image edges.

    The fixed 2 RGB-levels/pixel ceiling deliberately matches the earlier seam
    diagnostics, so different V1 runs remain directly comparable.
    """
    correction_luminance = np.mean(corrected - base, axis=2).astype(np.float32)
    edge_strength = np.hypot(
        cv2.Sobel(correction_luminance, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(correction_luminance, cv2.CV_32F, 0, 1, ksize=3),
    ) * 255.0
    levels = np.uint8(
        np.clip(
            edge_strength / SEAM_HEATMAP_CEILING_RGB_PER_PIXEL * 255.0,
            0.0,
            255.0,
        )
    )
    bgr = cv2.applyColorMap(levels, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return heatmap, edge_strength


def _format_report(
    index: int,
    report: RestorationReport,
    resized: bool,
    *,
    include_component_handoff: bool = False,
) -> str:
    values = asdict(report)
    rows = [
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
    if include_component_handoff:
        rows.extend(
            [
                f"component_internal_handoff_foreground: "
                f"{100 * values['component_handoff_fraction']:.2f}%",
            ]
        )
    return "\n".join(rows)


def _run_batch(
    ai_image: torch.Tensor,
    reference_image: torch.Tensor,
    *,
    resize_reference: bool,
    return_seam_heatmap: bool = False,
    include_component_handoff_report: bool = False,
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
    seam_heatmap_items = []
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
            restored = restore_reference_colors(
                edited,
                reference,
                return_occlusion_masks=True,
                return_diagnostics=return_seam_heatmap,
                **options,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Reference Color Restore (Occlusion) failed at batch item {index}: {exc}"
            ) from exc
        if return_seam_heatmap:
            (
                corrected,
                foreground,
                unchanged,
                report,
                occlusion_core,
                occlusion_zone,
                occlusion_compensated,
                diagnostics,
            ) = restored
            seam_heatmap, edge_strength = _generated_correction_edge_heatmap(
                corrected, diagnostics["base"]
            )
            seam_heatmap_items.append(seam_heatmap)
            foreground_strength = edge_strength[foreground]
            edge_summary = (
                "\ngenerated_correction_edge_p95: "
                f"{np.percentile(foreground_strength, 95):.3f} RGB/px "
                f"(heatmap max={SEAM_HEATMAP_CEILING_RGB_PER_PIXEL:.0f} RGB/px)"
                if foreground_strength.size
                else "\ngenerated_correction_edge_p95: n/a"
            )
        else:
            (
                corrected,
                foreground,
                unchanged,
                report,
                occlusion_core,
                occlusion_zone,
                occlusion_compensated,
            ) = restored
            edge_summary = ""
        corrected_items.append(corrected.astype(np.float32))
        foreground_items.append(foreground.astype(np.float32))
        unchanged_items.append(unchanged.astype(np.float32))
        occlusion_core_items.append(occlusion_core.astype(np.float32))
        occlusion_zone_items.append(occlusion_zone.astype(np.float32))
        occlusion_compensated_items.append(occlusion_compensated.astype(np.float32))
        reports.append(
            _format_report(
                index,
                report,
                resized,
                include_component_handoff=include_component_handoff_report,
            )
            + edge_summary
        )

    outputs = (
        torch.from_numpy(np.stack(corrected_items)),
        torch.from_numpy(np.stack(foreground_items)),
        torch.from_numpy(np.stack(unchanged_items)),
        torch.from_numpy(np.stack(occlusion_core_items)),
        torch.from_numpy(np.stack(occlusion_zone_items)),
        torch.from_numpy(np.stack(occlusion_compensated_items)),
        "\n\n".join(reports),
    )
    if return_seam_heatmap:
        return (*outputs, torch.from_numpy(np.stack(seam_heatmap_items)))
    return outputs


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

    TRUSTED_FIT = False
    STRUCTURAL_UNCHANGED_CLEANUP = False
    CONTINUOUS_SEAM_FIELD = False
    INCLUDE_SEAM_HEATMAP = False

    DESCRIPTION = (
        "V0 Advanced Reference Color Restore. seam_residual_blur and seam_bridge_width "
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
        component_internal_handoff=False,
        seam_handoff_smoothing=False,
        seam_handoff_silhouette_guard=3.0,
    ):
        return _run_batch(
            ai_image,
            reference_image,
            resize_reference=resize_reference,
            return_seam_heatmap=self.INCLUDE_SEAM_HEATMAP,
            include_component_handoff_report=self.INCLUDE_SEAM_HEATMAP,
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
            component_internal_handoff=component_internal_handoff,
            seam_handoff_smoothing=seam_handoff_smoothing,
            seam_handoff_silhouette_guard=seam_handoff_silhouette_guard,
            continuous_seam_field=self.CONTINUOUS_SEAM_FIELD,
            trusted_fit=self.TRUSTED_FIT,
            structural_unchanged_cleanup=self.STRUCTURAL_UNCHANGED_CLEANUP,
        )


class ReferenceColorRestoreOcclusionAdvancedV1(
    ReferenceColorRestoreOcclusionAdvanced
):
    """Advanced node with guarded trusted fitting and structural mask cleanup."""

    TRUSTED_FIT = True
    STRUCTURAL_UNCHANGED_CLEANUP = True
    CONTINUOUS_SEAM_FIELD = True
    INCLUDE_SEAM_HEATMAP = True
    RETURN_TYPES = RETURN_TYPES + ("IMAGE",)
    RETURN_NAMES = RETURN_NAMES + ("generated_correction_edge_heatmap",)
    DESCRIPTION = (
        "V1 uses only high-confidence aligned pixels for a validated affine color "
        "fit, then removes unchanged regions that lack multi-scale structural "
        "agreement. Its final heatmap output shows only sharp edges in the "
        "algorithm's correction field (Turbo; red is >=2 RGB/px). The optional "
        "seam-handoff smoothing patch keeps the outer silhouette fixed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        required = dict(types["required"])
        required["component_internal_handoff"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": "Smooth a qualifying independent unchanged island internally while preserving the existing continuous-field result outside it",
            },
        )
        required["seam_handoff_smoothing"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": "Optional local repair for a continuous-field color-weight handoff. It uses the existing V1 field as the fixed boundary and is off by default.",
            },
        )
        required["seam_handoff_silhouette_guard"] = (
            "FLOAT",
            {
                "default": 3.0,
                "min": 0.0,
                "max": 32.0,
                "step": 0.5,
                "tooltip": "Inner foreground-silhouette band (pixels) held exactly at the current V1 correction while seam_handoff_smoothing is enabled. 3px is the reviewed default.",
            },
        )
        return {"required": required}


class ReferenceColorRestoreOcclusionAdvancedV08(
    ReferenceColorRestoreOcclusionAdvancedV1
):
    """Advanced V1 with only structural unchanged-mask cleanup disabled.

    This is a direct V1 comparison node.  It retains V1's interface,
    trusted-region affine fit, continuous seam field, correction-edge heatmap,
    and optional component/seam handoff controls.  The class-level override
    below is intentionally its sole algorithmic difference from V1.
    """

    STRUCTURAL_UNCHANGED_CLEANUP = False
    DESCRIPTION = (
        "V0.8 is identical to Advanced V1 except that structural cleanup of "
        "the safe_unchanged mask is disabled for direct comparison."
    )


NODE_CLASS_MAPPINGS = {
    "CCROcclusionColorRestore": ReferenceColorRestoreOcclusion,
    "CCROcclusionColorRestoreAdvanced": ReferenceColorRestoreOcclusionAdvanced,
    "CCROcclusionColorRestoreAdvancedV08": ReferenceColorRestoreOcclusionAdvancedV08,
    "CCROcclusionColorRestoreAdvancedV1": ReferenceColorRestoreOcclusionAdvancedV1,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCROcclusionColorRestore": "Reference Color Restore (Occlusion Seam)",
    "CCROcclusionColorRestoreAdvanced": (
        "Reference Color Restore (Occlusion Seam Advanced) V0"
    ),
    "CCROcclusionColorRestoreAdvancedV08": (
        "Reference Color Restore (Occlusion Seam Advanced) V0.8"
    ),
    "CCROcclusionColorRestoreAdvancedV1": (
        "Reference Color Restore (Occlusion Seam Advanced) V1"
    ),
}
