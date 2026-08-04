"""Core reference-aware color restoration algorithm.

This module is deliberately independent from ComfyUI so it can be tested and
reused by other node packs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RestorationReport:
    background_rgb: tuple[int, int, int]
    foreground_fraction: float
    unchanged_fraction: float
    threshold: float
    before_mae: float
    mapped_mae: float
    structural_rescued_fraction: float
    hard_restored_fraction: float
    seam_compensated_fraction: float
    occlusion_core_fraction: float = 0.0
    occlusion_zone_fraction: float = 0.0
    occlusion_seam_compensated_fraction: float = 0.0


def _features(rgb: np.ndarray, quadratic: bool) -> np.ndarray:
    """Return [1, r, g, b, r2, g2, b2, rg, rb, gb]."""
    r, g, b = rgb.T
    columns = [np.ones(len(rgb)), r, g, b]
    if quadratic:
        columns.extend([r * r, g * g, b * b, r * g, r * b, g * b])
    return np.column_stack(columns)


def _solve_weighted(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    quadratic: bool,
    ridge: float,
) -> np.ndarray:
    design = _features(source, quadratic)
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))[:, None]
    a = design * sqrt_w
    y = target * sqrt_w
    regularizer = np.eye(design.shape[1])
    regularizer[0, 0] = 0.0
    identity = np.zeros((design.shape[1], 3))
    identity[1:4, :] = np.eye(3)
    return np.linalg.solve(
        a.T @ a + ridge * regularizer,
        a.T @ y + ridge * regularizer @ identity,
    )


def _robust_fit(
    source: np.ndarray,
    target: np.ndarray,
    quadratic: bool,
    base_weights: np.ndarray | None = None,
    iterations: int = 8,
) -> np.ndarray:
    """Fit with Tukey IRLS so genuinely changed pixels receive zero weight."""
    if base_weights is None:
        base_weights = np.ones(len(source), dtype=np.float64)
    weights = base_weights.copy()
    ridge = 10.0 if quadratic else 1.0
    transform = np.zeros((10 if quadratic else 4, 3))
    for _ in range(iterations):
        transform = _solve_weighted(source, target, weights, quadratic, ridge)
        residual = np.linalg.norm(
            _features(source, quadratic) @ transform - target, axis=1
        )
        residual *= 255.0
        active = residual[weights > 0.05 * np.mean(base_weights)]
        if len(active) == 0:
            raise RuntimeError("Robust fitting rejected every sample")
        median = float(np.median(active))
        mad = 1.4826 * float(np.median(np.abs(active - median)))
        scale = max(2.0, median + 2.0 * mad)
        u = residual / (4.685 * scale)
        robust = np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0)
        weights = base_weights * robust
    return transform


def _apply_transform(
    image: np.ndarray, transform: np.ndarray, quadratic: bool
) -> np.ndarray:
    height, width, _ = image.shape
    result = np.empty_like(image)
    rows_per_strip = max(1, 1_000_000 // width)
    for y0 in range(0, height, rows_per_strip):
        y1 = min(height, y0 + rows_per_strip)
        pixels = image[y0:y1].reshape(-1, 3)
        result[y0:y1] = (_features(pixels, quadratic) @ transform).reshape(
            y1 - y0, width, 3
        )
    return np.clip(result, 0.0, 1.0)


def _uniform_sample(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = source.shape[:2]
    stride = max(1, math.ceil(math.sqrt(height * width / max_samples)))
    sampled_mask = mask[::stride, ::stride]
    return source[::stride, ::stride][sampled_mask], target[::stride, ::stride][
        sampled_mask
    ]


def _balanced_color_weights(samples: np.ndarray, bins: int = 8) -> np.ndarray:
    index = np.clip((samples * bins).astype(np.int32), 0, bins - 1)
    flat = (index[:, 0] * bins + index[:, 1]) * bins + index[:, 2]
    counts = np.bincount(flat, minlength=bins**3)
    weights = 1.0 / np.sqrt(counts[flat] + 1.0)
    weights /= np.mean(weights)
    return np.clip(weights, 0.25, 4.0)


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    inverse = (~mask).astype(np.uint8)
    count, labels = cv2.connectedComponents(inverse, 8)
    if count <= 1:
        return mask
    border_labels = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    border_labels = border_labels[border_labels != 0]
    exterior_background = np.isin(labels, border_labels)
    enclosed_background = (inverse > 0) & ~exterior_background
    return mask | enclosed_background


def segment_solid_background(
    image: np.ndarray,
    threshold: float | None = None,
    border_width: int = 20,
    min_component_fraction: float = 0.0005,
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    """Find sizeable foreground islands on an approximately solid background."""
    u8_rgb = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    border_width = max(1, min(border_width, height // 4, width // 4))
    border = np.concatenate(
        [
            u8_rgb[:border_width].reshape(-1, 3),
            u8_rgb[-border_width:].reshape(-1, 3),
            u8_rgb[:, :border_width].reshape(-1, 3),
            u8_rgb[:, -border_width:].reshape(-1, 3),
        ]
    )
    background = np.median(border, axis=0).astype(np.uint8)
    lab = cv2.cvtColor(u8_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(background[None, None, :], cv2.COLOR_RGB2LAB)[0, 0]
    distance = np.linalg.norm(lab - background_lab, axis=2)
    if threshold is None:
        border_distance = np.concatenate(
            [
                distance[:border_width].ravel(),
                distance[-border_width:].ravel(),
                distance[:, :border_width].ravel(),
                distance[:, -border_width:].ravel(),
            ]
        )
        quiet = border_distance[border_distance <= np.quantile(border_distance, 0.70)]
        median = float(np.median(quiet))
        mad = 1.4826 * float(np.median(np.abs(quiet - median)))
        threshold = float(np.clip(max(8.0, median + 6.0 * mad), 8.0, 20.0))

    raw = (distance > threshold).astype(np.uint8) * 255
    raw = cv2.morphologyEx(
        raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    foreground = np.zeros((height, width), dtype=bool)
    minimum_area = max(100, int(height * width * min_component_fraction))
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            foreground |= labels == label
    foreground = _fill_binary_holes(foreground)
    fraction = float(np.mean(foreground))
    if fraction < 0.005 or fraction > 0.80:
        raise RuntimeError(
            f"Implausible solid-background foreground fraction ({100*fraction:.2f}%); "
            "adjust the background threshold or verify that AI image is connected first"
        )
    return foreground, tuple(int(x) for x in background), float(threshold)


def _best_local_zncc(
    source: np.ndarray,
    reference: np.ndarray,
    patch_size: int = 13,
    max_shift: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_gray = cv2.cvtColor(source.astype(np.float32), cv2.COLOR_RGB2GRAY) * 255.0
    reference_gray = (
        cv2.cvtColor(reference.astype(np.float32), cv2.COLOR_RGB2GRAY) * 255.0
    )
    source_gray = cv2.GaussianBlur(source_gray, (0, 0), 0.7)
    reference_gray = cv2.GaussianBlur(reference_gray, (0, 0), 0.7)
    mean_source = cv2.boxFilter(
        source_gray,
        -1,
        (patch_size, patch_size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    variance_source = np.maximum(
        cv2.boxFilter(
            source_gray * source_gray,
            -1,
            (patch_size, patch_size),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        - mean_source * mean_source,
        1e-3,
    )
    best = np.full(source_gray.shape, -1.0, dtype=np.float32)
    zero_shift = np.full(source_gray.shape, -1.0, dtype=np.float32)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(
                reference_gray,
                matrix,
                (reference_gray.shape[1], reference_gray.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            mean_reference = cv2.boxFilter(
                shifted,
                -1,
                (patch_size, patch_size),
                normalize=True,
                borderType=cv2.BORDER_REFLECT,
            )
            variance_reference = np.maximum(
                cv2.boxFilter(
                    shifted * shifted,
                    -1,
                    (patch_size, patch_size),
                    normalize=True,
                    borderType=cv2.BORDER_REFLECT,
                )
                - mean_reference * mean_reference,
                1e-3,
            )
            covariance = (
                cv2.boxFilter(
                    source_gray * shifted,
                    -1,
                    (patch_size, patch_size),
                    normalize=True,
                    borderType=cv2.BORDER_REFLECT,
                )
                - mean_source * mean_reference
            )
            correlation = covariance / np.sqrt(variance_source * variance_reference)
            if dx == 0 and dy == 0:
                zero_shift = np.clip(correlation, -1.0, 1.0)
            best = np.maximum(best, np.clip(correlation, -1.0, 1.0))
    return best, zero_shift, np.sqrt(variance_source)


def _propagate_seam_color(
    image: np.ndarray,
    base: np.ndarray,
    reference: np.ndarray,
    foreground: np.ndarray,
    unchanged: np.ndarray,
    max_distance: float,
    decay: float,
    residual_blur: float,
    color_sigma: float,
    bridge_width: float,
    region: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = (max_distance, decay, residual_blur, color_sigma, bridge_width)
    if any(value <= 0 for value in values):
        raise ValueError("Seam propagation parameters must be positive")
    trusted = unchanged.astype(bool)
    target = foreground.astype(bool) & ~trusted
    if region is not None:
        if region.shape != trusted.shape:
            raise ValueError("Seam propagation region must match image dimensions")
        region = region.astype(bool)
        target &= region
    if not np.any(trusted) or not np.any(target):
        return image, np.zeros_like(trusted)

    trusted_float = trusted.astype(np.float32)
    residual = (reference - base).astype(np.float32)
    denominator = cv2.GaussianBlur(
        trusted_float, (0, 0), residual_blur, borderType=cv2.BORDER_REFLECT
    )
    numerator = cv2.GaussianBlur(
        residual * trusted_float[:, :, None],
        (0, 0),
        residual_blur,
        borderType=cv2.BORDER_REFLECT,
    )
    smooth_residual = numerator / np.maximum(denominator[:, :, None], 1e-5)
    smooth_residual = np.clip(smooth_residual, -32.0 / 255.0, 32.0 / 255.0)

    distance, labels = cv2.distanceTransformWithLabels(
        (~trusted).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    max_label = int(labels.max())
    residual_lut = np.zeros((max_label + 1, 3), dtype=np.float32)
    lab_lut = np.zeros((max_label + 1, 3), dtype=np.float32)
    base_u8 = np.clip(np.rint(base * 255.0), 0, 255).astype(np.uint8)
    base_lab = cv2.cvtColor(base_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    trusted_labels = labels[trusted]
    residual_lut[trusted_labels] = smooth_residual[trusted]
    lab_lut[trusted_labels] = base_lab[trusted]
    nearest_residual = residual_lut[labels]
    nearest_lab = lab_lut[labels]
    lab_difference = np.linalg.norm(base_lab - nearest_lab, axis=2)
    color_weight = np.exp(-0.5 * (lab_difference / color_sigma) ** 2)
    distance_weight = np.exp(-distance / decay)
    cutoff = np.clip(1.0 - (distance / max_distance) ** 2, 0.0, 1.0) ** 2
    weight = color_weight * distance_weight * cutoff
    apply_mask = target & (distance < max_distance) & (weight > 0.01)

    corrected = image.copy()
    corrected[apply_mask] += nearest_residual[apply_mask] * weight[apply_mask, None]
    trusted_distance = cv2.distanceTransform(
        trusted.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    inner_weight = np.clip(1.0 - trusted_distance / bridge_width, 0.0, 1.0)
    inner_bridge = np.clip(base + smooth_residual, 0.0, 1.0)
    inner_mask = trusted & (inner_weight > 0.0)
    if region is not None:
        inner_mask &= region
    corrected[inner_mask] = (
        image[inner_mask] * (1.0 - inner_weight[inner_mask, None])
        + inner_bridge[inner_mask] * inner_weight[inner_mask, None]
    )
    return np.clip(corrected, 0.0, 1.0), apply_mask | inner_mask


def _detect_occlusion_region(
    edited: np.ndarray,
    reference: np.ndarray,
    mapped: np.ndarray,
    foreground: np.ndarray,
    unchanged: np.ndarray,
    unchanged_threshold: float,
    *,
    min_area_fraction: float = 0.001,
    influence_width: float = 48.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find a removed reference object and a local seam zone around it.

    A sizeable reference-only background component seeds the detection.  Only
    connected high-residual foreground pixels touching that component become
    direct occlusion.  A narrow low-frequency lightness-only halo is retained
    as indirect evidence.  The resulting zone selects alternate seam values;
    it does not change the old trusted mask or global color fit.
    """
    shape = foreground.shape
    direct = np.zeros(shape, dtype=bool)
    indirect = np.zeros(shape, dtype=bool)
    zone = np.zeros(shape, dtype=bool)
    raw_difference = np.max(np.abs(edited - reference), axis=2) * 255.0
    reference_only = (~foreground) & (
        raw_difference > max(48.0, 2.0 * unchanged_threshold)
    )
    background_area = max(1, int(np.sum(~foreground)))
    if np.sum(reference_only) / background_area >= 0.20:
        return direct, indirect, zone

    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(reference_only.astype(np.uint8), 8)
    )
    min_component_area = max(
        64, int(round(shape[0] * shape[1] * min_area_fraction))
    )
    reference_only_large = np.zeros(shape, dtype=bool)
    for component in range(1, component_count):
        if component_stats[component, cv2.CC_STAT_AREA] >= min_component_area:
            reference_only_large |= component_labels == component
    if not np.any(reference_only_large):
        return direct, indirect, zone

    direct_candidates = cv2.morphologyEx(
        (raw_difference > 48.0).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    _, direct_labels, _, _ = cv2.connectedComponentsWithStats(
        direct_candidates.astype(np.uint8), 8
    )
    touching_labels = np.unique(direct_labels[reference_only_large])
    touching_labels = touching_labels[touching_labels > 0]
    if touching_labels.size:
        direct = np.isin(direct_labels, touching_labels) & foreground
    if not np.any(direct):
        return direct, indirect, zone

    mapped_lab = cv2.cvtColor(
        np.clip(np.rint(mapped * 255.0), 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)
    reference_lab = cv2.cvtColor(
        np.clip(np.rint(reference * 255.0), 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)
    chroma_difference = cv2.GaussianBlur(
        np.linalg.norm(mapped_lab[:, :, 1:] - reference_lab[:, :, 1:], axis=2),
        (0, 0),
        2.0,
    )
    lightness_difference = cv2.GaussianBlur(
        mapped_lab[:, :, 0] - reference_lab[:, :, 0],
        (0, 0),
        3.0,
    )
    distance_to_direct = cv2.distanceTransform(
        (~direct).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    image_scale = float(min(shape))
    influence_reach = min(float(influence_width), max(8.0, 0.045 * image_scale))
    effect_candidates = (
        unchanged
        & ~direct
        & (distance_to_direct < influence_reach)
        & (np.abs(lightness_difference) > 4.0)
        & (np.abs(lightness_difference) < 48.0)
        & (chroma_difference < 18.0)
    )
    effect_candidates = cv2.morphologyEx(
        effect_candidates.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    effect_count, effect_labels, effect_stats, _ = (
        cv2.connectedComponentsWithStats(effect_candidates.astype(np.uint8), 8)
    )
    min_effect_area = max(32, int(round(shape[0] * shape[1] * 0.00010)))
    for component in range(1, effect_count):
        if effect_stats[component, cv2.CC_STAT_AREA] >= min_effect_area:
            indirect |= effect_labels == component

    core = direct | indirect
    if not np.any(core):
        return direct, indirect, zone
    radius = max(1, int(np.ceil(float(influence_width))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    zone = cv2.dilate(core.astype(np.uint8), kernel).astype(bool) & foreground
    return direct, indirect, zone


def restore_reference_colors(
    edited: np.ndarray,
    reference: np.ndarray,
    *,
    max_samples: int = 300_000,
    background_threshold: float | None = None,
    unchanged_threshold: float | None = None,
    structural_rescue: bool = True,
    structural_threshold: float = 0.95,
    structural_max_shift: int = 2,
    hard_restore_high_confidence: bool = True,
    hard_restore_threshold: float = 0.995,
    small_hole_no_feather_area: int = 16,
    seam_color_propagation: bool = True,
    seam_max_distance: float = 96.0,
    seam_decay: float = 64.0,
    seam_residual_blur: float = 12.0,
    seam_color_sigma: float = 20.0,
    seam_bridge_width: float = 12.0,
    occlusion_residual_blur: float = 0.0,
    occlusion_bridge_width: float = 0.0,
    boundary_guard: int = 4,
    transition: float = 6.0,
    return_occlusion_masks: bool = False,
) -> tuple:
    """Restore aligned content while preserving newly generated regions."""
    edited = np.asarray(edited, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if edited.shape != reference.shape:
        raise ValueError(f"Image shapes differ: {edited.shape} != {reference.shape}")
    if edited.ndim != 3 or edited.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB images, got {edited.shape}")
    if max_samples < 100:
        raise ValueError("max_samples must be at least 100")
    foreground, background_rgb, _ = segment_solid_background(
        edited, threshold=background_threshold
    )
    guard_size = max(1, 2 * boundary_guard + 1)
    interior = cv2.erode(
        foreground.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard_size, guard_size)),
    ).astype(bool)
    sample_edited, sample_reference = _uniform_sample(
        edited, reference, interior, max_samples
    )
    if len(sample_edited) < 100:
        raise RuntimeError("Too few foreground correspondences for color correction")
    transform = _robust_fit(
        sample_edited,
        sample_reference,
        quadratic=True,
        base_weights=_balanced_color_weights(sample_edited),
        iterations=10,
    )
    mapped = _apply_transform(edited, transform, quadratic=True)
    residual = np.max(np.abs(mapped - reference), axis=2) * 255.0
    local_rms = np.sqrt(
        cv2.GaussianBlur((residual * residual).astype(np.float32), (0, 0), 1.5)
    )
    if unchanged_threshold is None:
        values = local_rms[interior]
        quiet = values[values <= np.quantile(values, 0.70)]
        median = float(np.median(quiet))
        mad = 1.4826 * float(np.median(np.abs(quiet - median)))
        unchanged_threshold = float(
            np.clip(max(16.0, median + 6.0 * mad), 16.0, 32.0)
        )

    unchanged = interior & (local_rms < unchanged_threshold)
    unchanged = cv2.morphologyEx(
        unchanged.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    baseline_unchanged = unchanged.copy()
    correlation = zero_shift = texture = chroma_difference = None
    if structural_rescue or hard_restore_high_confidence:
        correlation, zero_shift, texture = _best_local_zncc(
            mapped, reference, patch_size=13, max_shift=structural_max_shift
        )
        mapped_lab = cv2.cvtColor(
            np.clip(np.rint(mapped * 255.0), 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2LAB,
        ).astype(np.float32)
        reference_lab = cv2.cvtColor(
            np.clip(np.rint(reference * 255.0), 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2LAB,
        ).astype(np.float32)
        chroma_difference = np.linalg.norm(
            mapped_lab[:, :, 1:] - reference_lab[:, :, 1:], axis=2
        )
        chroma_difference = cv2.GaussianBlur(chroma_difference, (0, 0), 2.0)
        if structural_rescue:
            candidates = (
                interior
                & ~unchanged
                & (correlation >= structural_threshold)
                & (texture >= 4.0)
                & (local_rms < 64.0)
                & (chroma_difference < 20.0)
            )
            for _ in range(2):
                support = cv2.dilate(
                    unchanged.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                ).astype(bool)
                unchanged |= candidates & support

    relative_unchanged = float(np.sum(unchanged) / max(1, np.sum(foreground)))
    if relative_unchanged < 0.20:
        raise RuntimeError(
            f"Only {100*relative_unchanged:.1f}% of foreground is confidently unchanged; "
            "refusing reference restoration"
        )
    foreground_distance = cv2.distanceTransform(
        foreground.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    feather_mask = unchanged.copy()
    tiny_uncertain = np.zeros_like(unchanged)
    if small_hole_no_feather_area > 0:
        uncertain = interior & ~unchanged
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            uncertain.astype(np.uint8), 8
        )
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] <= small_hole_no_feather_area:
                component = labels == label
                tiny_uncertain |= component
                feather_mask |= component
    unchanged_distance = cv2.distanceTransform(
        feather_mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    color_weight = np.clip(foreground_distance / 3.0, 0.0, 1.0)
    reference_weight = np.clip(
        unchanged_distance / max(1.0, transition), 0.0, 1.0
    )
    reference_weight[tiny_uncertain] = 0.0
    hard_restore = np.zeros_like(unchanged)
    if hard_restore_high_confidence:
        hard_restore = (
            unchanged
            & (zero_shift >= hard_restore_threshold)
            & (texture >= 4.0)
            & (chroma_difference < 12.0)
            & (local_rms < 48.0)
        )
        reference_weight[hard_restore] = 1.0
    base = edited * (1.0 - color_weight[:, :, None]) + mapped * color_weight[:, :, None]
    result = base * (1.0 - reference_weight[:, :, None]) + reference * reference_weight[
        :, :, None
    ]
    occlusion_core = np.zeros_like(unchanged)
    occlusion_zone = np.zeros_like(unchanged)
    occlusion_seam_compensated = np.zeros_like(unchanged)
    if occlusion_residual_blur > 0 and occlusion_bridge_width > 0:
        direct_mask, indirect_mask, occlusion_zone = _detect_occlusion_region(
            edited,
            reference,
            mapped,
            foreground,
            unchanged,
            float(unchanged_threshold),
            influence_width=occlusion_bridge_width,
        )
        occlusion_core = direct_mask | indirect_mask

    seam_compensated = np.zeros_like(unchanged)
    if seam_color_propagation:
        seam_input = result
        global_result, global_compensated = _propagate_seam_color(
            result,
            base,
            reference,
            foreground,
            unchanged,
            seam_max_distance,
            seam_decay,
            seam_residual_blur,
            seam_color_sigma,
            seam_bridge_width,
        )
        result = global_result
        seam_compensated = global_compensated
        if np.any(occlusion_zone):
            occlusion_result, occlusion_compensated = _propagate_seam_color(
                seam_input,
                base,
                reference,
                foreground,
                unchanged,
                seam_max_distance,
                seam_decay,
                occlusion_residual_blur,
                seam_color_sigma,
                occlusion_bridge_width,
                region=occlusion_zone,
            )
            local_selection = occlusion_compensated & occlusion_zone
            result = np.where(
                local_selection[:, :, None], occlusion_result, global_result
            )
            seam_compensated = global_compensated | local_selection
            occlusion_seam_compensated = local_selection
    report = RestorationReport(
        background_rgb=background_rgb,
        foreground_fraction=float(np.mean(foreground)),
        unchanged_fraction=float(np.mean(unchanged)),
        threshold=float(unchanged_threshold),
        before_mae=float(
            np.mean(np.abs(edited[unchanged] - reference[unchanged])) * 255.0
        ),
        mapped_mae=float(
            np.mean(np.abs(mapped[unchanged] - reference[unchanged])) * 255.0
        ),
        structural_rescued_fraction=float(
            np.sum(unchanged & ~baseline_unchanged) / max(1, np.sum(foreground))
        ),
        hard_restored_fraction=float(
            np.sum(hard_restore) / max(1, np.sum(foreground))
        ),
        seam_compensated_fraction=float(
            np.sum(seam_compensated) / max(1, np.sum(foreground))
        ),
        occlusion_core_fraction=float(
            np.sum(occlusion_core & foreground) / max(1, np.sum(foreground))
        ),
        occlusion_zone_fraction=float(
            np.sum(occlusion_zone & foreground) / max(1, np.sum(foreground))
        ),
        occlusion_seam_compensated_fraction=float(
            np.sum(occlusion_seam_compensated & foreground)
            / max(1, np.sum(foreground))
        ),
    )
    output = np.clip(result, 0.0, 1.0)
    if return_occlusion_masks:
        return (
            output,
            foreground,
            unchanged,
            report,
            occlusion_core,
            occlusion_zone,
            occlusion_seam_compensated,
        )
    return output, foreground, unchanged, report
