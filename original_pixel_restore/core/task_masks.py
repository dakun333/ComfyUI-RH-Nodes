"""Task-specific masks without final color harmonization or semantic models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import cv2
import numpy as np
from scipy import ndimage


def _gray(array):
    return cv2.cvtColor(array.astype(np.float32), cv2.COLOR_RGB2GRAY)


def _blur(array, sigma):
    return cv2.GaussianBlur(
        array, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101
    )


def _components(mask):
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)


def _distance(mask):
    return (
        ndimage.distance_transform_edt(~mask)
        if mask.any()
        else np.full(mask.shape, np.inf)
    )


def _mask(value, shape, name):
    if value is None:
        return np.zeros(shape, bool)
    array = np.asarray(value)
    if (
        array.shape != shape
        or not np.isfinite(array).all()
        or array.min() < 0
        or array.max() > 1
    ):
        raise ValueError(
            f"{name} must be finite [0,1] at original image resolution"
        )
    return array > 0.5


@dataclass
class LargeMaskConfig:
    high: float = 4.0
    low: float = 2.0
    min_area_fraction: float = 0.001
    min_radius: float = 4.0
    close_radius: int = 5
    fill_holes: bool = True
    padding: int = 8
    feather: int = 24
    exclude_expand: int = 32
    alignment: str = "global"


def _fit_detection_color(original, generated, safe):
    indices = np.flatnonzero(safe)
    if len(indices) < 100:
        raise ValueError(
            "Fewer than 100 shared-background candidates; supply/revise a mask."
        )
    indices = indices[
        np.linspace(0, len(indices) - 1, min(80000, len(indices)), dtype=int)
    ]
    generated_samples = generated.reshape(-1, 3)[indices].astype(np.float64) / 255
    original_samples = original.reshape(-1, 3)[indices].astype(np.float64) / 255
    design = np.c_[generated_samples, np.ones(len(generated_samples))]
    condition = float(np.linalg.cond(design))
    offset_only = (
        np.linalg.matrix_rank(design) < 4
        or not np.isfinite(condition)
        or condition > 1e5
    )
    if offset_only:
        coefficients = np.vstack(
            [np.eye(3), np.median(original_samples - generated_samples, axis=0)]
        )
        model = "offset_fallback"
    else:
        coefficients = np.vstack(
            [np.eye(3), np.median(original_samples - generated_samples, axis=0)]
        )
        for _ in range(12):
            residual = np.sqrt(
                np.mean((design @ coefficients - original_samples) ** 2, axis=1)
            )
            weights = np.sqrt(
                np.minimum(1, (3 / 255) / np.maximum(residual, 1e-10))
            )
            coefficients = np.linalg.lstsq(
                design * weights[:, None],
                original_samples * weights[:, None],
                rcond=None,
            )[0]
        if (
            not np.isfinite(coefficients).all()
            or np.linalg.norm(coefficients[:3] - np.eye(3)) > 2
        ):
            coefficients = np.vstack(
                [np.eye(3), np.median(original_samples - generated_samples, axis=0)]
            )
            model = "offset_guard"
        else:
            model = "robust_rgb_affine"
    result = np.clip(
        (generated.astype(np.float64) / 255 @ coefficients[:3] + coefficients[3])
        * 255,
        0,
        255,
    ).astype(np.float32)
    return result, {
        "model": model,
        "coefficients": coefficients.tolist(),
        "fit_pixels": int(len(indices)),
        "design_condition": condition if np.isfinite(condition) else None,
        "used_for_detection_only": True,
    }


def _large_evidence(original, corrected, safe):
    residual = original.astype(np.float32) - corrected
    fine = np.sqrt(np.mean(_blur(residual, 0.8) ** 2, axis=2))
    broad = np.sqrt(np.mean(_blur(residual, 2.5) ** 2, axis=2))
    gray = _gray(corrected)
    edge = _blur(
        np.hypot(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8,
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8,
        ),
        1,
    )
    values = fine[safe]
    noise = max(1, float(np.quantile(values, 0.35)) * 1.6) if len(values) else 1
    score = np.maximum(fine / (noise + 0.20 * edge), broad / (noise + 0.10 * edge))
    return score.astype(np.float32), {"noise_255": noise}


def _large_components(score, valid, config):
    possible = (score >= config.low) & valid
    seeds = (score >= config.high) & valid
    if config.close_radius:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * config.close_radius + 1,) * 2,
        )
        padding = config.close_radius * 2
        padded = np.pad(possible.astype(np.uint8), padding, mode="edge")
        possible = (
            cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)[
                padding:-padding, padding:-padding
            ].astype(bool)
            & valid
        )
    count, labels, stats, _ = _components(possible)
    touched = np.zeros(count, bool)
    touched[np.unique(labels[seeds])] = True
    touched[0] = False
    radii = ndimage.maximum(
        cv2.distanceTransform(possible.astype(np.uint8), cv2.DIST_L2, 5),
        labels,
        np.arange(count),
    )
    keep = (
        stats[:, cv2.CC_STAT_AREA]
        >= max(64, config.min_area_fraction * score.size)
    ) & (radii >= config.min_radius) & touched
    keep[0] = False
    return keep[labels]


def large_object_mask(
    original, edited, config=None, include_mask=None, protect_mask=None
):
    """Produce inspectable candidate masks on the original image grid."""
    from .pipeline import global_align

    started = time.perf_counter()
    config = config or LargeMaskConfig()
    values = [
        config.high,
        config.low,
        config.min_area_fraction,
        config.min_radius,
        config.close_radius,
        config.padding,
        config.feather,
        config.exclude_expand,
    ]
    if (
        not np.isfinite(values).all()
        or not 0 < config.low <= config.high
        or not 0 <= config.min_area_fraction <= 1
        or config.min_radius < 0
        or min(values[4:]) < 0
    ):
        raise ValueError("Invalid large-mask thresholds, area, radius or margins")
    for name in ("close_radius", "padding", "feather", "exclude_expand"):
        if int(getattr(config, name)) != getattr(config, name):
            raise ValueError(f"{name} must be an integer")
    original = np.asarray(original, dtype=np.float32)
    edited = np.asarray(edited, dtype=np.float32)
    for name, array in (("original", original), ("edited", edited)):
        if (
            array.ndim != 3
            or array.shape[-1] != 3
            or min(array.shape[:2]) < 16
            or not np.isfinite(array).all()
            or array.min() < 0
            or array.max() > 255
        ):
            raise ValueError(
                f"{name} must be finite RGB HxWx3 in 0..255, minimum 16x16"
            )
    shape = original.shape[:2]
    include = _mask(include_mask, shape, "include_mask")
    protect = _mask(protect_mask, shape, "protect_mask")
    if config.alignment == "global":
        generated, valid, registration = global_align(original, edited)
    elif config.alignment == "none":
        if original.shape != edited.shape:
            raise ValueError("alignment=none requires matching dimensions")
        generated = edited.copy()
        valid = np.ones(shape, bool)
        registration = {"method": "none"}
    else:
        raise ValueError("alignment must be global or none")
    clipped_samples = int(np.count_nonzero((generated < 0) | (generated > 255)))
    generated = np.clip(generated, 0, 255).astype(np.float32)
    include &= valid & ~protect
    core = include.copy()
    warnings, fit, noise = [], {}, {}
    score = np.zeros(shape, np.float32)
    for _ in range(3):
        safe = valid & (_distance(core) > config.exclude_expand)
        if safe.sum() < max(100, 0.08 * core.size):
            warnings.append(
                "Insufficient background for another detection fit; retained preceding estimate."
            )
            break
        corrected, fit = _fit_detection_color(original, generated, safe)
        score, noise = _large_evidence(original, corrected, safe)
        proposed = (
            _large_components(score, valid & ~protect, config) | include
        ) & ~protect
        if np.array_equal(proposed, core):
            core = proposed
            break
        core = proposed
    before_fill = core.copy()
    if config.fill_holes:
        core = ndimage.binary_fill_holes(core) & valid & ~protect
    core = (core | include) & valid & ~protect
    distance = _distance(core)
    if config.feather:
        transition = np.clip(
            (distance - config.padding) / config.feather, 0, 1
        )
        alpha = 1 - transition**3 * (10 + transition * (-15 + 6 * transition))
    else:
        alpha = (distance <= config.padding).astype(np.float64)
    alpha[~valid | protect] = 0
    alpha[core] = 1
    support = alpha > 0
    exclude = (
        distance
        <= max(config.exclude_expand, config.padding + config.feather)
    ) | ~valid
    review = (
        (core & ~before_fill)
        | ((score >= config.low * 0.65) & (score < config.high))
        | (support & ~core)
    ) & valid & ~protect
    if core.mean() > 0.4:
        warnings.append(
            "Large candidate (>40%): color/content ambiguity; inspect fit and masks."
        )
    if not core.any():
        warnings.append("No thick edit found. This node intentionally ignores thin edits.")
    if (~valid).any():
        warnings.append(
            "Invalid alignment borders are preserved and excluded from fitting."
        )
    if include.any() and include_mask is not None:
        warnings.append("Include mask is user guidance, not inferred ground truth.")
    warnings.append(
        "Paired differences are not semantic truth; inspect same-color objects, "
        "shadows and AI redraws."
    )
    return {
        "edit_mask": core,
        "exclude_mask": exclude,
        "blend_mask": alpha.astype(np.float32),
        "support_mask": support,
        "aligned_edited": generated,
        "evidence": np.clip(score / max(config.high, 1e-6), 0, 1),
        "review_mask": review,
        "stats": {
            "version": "task_split_v1",
            "seconds": time.perf_counter() - started,
            "config": asdict(config),
            "registration": registration,
            "alignment_clipped_samples": clipped_samples,
            "color_fit": fit,
            "noise": noise,
            "warnings": warnings,
            "edit_fraction": float(core.mean()),
            "support_fraction": float(support.mean()),
            "filled_pixels": int((core & ~before_fill).sum()),
            "no_final_color_correction": True,
            "no_local_flow": True,
            "no_poisson": True,
        },
    }
