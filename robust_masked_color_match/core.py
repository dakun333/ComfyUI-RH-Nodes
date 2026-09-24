"""Paired-pixel robust RGB affine background matching."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def _unit_array(value, name, ndim):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != ndim or not a.size or any(n == 0 for n in a.shape):
        raise ValueError(f"{name}: expected nonempty {ndim}D array, got {a.shape}")
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError(
            f"{name}: values must be finite and in [0,1]; no automatic normalization"
        )
    return a


def _number(value, name, minimum=0, maximum=float("inf")):
    value = float(value)
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name}: must be finite in [{minimum}, {maximum}]")
    return value


def prepare_mask(mask, height, width, resize="error"):
    a = _unit_array(mask, "mask", 2)
    if resize not in ("error", "nearest"):
        raise ValueError("mask_resize must be 'error' or 'nearest'")
    resized = a.shape != (height, width)
    if resized:
        if resize == "error":
            raise ValueError(
                f"Mask is {a.shape[1]}x{a.shape[0]}, image is {width}x{height}. "
                "Align the mask first, or explicitly set mask_resize=nearest if it "
                "is only a size mismatch."
            )
        a = np.asarray(
            Image.fromarray(a.astype(np.float32)).resize(
                (width, height), Image.Resampling.NEAREST
            ),
            dtype=np.float64,
        )
    return a, resized


def outside_distance(binary):
    if not binary.any():
        return np.full(binary.shape, np.inf, dtype=np.float64)
    return distance_transform_edt(~binary)


def core_feather(mask, core_expand=24, feather_width=96, threshold=0.5):
    a = _unit_array(mask, "mask", 2)
    core_expand = _number(core_expand, "core_expand")
    feather_width = _number(feather_width, "feather_width", 0.001)
    threshold = _number(threshold, "threshold", 0.001, 1)
    binary = a >= threshold
    if not binary.any():
        return np.zeros_like(a)
    t = np.clip(
        (outside_distance(binary) - core_expand) / feather_width, 0, 1
    )
    return np.clip(1 - t**3 * (10 + t * (-15 + 6 * t)), 0, 1)


def _metrics(candidate, reference, region):
    if not region.any():
        return None
    diff = (candidate[region] - reference[region]) * 255
    return {
        "pixels": int(region.sum()),
        "mae_255": float(np.abs(diff).mean()),
        "rmse_255": float(np.sqrt(np.mean(diff**2))),
        "mean_rgb_255": diff.mean(axis=0).tolist(),
    }


def match_frame(
    reference,
    target,
    exclude_mask,
    *,
    strength=1.0,
    exclude_expand=32,
    mask_threshold=0.5,
    mask_resize="error",
    max_samples=100000,
    iterations=12,
    outlier_tolerance=3.0,
):
    reference = _unit_array(reference, "reference", 3)
    target = _unit_array(target, "target", 3)
    if reference.shape != target.shape or target.shape[-1] != 3:
        raise ValueError(
            "reference and target must be aligned, same-size [H,W,3] RGB images"
        )
    strength = _number(strength, "strength", 0, 1)
    exclude_expand = _number(exclude_expand, "exclude_expand")
    mask_threshold = _number(mask_threshold, "mask_threshold", 0.001, 1)
    outlier_tolerance = _number(
        outlier_tolerance, "outlier_tolerance", 0.001
    )
    if int(iterations) != iterations or not 1 <= iterations <= 100:
        raise ValueError("iterations must be an integer in [1,100]")
    if int(max_samples) != max_samples or max_samples < 100:
        raise ValueError("max_samples must be an integer >=100")

    h, w = target.shape[:2]
    mask, resized = prepare_mask(exclude_mask, h, w, mask_resize)
    binary = mask >= mask_threshold
    safe = outside_distance(binary) > exclude_expand
    warnings = []
    if resized:
        warnings.append(
            "Mask resized by nearest neighbor; this is not geometric registration."
        )
    if not binary.any():
        warnings.append(
            "Exclusion mask is empty. All content is eligible; check JPEG "
            "IMAGE-to-MASK wiring."
        )
    identity = np.vstack([np.eye(3), np.zeros(3)])
    report = {
        "algorithm": "paired robust RGB affine / float64 / encoded RGB",
        "image_size": [w, h],
        "strength": strength,
        "exclude_expand_px": exclude_expand,
        "mask_resized": resized,
        "safe_pixels": int(safe.sum()),
        "warnings": warnings,
    }
    if strength == 0:
        report.update(
            {
                "fit_model": "bypass",
                "coefficients_rgb_rowvector": identity.tolist(),
            }
        )
        return target.copy(), np.zeros((h, w), dtype=np.float32), report
    if safe.sum() < 100:
        raise ValueError(
            f"Only {safe.sum()} usable background pixels remain; need at least 100. "
            "White means EXCLUDE. Check mask polarity, reduce exclude_expand, or "
            "provide more shared background."
        )

    y, x = np.indices((h, w))
    train = safe & ((x // 32 + y // 32) % 2 == 0)
    holdout = safe & ~train
    if train.sum() < 100:
        train = safe.copy()
        holdout = np.zeros_like(safe)
        warnings.append(
            "Small sampling region: all safe pixels used, no independent holdout."
        )
    indices = np.flatnonzero(train)
    stride = min(7, max(1, indices.size // 100))
    indices = indices[::stride]
    if indices.size > max_samples:
        indices = indices[
            np.linspace(0, indices.size - 1, int(max_samples), dtype=np.int64)
        ]
    design_matrix = np.c_[target.reshape(-1, 3)[indices], np.ones(indices.size)]
    response_matrix = reference.reshape(-1, 3)[indices]
    rank = int(np.linalg.matrix_rank(design_matrix))
    condition = float(np.linalg.cond(design_matrix))
    model = "rgb_affine"
    design, response = design_matrix, response_matrix
    if rank < 4 or not np.isfinite(condition) or condition > 100000:
        model = "rgb_offset_fallback"
        design = np.ones((indices.size, 1))
        response = response_matrix - design_matrix[:, :3]
        warnings.append(
            "Insufficient color diversity for affine fit; using robust RGB offset only."
        )
    beta = np.linalg.lstsq(design, response, rcond=None)[0]
    for _ in range(int(iterations)):
        residual = np.sqrt(np.mean((design @ beta - response) ** 2, axis=1))
        weights = np.minimum(
            1, (outlier_tolerance / 255) / np.maximum(residual, 1e-8)
        ) ** 0.5
        beta = np.linalg.lstsq(
            design * weights[:, None], response * weights[:, None], rcond=None
        )[0]
    coefficients = (
        beta if model == "rgb_affine" else np.vstack([np.eye(3), beta])
    )
    if not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite color fit; check input content and alignment")
    fitted = target @ coefficients[:3] + coefficients[3]
    clipped_fraction = float(np.mean((fitted < 0) | (fitted > 1)))
    result = target + strength * (np.clip(fitted, 0, 1) - target)
    before = _metrics(target, reference, holdout)
    after = _metrics(result, reference, holdout)
    if before and after["mae_255"] > before["mae_255"] * 1.2 + 0.25:
        warnings.append(
            "Held-out background error increased. Check image alignment, mask, "
            "and global-fit suitability."
        )
    report.update(
        {
            "fit_model": model,
            "fit_samples": int(indices.size),
            "training_region_pixels": int(train.sum()),
            "iterations": int(iterations),
            "outlier_tolerance_255": outlier_tolerance,
            "design_rank": rank,
            "design_condition": condition if np.isfinite(condition) else None,
            "coefficients_rgb_rowvector": coefficients.tolist(),
            "coefficient_equation": (
                "corrected_rgb = target_rgb @ coefficients[:3] + coefficients[3]"
            ),
            "full_strength_clipped_channel_fraction": clipped_fraction,
            "heldout_before": before,
            "heldout_after": after,
        }
    )
    return np.clip(result, 0, 1), train.astype(np.float32), report
