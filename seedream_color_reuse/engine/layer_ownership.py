from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def ownership_reuse_anchors(
    alpha: np.ndarray,
    visibility: np.ndarray,
    reuse_mask: np.ndarray,
    eligible: np.ndarray,
    structural_error: np.ndarray,
    locked: np.ndarray,
    *,
    alpha_threshold: float = 8 / 255,
    guard_radius: int = 6,
    structure_threshold: float = 12.0,
    min_occlusion_fraction: float = .10,
):
    """Return trustworthy same-layer reuse anchors away from upper occluders."""
    shape = alpha.shape
    for name, array in (
        ("visibility", visibility),
        ("reuse_mask", reuse_mask),
        ("eligible", eligible),
        ("structural_error", structural_error),
        ("locked", locked),
    ):
        if array.shape != shape:
            raise ValueError(f"{name} must match the layer grid")
    if guard_radius < 1 or structure_threshold <= 0 or not 0 <= min_occlusion_fraction < 1:
        raise ValueError("ownership guard/structure threshold must be positive and occlusion fraction in [0,1)")

    own_alpha = alpha.astype(np.float32) / 255
    transmittance = np.divide(
        visibility.astype(np.float32),
        own_alpha,
        out=np.ones(shape, np.float32),
        where=own_alpha > 1e-7,
    )
    upper_coverage = np.clip(1 - transmittance, 0, 1)
    alpha_mass = float(own_alpha.sum())
    occlusion_fraction = float((own_alpha * upper_coverage).sum() / max(alpha_mass, 1e-7))
    covered = (upper_coverage > alpha_threshold) & (alpha > 0)
    if not covered.any() or occlusion_fraction < min_occlusion_fraction:
        return np.zeros(shape, bool), np.zeros(shape, bool), occlusion_fraction
    distance = ndi.distance_transform_edt(~covered)
    safe = (
        reuse_mask.astype(bool)
        & (alpha >= 250)
        & (visibility >= .97)
        & eligible.astype(bool)
        & ~locked.astype(bool)
        & (distance > guard_radius)
        & (structural_error < structure_threshold)
    )
    return safe, covered, occlusion_fraction


def receiver_texture_score(
    guide: np.ndarray,
    alpha: np.ndarray,
    *,
    sigma: float = 3.0,
    percentile: float = 75.0,
):
    """Measure high-frequency material detail away from transparent edges.

    The second colour pass deliberately models only a smooth receiver surface.
    Applying it to a multi-material or strongly textured layer would erase real
    model detail, so callers use this robust score as a conservative gate.
    """
    if guide.ndim != 3 or guide.shape[2] != 3 or alpha.shape != guide.shape[:2]:
        raise ValueError("guide must be HxWx3 and alpha must match its grid")
    if sigma <= 0 or not 0 < percentile < 100:
        raise ValueError("texture sigma must be positive and percentile in (0,100)")
    opaque = alpha >= 250
    if not opaque.any():
        return float("inf")
    interior = ndi.distance_transform_edt(opaque) > max(1.0, 2.0 * sigma)
    if int(interior.sum()) < 32:
        interior = opaque
    low = ndi.gaussian_filter(
        guide.astype(np.float32), (sigma, sigma, 0), mode="nearest")
    high = np.max(np.abs(guide.astype(np.float32) - low), axis=-1)
    return float(np.percentile(high[interior], percentile))


def _spatial_design(shape: tuple[int, int], support: np.ndarray):
    """A bounded quadratic field: smooth enough to avoid source-texture dirt."""
    yy, xx = np.indices(shape, dtype=np.float64)
    sy, sx = np.nonzero(support)
    if len(sx):
        x0, x1 = float(sx.min()), float(sx.max())
        y0, y1 = float(sy.min()), float(sy.max())
    else:
        x0, x1, y0, y1 = 0.0, float(shape[1] - 1), 0.0, float(shape[0] - 1)
    xn = (xx - (x0 + x1) / 2) / max((x1 - x0) / 2, 1.0)
    yn = (yy - (y0 + y1) / 2) / max((y1 - y0) / 2, 1.0)
    return np.stack(
        (np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn), axis=-1)


def _robust_spatial_fit(design: np.ndarray, values: np.ndarray, selected: np.ndarray):
    x = design[selected]
    y = values[selected].astype(np.float64)
    if len(x) < design.shape[-1] * 4 or np.linalg.matrix_rank(x) < design.shape[-1]:
        return np.broadcast_to(np.median(y, axis=0), (*selected.shape, 3)).copy()
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(6):
        error = y - x @ coefficients
        scale = np.median(np.abs(error), axis=0) + .5
        standardized = np.max(np.abs(error) / scale, axis=1)
        weight = np.minimum(1, 2.5 / np.maximum(standardized, 1e-6))
        coefficients = np.linalg.lstsq(
            x * weight[:, None], y * weight[:, None], rcond=None)[0]
    field = (design.reshape(-1, design.shape[-1]) @ coefficients).reshape(*selected.shape, 3)
    # Quadratic extrapolation is useful for broad illumination, but it must not
    # invent shifts outside the robust range actually seen in reused pixels.
    low = np.percentile(y, 2, axis=0) - 2
    high = np.percentile(y, 98, axis=0) + 2
    return np.clip(field, low, high)


def harmonize_nonreuse(
    guide: np.ndarray,
    calibrated_base: np.ndarray,
    corrected: np.ndarray,
    alpha: np.ndarray,
    safe_reuse: np.ndarray,
    covered: np.ndarray,
    eligible: np.ndarray,
    locked: np.ndarray,
    *,
    max_color_shift: float = 48.0,
    seam_sigma: float = 3.0,
    min_anchor_pixels: int = 32,
):
    """Second colour pass for non-reused pixels of an occluded receiver.

    ``calibrated_base`` is the first whole-layer colour fit learned from safe
    exact-reuse pixels. A robust quadratic RGB residual models only broad
    illumination. Exact safe pixels are then restored, and a small normalized
    low-pass correction makes the non-reused side meet them without importing
    their high-frequency texture. The model guide remains the sole source of
    high frequencies throughout.
    """
    if guide.shape != corrected.shape or guide.shape != calibrated_base.shape:
        raise ValueError("guide/base/corrected must be equal HxWx3 arrays")
    if guide.ndim != 3 or guide.shape[2] != 3:
        raise ValueError("guide/base/corrected must be HxWx3 arrays")
    shape = guide.shape[:2]
    for name, array in (
        ("alpha", alpha),
        ("safe_reuse", safe_reuse),
        ("covered", covered),
        ("eligible", eligible),
        ("locked", locked),
    ):
        if array.shape != shape:
            raise ValueError(f"{name} must match the layer grid")
    if seam_sigma <= 0 or min_anchor_pixels < 1 or max_color_shift <= 0:
        raise ValueError("seam sigma, anchor count and max shift must be positive")
    anchor_count = int(safe_reuse.sum())
    if not covered.any() or anchor_count < min_anchor_pixels:
        return corrected.copy(), np.zeros(shape, bool), np.zeros(shape, np.float32), anchor_count

    residual = corrected.astype(np.float32) - calibrated_base.astype(np.float32)
    design = _spatial_design(shape, alpha > 0)
    broad = _robust_spatial_fit(design, residual, safe_reuse)
    broad = np.clip(broad, -max_color_shift, max_color_shift)
    prediction = np.clip(calibrated_base.astype(np.float32) + broad, 0, 255)

    # Match the exact side of the seam using only a low-pass residual. Density
    # tapering prevents sparse anchors from painting remote parts of the layer.
    mismatch = (corrected.astype(np.float32) - prediction) * safe_reuse[..., None]
    density = ndi.gaussian_filter(safe_reuse.astype(np.float32), seam_sigma, mode="nearest")
    numerator = ndi.gaussian_filter(mismatch, (seam_sigma, seam_sigma, 0), mode="nearest")
    local = np.divide(
        numerator, density[..., None], out=np.zeros_like(numerator),
        where=density[..., None] > 1e-4)
    local_amount = np.clip(density / .25, 0, 1)
    prediction = np.clip(prediction + local * local_amount[..., None], 0, 255)

    editable = (alpha > 0) & eligible.astype(bool) & ~locked.astype(bool)
    prediction = np.rint(prediction).astype(np.float32)
    adjusted = corrected.copy().astype(np.float32)
    adjusted[editable] = prediction[editable]
    adjusted[safe_reuse] = corrected[safe_reuse]
    changed = editable & ~safe_reuse & np.any(adjusted != corrected, axis=-1)
    return adjusted, changed, editable.astype(np.float32), anchor_count


def harmonize_occlusion_seams(
    guide: np.ndarray,
    corrected: np.ndarray,
    alpha: np.ndarray,
    safe_reuse: np.ndarray,
    covered: np.ndarray,
    eligible: np.ndarray,
    locked: np.ndarray,
    *,
    max_color_shift: float = 48.0,
    seam_sigma: float = 16.0,
    min_anchor_pixels: int = 32,
):
    """Extend a receiver's colour residual through an upper-layer footprint.

    Unlike the whole-layer second fit, this fallback is safe for textured
    receivers and small occluders: it edits only opaque material under or near
    the occluder, and smooths *correction residuals*, never model RGB. Each
    connected opaque component gets its own anchors, so disconnected materials
    cannot supply one another's colour. Protected receiver pixels also supply
    boundary conditions but are never edited. Exact reuse outside the feather
    remains unchanged; callers must withdraw claims for changed seam pixels.
    """
    if guide.shape != corrected.shape or guide.ndim != 3 or guide.shape[2] != 3:
        raise ValueError("guide/corrected must be equal HxWx3 arrays")
    shape = guide.shape[:2]
    for name, array in (("alpha", alpha), ("safe_reuse", safe_reuse),
                        ("covered", covered), ("eligible", eligible), ("locked", locked)):
        if array.shape != shape:
            raise ValueError(f"{name} must match the layer grid")
    if (not np.isfinite(guide).all() or not np.isfinite(corrected).all()
            or not np.isfinite(seam_sigma) or seam_sigma <= 0
            or min_anchor_pixels < 1 or not np.isfinite(max_color_shift) or max_color_shift <= 0):
        raise ValueError("finite images, positive seam sigma, anchor count and max shift required")

    adjusted = corrected.astype(np.float32).copy()
    amount = np.zeros(shape, np.float32)
    solid = alpha == 255
    editable = solid & eligible.astype(bool) & ~locked.astype(bool)
    labels, _ = ndi.label(solid)
    residual = corrected.astype(np.float32) - guide.astype(np.float32)
    anchors_used = 0
    for label, region in enumerate(ndi.find_objects(labels), start=1):
        if region is None:
            continue
        component = labels[region] == label
        occluded = component & covered[region].astype(bool)
        if int(occluded.sum()) < min_anchor_pixels:
            continue
        safe = component & safe_reuse[region].astype(bool) & eligible[region].astype(bool)
        safe &= ~locked[region].astype(bool)
        count = int(safe.sum())
        if count < min_anchor_pixels:
            continue
        # Nearest extension is constructed from one frozen receiver residual.
        # No reference pixels or upper-layer RGB are used as hidden evidence.
        boundary_anchors = safe | (component & locked[region].astype(bool))
        _, indices = ndi.distance_transform_edt(~boundary_anchors, return_indices=True)
        filled = residual[region][tuple(indices)]
        field = ndi.gaussian_filter(filled, (seam_sigma, seam_sigma, 0), mode="nearest")
        field = np.clip(field, -max_color_shift, max_color_shift)
        prediction = np.clip(guide[region].astype(np.float32) + field, 0, 255)
        distance = ndi.distance_transform_edt(~occluded)
        local_amount = np.clip(1 - distance / (3 * seam_sigma), 0, 1).astype(np.float32)
        local_amount *= local_amount * (3 - 2 * local_amount)
        local_amount *= component & editable[region]
        # Taper into immutable boundaries instead of creating a second seam at
        # their protection masks. Fully hidden opaque interiors stay editable.
        protected = component & locked[region].astype(bool)
        if protected.any():
            lock_distance = ndi.distance_transform_edt(~protected)
            lock_amount = np.clip(lock_distance / seam_sigma, 0, 1).astype(np.float32)
            local_amount *= lock_amount * lock_amount * (3 - 2 * lock_amount)
        delta = prediction - corrected[region]
        # Avoid sacrificing exact claims for sub-code-value cosmetic changes.
        if not np.any((local_amount > 0) & (np.max(np.abs(delta), axis=-1) > 1)):
            continue
        adjusted[region] += local_amount[..., None] * delta
        amount[region] = np.maximum(amount[region], local_amount)
        anchors_used += count
    adjusted = np.rint(np.clip(adjusted, 0, 255)).astype(np.float32)
    # Preserve float-valued hidden, transparent, protected and remote pixels.
    adjusted[amount == 0] = corrected[amount == 0]
    changed = (amount > 0) & np.any(adjusted != np.rint(corrected), axis=-1)
    return adjusted, changed, amount, anchors_used
