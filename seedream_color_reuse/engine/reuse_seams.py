"""Local continuity of unsupported pixel-borrowing residuals in opaque material.

No source RGB is used here. A frozen first-colour-fit layer supplies material
geometry; the final minus fitted RGB supplies only the correction residual.
Disconnected quiet material domains and immutable pixels are barriers.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def harmonize_reuse_boundaries(guide, corrected, alpha, visibility,
                               reuse_amount, eligible, locked):
    """Smooth only evidenced sharp reuse transitions, never whole layer RGB.

    A seed requires a sharp frozen borrowing-amount transition, a residual jump,
    and locally quiet model material. Soft/hidden alpha and guarded pixels are
    immutable. Filtering is restricted to the same quiet connected component;
    normalized convolution prevents transparent RGB or another object from
    contributing evidence. A bounded C1 feather preserves remote exact reuse.
    """
    guide = np.asarray(guide)
    corrected = np.asarray(corrected)
    shape = alpha.shape
    if guide.shape != (*shape, 3) or corrected.shape != guide.shape:
        raise ValueError("guide/corrected must match alpha as HxWx3")
    for value in (visibility, reuse_amount, eligible, locked):
        if value.shape != shape:
            raise ValueError("visibility/reuse/eligible/locked must match alpha")
    if (not np.isfinite(guide).all() or not np.isfinite(corrected).all()
            or not np.isfinite(visibility).all() or not np.isfinite(reuse_amount).all()):
        raise ValueError("finite RGB and evidence arrays required")
    result = corrected.astype(np.float32).copy()
    amount = np.zeros(shape, np.float32)
    seed = np.zeros(shape, bool)
    allowed = ((ndi.minimum_filter(alpha, size=3, mode="constant", cval=0) == 255)
               & (visibility >= 1 - 1e-7) & eligible.astype(bool)
               & ~locked.astype(bool))
    if not allowed.any() or float(np.ptp(reuse_amount[allowed])) < .35:
        return result, seed, amount, seed.copy()
    # Undefined transparent/hidden/locked RGB must not influence even the
    # geometry detector, not just its residual filter.
    support = allowed.astype(np.float32)
    geometry_mass = ndi.gaussian_filter(support, .65, mode="constant", cval=0)
    smooth = ndi.gaussian_filter(guide.astype(np.float32) * support[..., None],
                                 (.65, .65, 0), mode="constant", cval=0)
    smooth /= np.maximum(geometry_mass[..., None], 1e-7)
    if not allowed.all():
        _, nearest = ndi.distance_transform_edt(~allowed, return_indices=True)
        smooth[~allowed] = smooth[tuple(nearest[:, ~allowed])]

    gx = np.stack([ndi.sobel(smooth[..., c], axis=1, mode="nearest") / 8
                   for c in range(3)], axis=-1)
    gy = np.stack([ndi.sobel(smooth[..., c], axis=0, mode="nearest") / 8
                   for c in range(3)], axis=-1)
    gradient = np.sqrt(np.max(gx * gx + gy * gy, axis=-1))
    # Quiet material domains are NOT semantic labels. Strong model profiles
    # split them so residuals cannot leak across wood strands/material edges.
    quiet = allowed & (gradient < 4.0)
    residual = corrected.astype(np.float32) - guide.astype(np.float32)
    for axis in (0, 1):
        left = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        right = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        transition = np.abs(reuse_amount[left] - reuse_amount[right]) > .35
        jump = np.max(np.abs(residual[left] - residual[right]), axis=-1) > 3.0
        model_step = np.max(np.abs(guide[left] - guide[right]), axis=-1) < 5.0
        paired = quiet[left] & quiet[right] & transition & jump & model_step
        seed[left] |= paired
        seed[right] |= paired
    # Isolated subpixel decisions are not a coherent material seam.
    labels, _ = ndi.label(seed, structure=np.ones((3, 3), bool))
    counts = np.bincount(labels.ravel())
    seed &= counts[labels] >= 8
    if not seed.any():
        return result, np.zeros(shape, bool), amount, seed
    domains, _ = ndi.label(quiet)
    domain_regions = ndi.find_objects(domains)
    for label in np.unique(domains[seed]):
        if not label:
            continue
        region = domain_regions[label - 1]
        component = domains[region] == label
        if int(component.sum()) < 32:
            continue
        local_seed = seed[region] & component
        if not local_seed.any():
            continue
        support = component.astype(np.float32)
        # Smooth residuals over a few pixels, not guide RGB; constant colour
        # corrections remain constant and original strand detail is retained.
        denominator = ndi.gaussian_filter(support, 3.0, mode="constant", cval=0)
        numerator = ndi.gaussian_filter(residual[region] * support[..., None],
                                         (3.0, 3.0, 0), mode="constant", cval=0)
        field = numerator / np.maximum(denominator[..., None], 1e-7)
        distance = ndi.distance_transform_edt(~local_seed)
        local_amount = np.clip(1 - distance / 9.0, 0, 1).astype(np.float32)
        local_amount *= local_amount * (3 - 2 * local_amount)
        # Taper into a hard material/protection boundary as well as away from
        # the seed. Never create a fresh edge immediately beside a lock.
        boundary_distance = ndi.distance_transform_edt(np.pad(component, 1))[1:-1, 1:-1]
        boundary_amount = np.clip((boundary_distance - 1) / 3.0, 0, 1).astype(np.float32)
        boundary_amount *= boundary_amount * (3 - 2 * boundary_amount)
        local_amount *= component * boundary_amount
        prediction = np.clip(guide[region] + field, 0, 255)
        proposed = np.rint(np.clip(corrected[region] + local_amount[..., None] *
                                  (prediction - corrected[region]), 0, 255))
        selected = local_amount > 0
        result[region][selected] = proposed[selected]
        amount[region] = np.maximum(amount[region], local_amount)
    changed = (amount > 0) & np.any(result != corrected, axis=-1)
    return result, changed, amount, seed
