from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def flat_receiver_guard(guide, alpha, visibility, eligible, locked, *,
                        radius=12., min_samples=32):
    """Keep an upper silhouette out of a locally quiet opaque receiver.

    The guide is frozen BEFORE reference borrowing. Connected, low-gradient
    material patches are evaluated locally, not with a whole-character texture
    score. Only patches with a substantial upper occluder and reliable visible
    material around it qualify. The result attenuates reference borrowing and
    locks that receiver against composite-domain contour projection; it never
    locks the upper owner or modifies coverage. Detail in the guide is retained.
    """
    shape = alpha.shape
    if guide.shape != (*shape, 3):
        raise ValueError("guide must be HxWx3 on the alpha grid")
    for value in (visibility, eligible, locked):
        if value.shape != shape:
            raise ValueError("receiver masks must match the alpha grid")
    if (not np.isfinite(guide).all() or not np.isfinite(visibility).all()
            or np.any(visibility < 0) or np.any(visibility > 1 + 1e-6)
            or not np.isfinite(radius) or radius <= 0 or min_samples < 1):
        raise ValueError("finite inputs and positive radius/sample count required")
    amount = np.zeros(shape, np.float32)
    solid = ndi.minimum_filter(alpha, size=3, mode="constant", cval=0) == 255
    own = alpha.astype(np.float32) / 255
    transmittance = np.divide(visibility, own, out=np.ones(shape, np.float32),
                             where=own > 1e-7)
    upper = np.clip(1 - transmittance, 0, 1)
    valid = solid & eligible.astype(bool) & ~locked.astype(bool)
    covered = valid & (upper > 1 / 255)
    opaque_cover = valid & (upper >= .5)
    if int(opaque_cover.sum()) < min_samples:
        return amount

    low = ndi.gaussian_filter(guide.astype(np.float32), (.8, .8, 0), mode="nearest")
    gx = np.stack([ndi.sobel(low[..., c], axis=1, mode="nearest") / 8
                   for c in range(3)], axis=-1)
    gy = np.stack([ndi.sobel(low[..., c], axis=0, mode="nearest") / 8
                   for c in range(3)], axis=-1)
    gradient = np.max(np.sqrt(gx * gx + gy * gy), axis=-1)
    high = np.max(np.abs(guide - ndi.gaussian_filter(
        guide.astype(np.float32), (2., 2., 0), mode="nearest")), axis=-1)
    # Black outlines, colour boundaries and texture are barriers. These fixed
    # code-value limits concern model material only, never reference conflict.
    quiet = valid & (gradient < 3.) & (high < 3.)
    labels, _ = ndi.label(quiet)
    regions = ndi.find_objects(labels)
    candidates = np.unique(labels[opaque_cover])
    for label in candidates:
        if label == 0:
            continue
        region = regions[int(label) - 1]
        component = labels[region] == label
        core = component & opaque_cover[region]
        if int(core.sum()) < min_samples:
            continue
        local_cover = component & covered[region]
        # Padding makes distances meaningful even when a component fills its
        # bounding rectangle or the occluder touches the rectangle's edge.
        distance = ndi.distance_transform_edt(
            ~np.pad(local_cover, 1, constant_values=False))[1:-1, 1:-1]
        anchors = component & (upper[region] <= 1e-7) & (distance > radius)
        if int(anchors.sum()) < min_samples:
            continue
        t = np.clip(1 - distance / radius, 0, 1).astype(np.float32)
        local = t * t * (3 - 2 * t)
        boundary_distance = ndi.distance_transform_edt(
            np.pad(component, 1, constant_values=False))[1:-1, 1:-1]
        b = np.clip(boundary_distance / 3., 0, 1).astype(np.float32)
        local *= b * b * (3 - 2 * b)
        local *= component
        amount[region] = np.maximum(amount[region], local)
    return amount
