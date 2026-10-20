"""Conservative source-only cast-shadow exclusion on receiving layers.

This is a geometric/photometric heuristic, not a semantic shadow oracle.
The receiver must have a higher-layer occluder and a broad, connected dark
residual after provisional color calibration. Uniform recoloring, tiny marks,
already represented shading and alpha fringes are not sufficient evidence.
"""
import numpy as np
from scipy import ndimage as ndi


def removed_shadow_guard(predicted, reference, visible, occluder, *, scale=1.):
    """Return receiver-grid suppression [0,1] and a fully protected core.

    A dark core seeds a more permissive connected penumbra. Only components
    reaching an occluder neighborhood survive. Extend beyond that penumbra and
    feather *outside* the core, so low-contrast shadow tails cannot be copied
    while a hard protection boundary creates another seam. The caller clips
    it to receiver coverage and excludes it from fit and all copying stages.
    """
    empty = np.zeros(visible.shape, np.float32)
    if not visible.any() or not occluder.any():
        return empty, empty.astype(bool)
    sigma=max(1., 2.*scale)
    residual=predicted-reference
    # Average darkening permits colored penumbrae, but reject brightenings
    # masquerading as darkening due to a large change in one channel.
    signed=residual.mean(axis=-1)
    coherent=np.min(residual, axis=-1) > -12
    valid=visible & coherent
    denominator=ndi.gaussian_filter(valid.astype(np.float32), sigma)
    darkness=ndi.gaussian_filter(np.where(valid,signed,0), sigma)/np.maximum(denominator,1e-6)
    weak=(darkness>3.) & valid
    # A displaced black outline can also be a large negative residual. It is
    # not a soft cast shadow: require actual (not blur-spilled) midtone
    # attenuation and a spatially thick seed, rather than black-ink mismatch.
    attenuation=reference.mean(axis=-1)/np.maximum(predicted.mean(axis=-1),1.)
    strong=(darkness>18.) & (signed>18.) & (attenuation>.35) & valid
    near=ndi.distance_transform_edt(~occluder) <= max(6.,16.*scale)
    labels,count=ndi.label(weak)
    accepted=np.zeros(count+1,bool)
    min_area=max(16,int(round(64*scale*scale)))
    min_seed=max(8,int(round(32*scale*scale)))
    for index,box in enumerate(ndi.find_objects(labels),1):
        component=labels[box]==index
        if component.sum()<min_area or not np.any(component & near[box]):
            continue
        seeds=component & strong[box]
        if seeds.sum()<min_seed:
            continue
        depth=ndi.distance_transform_edt(np.pad(seeds,1))[1:-1,1:-1]
        if depth.max()<max(3.,3.*scale):
            continue
        accepted[index]=True
    core=accepted[labels]
    if not core.any():
        return empty,core
    core=ndi.binary_fill_holes(core) & visible
    # Small holes caused by grooves/texture remain inside the shadow envelope;
    # distance extension covers coverage fringes without borrowing source RGB.
    distance=ndi.distance_transform_edt(~core)
    margin=max(2.,4.*scale)
    feather=max(4.,12.*scale)
    amount=np.clip(1-(distance-margin)/feather,0,1).astype(np.float32)
    amount=amount*amount*(3-2*amount)  # smooth endpoints
    return amount,core
