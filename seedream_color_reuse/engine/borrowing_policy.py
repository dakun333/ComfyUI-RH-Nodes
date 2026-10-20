"""Separate reference-borrowing evidence from model-derived contour repair."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def contact_shadow_tail_guard(guide, reference, visible, conflict_core, *, scale=1.):
    """Exclude broad dark tails connected to an existing material conflict.

    This is deliberately NOT a generic shadow detector. A changed-material
    core supplies the missing-reference-object evidence when layer-order cast
    shadow detection has no caster. Flat model material, neutral dark residual,
    local contact and thick connected seeds are all required. Tiny marks,
    distant detail, global recolouring and represented shading remain outside.
    """
    shape=visible.shape
    empty=np.zeros(shape,np.float32)
    if not visible.any() or not conflict_core.any():
        return empty
    distance=ndi.distance_transform_edt(~conflict_core)
    residual=guide-reference
    signed=residual.mean(axis=-1)
    low=ndi.gaussian_filter(guide,(.65,.65,0),mode="nearest")
    gx=ndi.sobel(low,axis=1,mode="nearest")/8
    gy=ndi.sobel(low,axis=0,mode="nearest")/8
    gradient=np.sqrt(np.max(gx*gx+gy*gy,axis=-1))
    # Small uniform offsets must not become a new local protection region.
    outside=visible & (distance>max(24.,48.*scale)) & (gradient<5.)
    if outside.sum()<32:
        return empty  # No distant same-material evidence for a local excess.
    offset=float(np.median(signed[outside]))
    darkness=signed-offset
    neutral=(np.min(residual,axis=-1)>-8.) & (np.ptp(residual,axis=-1)<32.)
    valid=visible & ~conflict_core & neutral & (gradient<5.)
    weak=valid & (darkness>3.) & (distance<=max(16.,32.*scale))
    strong=weak & (darkness>18.) & (distance<=max(12.,16.*scale))
    labels,count=ndi.label(weak)
    accepted=np.zeros(count+1,bool)
    for index,box in enumerate(ndi.find_objects(labels),1):
        component=labels[box]==index
        seeds=component & strong[box]
        if component.sum()<max(16,int(32*scale*scale)) or seeds.sum()<max(6,int(12*scale*scale)):
            continue
        depth=ndi.distance_transform_edt(np.pad(seeds,1))[1:-1,1:-1]
        if depth.max()<max(2.,2.*scale):
            continue
        accepted[index]=True
    core=accepted[labels]
    if not core.any():
        return empty
    # Propagate in model-flat material only; an EDT feather must not jump
    # across a model colour/coverage barrier into an unrelated surface.
    traversable=visible & (gradient<5.)
    permitted=ndi.binary_propagation(core,mask=traversable)
    dist=ndi.distance_transform_edt(~core)
    amount=np.clip(1-(dist-max(2.,3.*scale))/max(6.,10.*scale),0,1).astype(np.float32)
    amount=amount*amount*(3-2*amount)
    return amount*permitted


def protected_model_target(current, model_target, gate, protection):
    """Continuous model-only repair, preserving protected cores exactly.

    Reference borrowing remains independently excluded. A nonzero feather is
    not a boolean model-repair veto; its continuous weight attenuates repair.
    All callers still enforce user locks, coverage and layer ownership.
    """
    amount=np.clip(gate,0,1)*(1-np.clip(protection,0,1))
    return np.clip(current+amount[...,None]*(model_target-current),0,255).astype(np.float32)
