"""Respect foreground material changes; extrapolate colors, never source geometry."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def surface_consistency(mapped, target, visible, forbidden, *, scale=1., coverage=None):
    """Return a borrowing exclusion and bounded same-material color residual.

    Strong *chromatic* disagreement seeds changed-material components. Neutral
    shadows remain the responsibility of the separate removed-shadow detector.
    Weak connected fringes are protected too, rather than copied up to a hard
    RGB cutoff. Local color/spatial anchors exclude all such components and
    explicit/shadow protection. No target texture is synthesized in the hole.
    """
    shape = visible.shape
    guard = np.zeros(shape, np.float32)
    correction = np.zeros_like(mapped)
    delta = target - mapped
    chroma = np.ptp(delta, axis=-1)
    difference = np.max(np.abs(delta), axis=-1)
    allowed = visible & ~forbidden
    ratio = np.mean(target, axis=-1) / np.maximum(np.mean(mapped, axis=-1), 1)
    strong = allowed & (chroma > 48) & (difference > 48) & (ratio > .5)
    weak = allowed & (chroma > 20) & (difference > 16)
    labels, count = ndi.label(weak)
    if not count or not strong.any():
        return guard, correction
    sizes = np.bincount(labels.ravel())
    seeds = np.bincount(labels[strong], minlength=count+1)
    depth = ndi.maximum(ndi.distance_transform_edt(strong), labels, np.arange(count+1))
    keep = (sizes >= max(64, 128*scale*scale)) & (seeds >= max(16, 32*scale*scale)) & (depth >= max(6, 8*scale))
    keep[0] = False
    core = keep[labels]
    if not core.any():
        return guard, correction
    distance = ndi.distance_transform_edt(~core)
    margin, feather = max(2, 3*scale), max(6, 16*scale)
    guard = np.clip(1-(distance-margin)/feather, 0, 1).astype(np.float32)
    guard = guard*guard*(3-2*guard)
    guard *= ((visible if coverage is None else coverage) & ~forbidden)
    # Learn only small, locally consistent residuals from unchanged pixels.
    safe = allowed & (distance > margin+feather) & (difference < 20) & (chroma < 16)
    near = distance < max(48, 160*scale)
    yy, xx = np.indices(shape)
    step = max(1, round(3*scale))
    anchors = safe & near & (yy % step == 0) & (xx % step == 0)
    ay, ax = np.nonzero(anchors)
    qy, qx = np.nonzero(guard > 0)
    if len(ay) < 8:
        return guard, correction
    spatial = max(24, 64*scale)
    colors = mapped[ay, ax]
    features = np.column_stack((ay/spatial, ax/spatial, colors/16))
    tree = cKDTree(features)
    residuals = delta[ay, ax]
    for start in range(0, len(qy), 40000):
        y, x = qy[start:start+40000], qx[start:start+40000]
        queries = np.column_stack((y/spatial, x/spatial, mapped[y,x]/16))
        _, indices = tree.query(queries, k=min(24,len(ay)))
        sd = ((ay[indices]-y[:,None])**2+(ax[indices]-x[:,None])**2)/spatial**2
        cd = np.mean((colors[indices]-mapped[y,x,None])**2,axis=-1)
        weights = np.exp(-.5*sd-cd/(2*16**2))
        weights *= (sd < 9) & (cd < 32**2)
        total = weights.sum(axis=1)
        estimate = (residuals[indices]*weights[...,None]).sum(axis=1)/np.maximum(total[:,None],1e-8)
        # Sparse/far evidence fades continuously instead of making a new seam.
        confidence = np.clip(total/4,0,1)
        correction[y,x] = np.clip(estimate,-20,20)*confidence[:,None]
    correction *= guard[...,None]
    return guard, correction
