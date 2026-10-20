"""Keep the model's antialiased contour profile; infer only slow color offsets."""
import numpy as np
from scipy import ndimage as ndi
from .contours import smoothstep


def chromatic_ink_allowance(guide, alpha, visibility, protected):
    """Relax legacy low-channel ink only with bright chromatic material evidence.

    A low G/B value is not black ink when R is bright. Keep the old exclusion
    for dark/neutral colors, hidden/transparent witnesses and protected material.
    Every dark witness in the same 5x5 footprint must support the relaxation;
    one actual ink pixel still locks the neighborhood. Smooth color confidence
    tapers partial evidence instead of adding another binary RGB cutoff.
    """
    lo = np.min(guide, axis=-1)
    hi = np.max(guide, axis=-1)
    legacy_ink = ndi.minimum_filter(lo, size=5) < 32
    bright = smoothstep((hi - 144.0) / 48.0)
    chromatic = smoothstep((hi - lo - 80.0) / 48.0)
    witness = ((alpha == 255) & (visibility >= .97) & ~protected)
    confidence = bright * chromatic * witness
    veto = (lo < 32).astype(np.float32) * (1.0 - confidence)
    allowance = 1.0 - ndi.maximum_filter(veto, size=5)
    solid = ndi.minimum_filter(alpha, size=3, mode="constant", cval=0) == 255
    # Soft mattes and hidden regions stay byte-identical to their legacy path.
    can_relax = solid & (visibility >= .97) & ~protected
    return np.where(legacy_ink & ~can_relax, 0.0, allowance).astype(np.float32)


def preserve_model_contours(guide, corrected, alpha, visibility, protected, reuse_mask,
                            *, continuation=None):
    # Reuse is reporting evidence, not permission to overwrite a model edge:
    # an entirely source-exact contour can itself have incompatible geometry.
    residual=corrected-guide
    low=ndi.gaussian_filter(guide,(.65,.65,0))
    gx=np.stack([ndi.sobel(low[...,c],axis=1)/8 for c in range(3)],axis=-1)
    gy=np.stack([ndi.sobel(low[...,c],axis=0)/8 for c in range(3)],axis=-1)
    gradient=np.sqrt(np.max(gx*gx+gy*gy,axis=-1))
    valid=(alpha>8)&(visibility>.05)&~protected
    valid &= np.max(np.abs(residual),axis=-1)<80
    ink=ndi.minimum_filter(np.min(guide,axis=-1),size=5)<32
    # Keep legacy seeds/component selection: do not enable new edge segments.
    edge=(gradient>5)&valid&~ink
    hp=np.max(np.abs(residual-ndi.gaussian_filter(residual,(1.2,1.2,0))),axis=-1)
    # A contour-position error must oscillate ALONG the model edge, not just
    # have a large normal gradient (a clean translated contour has that too).
    channel=np.argmax(gx*gx+gy*gy,axis=-1)
    nx=np.take_along_axis(gx,channel[...,None],axis=-1)[...,0]/np.maximum(gradient,1e-6)
    ny=np.take_along_axis(gy,channel[...,None],axis=-1)[...,0]/np.maximum(gradient,1e-6)
    yy,xx=np.indices(alpha.shape,dtype=np.float32)
    deviations=[];consistent=np.ones(alpha.shape,bool)
    for direction in (-1,1):
        coords=[yy+direction*nx*3,xx-direction*ny*3]
        nearby=np.stack([ndi.map_coordinates(residual[...,c],coords,order=1,mode="nearest") for c in range(3)],axis=-1)
        deviations.append(nearby)
        guide_neighbor=np.stack([ndi.map_coordinates(guide[...,c],coords,order=1,mode="nearest") for c in range(3)],axis=-1)
        consistent &= np.max(np.abs(guide_neighbor-guide),axis=-1)<24
    oscillation=np.max(np.abs(residual-(deviations[0]+deviations[1])*.5),axis=-1)
    # A successfully removed crop seam has no output contour to repair.
    corrected_low=ndi.gaussian_filter(corrected,(.65,.65,0))
    cgx=np.stack([ndi.sobel(corrected_low[...,c],axis=1)/8 for c in range(3)],axis=-1)
    cgy=np.stack([ndi.sobel(corrected_low[...,c],axis=0)/8 for c in range(3)],axis=-1)
    output_gradient=np.sqrt(np.max(cgx*cgx+cgy*cgy,axis=-1))
    still_an_edge=ndi.maximum_filter(output_gradient,size=7)>gradient*.5
    seeds=edge & still_an_edge & consistent & (oscillation>5) & (hp>3) & (np.max(np.abs(residual),axis=-1)>5)
    # Thin to the centerline before grouping: a broad gradient patch can
    # connect unrelated contours and unnecessarily suppress large interiors.
    normal_plus=ndi.map_coordinates(gradient,[yy+ny,xx+nx],order=1,mode="nearest")
    normal_minus=ndi.map_coordinates(gradient,[yy-ny,xx-nx],order=1,mode="nearest")
    ridge=edge & (gradient>=normal_plus) & (gradient>=normal_minus)
    seeds=ridge & ndi.maximum_filter(seeds,size=5)
    labels,n=ndi.label(ridge,structure=np.ones((3,3),bool))
    counts=np.bincount(labels[seeds],minlength=n+1)
    sizes=np.bincount(labels.ravel(),minlength=n+1)
    accepted=(counts>=2)&(sizes>=12);accepted[0]=False
    core=accepted[labels]
    if not core.any():return corrected.copy(),np.zeros(alpha.shape,bool),np.zeros(alpha.shape,np.float32)
    distance=ndi.distance_transform_edt(~core)
    amount=(1-smoothstep((distance-2)/3))*valid*(~ink)

    # Do not estimate color offsets from the shifted contours being rejected.
    stable=(alpha>=245)&(visibility>=.97)&~protected&(distance>4)&(gradient<1.5)&(hp<3)
    stable &= np.max(np.abs(residual),axis=-1)<24
    den=ndi.gaussian_filter(stable.astype(np.float32),8.)
    num=ndi.gaussian_filter(np.clip(residual,-24,24)*stable[...,None],(8.,8.,0))
    estimate=num/np.maximum(den[...,None],1e-6)
    estimate*=smoothstep(den/.1)[...,None]
    if continuation is not None and np.any(valid & ink & (distance < 5)):
        # Frozen evidence for a separate FINAL repair. Never change legacy
        # contour output, seeds, stable samples or downstream filter evidence.
        ink_allowance = chromatic_ink_allowance(guide, alpha, visibility, protected)
        # Follow the SAME gradient ridge across a short chromatic ink gap.
        # No new artifact seeds: only components touching an old accepted core
        # can continue, and their extension is bounded to eight native pixels.
        chromatic_solid = ndi.minimum_filter(alpha, size=3, mode="constant", cval=0) == 255
        chromatic_ridge = ((gradient > 5) & valid & chromatic_solid & (visibility >= .97) &
                           (ink_allowance >= 1-1e-7) &
                           (gradient >= normal_plus) & (gradient >= normal_minus))
        continuation_labels, _ = ndi.label(chromatic_ridge, structure=np.ones((3,3),bool))
        touching = np.unique(continuation_labels[core & chromatic_ridge])
        touching = touching[touching != 0]
        extension_core = (ink & chromatic_ridge &
                          np.isin(continuation_labels, touching) & (distance <= 8))
        extension_distance = ndi.distance_transform_edt(~(core | extension_core))
        extension = ((1-smoothstep((extension_distance-2)/3))*valid*ink*ink_allowance)
        if extension.any():
            continuation["amount"] = extension.astype(np.float32)
            continuation["target"] = np.clip(guide+estimate, 0, 255).astype(np.float32)
    adjusted=corrected+amount[...,None]*(guide+estimate-corrected)
    adjusted=np.clip(adjusted,0,255)
    changed=np.any(np.rint(adjusted)!=np.rint(corrected),axis=-1)
    return adjusted,changed,amount
