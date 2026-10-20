"""Regularize only correction residuals at disturbed model contours."""
import numpy as np
from scipy import ndimage as ndi


def smoothstep(x):
    x=np.clip(x,0,1)
    return x*x*(3-2*x)


def introduced_contour_detail(current, model, reference):
    """Only detail exceeding BOTH input contours is evidence of a new tooth.

    A clean source contour can differ from the model (subpixel displacement,
    matte repair, crop seam) without being a defect. Do not revert it merely
    because its correction residual has a gradient.
    """
    def detail(image):
        return np.max(np.abs(image-ndi.gaussian_filter(image,(.8,.8,0))),axis=-1)
    excess=detail(current)-np.maximum(detail(model),detail(reference))
    strength=smoothstep((ndi.maximum_filter(excess,size=5)-1)/5)
    return ndi.gaussian_filter(strength,.65).astype(np.float32)


def regularize_contours(guide, corrected, alpha, visibility, protected, *, reuse_mask=None, artifact_strength=None):
    residual=corrected-guide
    low=ndi.gaussian_filter(guide,(.7,.7,0))
    gradient=np.maximum(np.max(np.abs(ndi.sobel(low,axis=0)/8),axis=-1),np.max(np.abs(ndi.sobel(low,axis=1)/8),axis=-1))
    # Include fixed coverage contours without changing their alpha.
    boundary=ndi.maximum_filter(alpha,size=5)>ndi.minimum_filter(alpha,size=5)
    strength=np.maximum(smoothstep((gradient-3)/7),boundary.astype(np.float32))
    strength=ndi.gaussian_filter(ndi.maximum_filter(strength,size=3),.65)
    hp=np.max(np.abs(residual-ndi.gaussian_filter(residual,(1.,1.,0))),axis=-1)
    roughness=ndi.gaussian_filter(ndi.maximum_filter(hp,size=3),.7)
    amount=strength*smoothstep((roughness-2)/6)
    if reuse_mask is not None:
        mixed=ndi.maximum_filter(reuse_mask,size=5)&~ndi.minimum_filter(reuse_mask,size=5)
        amount*=ndi.gaussian_filter(mixed.astype(np.float32),.65)
    if artifact_strength is not None:
        amount*=artifact_strength
    allowed=(alpha>8)&(visibility>.05)&~protected
    stable=(alpha>=245)&(visibility>=.97)&~protected
    stable_weight=(1-smoothstep((gradient-3)/5))*(1-smoothstep((hp-2)/5))
    stable_weight*=stable & (np.max(np.abs(residual),axis=-1)<32)
    allowed &= np.max(np.abs(residual),axis=-1)<80
    amount*=allowed
    yy,xx=np.nonzero(amount>.001)
    if not len(yy):return corrected.copy(),np.zeros(alpha.shape,bool),amount
    estimate=np.zeros((len(yy),3),np.float32);total=np.zeros(len(yy),np.float32)
    h,w=alpha.shape
    # One frozen pass; never propagate edited colors back as new evidence.
    for dy in range(-8,9):
        for dx in range(-8,9):
            y=np.clip(yy+dy,0,h-1);x=np.clip(xx+dx,0,w-1)
            cd=np.mean((guide[y,x]-guide[yy,xx])**2,axis=-1)
            wt=np.exp(-(dy*dy+dx*dx)/(2*4**2)-cd/(2*18**2))
            wt*=stable_weight[y,x]
            estimate+=residual[y,x]*wt[:,None];total+=wt
    estimate/=np.maximum(total[:,None],1e-8)
    estimate*=np.clip(total[:,None]/2,0,1)
    result=corrected.copy()
    adjustment=estimate-residual[yy,xx]
    # Do not trade exact pixels for sub-code-value cosmetic changes.
    amount[yy,xx]*=smoothstep((np.max(np.abs(adjustment),axis=-1)-1)/2)
    result[yy,xx]+=amount[yy,xx,None]*adjustment
    result=np.clip(result,0,255)
    changed=np.any(np.rint(result)!=np.rint(corrected),axis=-1)
    return result,changed,amount
