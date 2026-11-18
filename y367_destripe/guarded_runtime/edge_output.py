"""CE4: CG1 plus a narrow native-resolution input-edge output veto.

No CNN/FFT/refit, no broad analysis-resolution halo and no extra deletion pass.
A protected output pixel is restored wholesale to input; all other pixels are
byte-exact CG1. The color contract and original alpha protection stay intact.
"""
import os
import time
import cv2
import numpy as np
from . import chroma_guard as cg
try:
    from .native_output import finish_edges as _native_finish
except ImportError:
    _native_finish=None
EDGE_THRESHOLD_1K = 4.0
EDGE_RADIUS_1K = 4.0
BLUR_SIGMA_1K = 1.0
METHOD = 'CE4_CG1_narrow_strong_edge_veto'


def protected_edges(rgb, *, radius=EDGE_RADIUS_1K, threshold=EDGE_THRESHOLD_1K, diagnostics=False):
    """RGB gradient seeds, then a Euclidean 4px band in short-side-1024 units.

    Detection uses ONLY the input, never the correction/difference image or ROI.
    Gaussian blur separates the weak carrier ripples from stronger object edges;
    the Scharr derivative is normalized to RGB levels per reference pixel.
    Threshold/radius are empirical global operating settings, not model output.
    """
    started=time.perf_counter()
    rgb=np.asarray(rgb,np.float32)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<2:
        raise ValueError('Expected HxWx3 RGB, sides at least 2')
    if radius<0 or threshold<=0:
        raise ValueError('Nonnegative radius and positive edge threshold required')
    scale=min(rgb.shape[:2])/1024.
    smooth=cv2.GaussianBlur(rgb,(0,0),BLUR_SIGMA_1K*scale,borderType=cv2.BORDER_REFLECT_101)
    gx=cv2.Scharr(smooth,cv2.CV_32F,1,0,scale=scale/32.,borderType=cv2.BORDER_REFLECT_101)
    gy=cv2.Scharr(smooth,cv2.CV_32F,0,1,scale=scale/32.,borderType=cv2.BORDER_REFLECT_101)
    gradient=np.sqrt(np.mean(gx*gx+gy*gy,axis=2))
    seeds=gradient>=threshold
    if seeds.any():
        distance=cv2.distanceTransform((~seeds).astype(np.uint8),cv2.DIST_L2,5)
        protected=distance<=radius*scale
    else:
        distance=np.full(rgb.shape[:2],np.inf,np.float32)
        protected=np.zeros(rgb.shape[:2],bool)
    report=dict(method='input_RGB_Scharr_then_native_distance',radius_1K_pixels=float(radius),
        radius_native_pixels=float(radius*scale),blur_sigma_native_pixels=float(BLUR_SIGMA_1K*scale),
        gradient_threshold_RGB_levels_per_1K_pixel=float(threshold),
        scale_convention='short side / 1024',native_resolution=True,
        protected_pixel_percent=float(protected.mean()*100),
        edge_seed_percent=float(seeds.mean()*100),seconds=time.perf_counter()-started,
        references_read=False,ROI_read=False,removal_map_used_as_edge_detector=False)
    if diagnostics:return protected,report,dict(gradient=gradient,seeds=seeds,distance=distance)
    return protected,report


def finish(rgb,target,candidate8,protected,edge_report=None):
    """Native RGB whole-pixel admission after CG1; never blend or recolor."""
    started=time.perf_counter()
    rgb=np.asarray(rgb,np.float32);candidate8=np.asarray(candidate8)
    if rgb.shape!=target.shape or rgb.shape!=candidate8.shape or protected.shape!=rgb.shape[:2]:
        raise ValueError('Array shapes must match')
    if _native_finish is not None and rgb.dtype==target.dtype==np.float32 and candidate8.dtype==np.uint8:
        effective,eight,active_count,restored,error_max=_native_finish(rgb,target,candidate8,protected)
        assert error_max<1.00001
    else:
        eight=candidate8.copy();effective=target.copy()
        active=np.any(eight.astype(np.float32)!=rgb,axis=2)
        eight[protected]=np.uint8(rgb[protected]);effective[protected]=rgb[protected]
        assert np.array_equal(eight[~protected],candidate8[~protected])
        assert np.array_equal(eight[protected],np.uint8(rgb[protected]))
        assert np.max(abs(eight.astype(np.float32)-effective))<1.00001
        active_count=int(active.sum());restored=int((active&protected).sum())
    report=dict(method=METHOD,edge_detector=edge_report,
        candidate_changed8_pixels=int(active_count),edge_changed8_pixels_restored=int(restored),
        unprotected_output_exact_CG1=True,protected_output_exact_input=True,
        no_channel_projection_or_colorization=True,finish_seconds=time.perf_counter()-started,
        limitation='A target stripe can also cross an object edge and remain in this band; outside-edge colorful false correction is not solved.')
    return effective,eight,report


def process(rgb,alpha,*,orientation='both',keep_edge_protection=False,maps=False):
    started=time.perf_counter()
    target,eight,report,cache=cg.process(rgb,alpha,orientation=orientation,keep_edge_protection=keep_edge_protection,maps=True)
    protected,edge_report=protected_edges(rgb)
    final,actual,edge=finish(rgb,target,eight,protected,edge_report)
    report.update(method=METHOD,narrow_edge_output_veto=edge,additional_CNN_calls=0,additional_FFT_calls=0,
        compute_and_encode_seconds=time.perf_counter()-started,original_five_refreshes_per_axis=True,
        color_contract_unchanged=True,references_read=False,ROI_read=False,training_this_round=False)
    if maps:
        cache.update(CG1_target=target,CG1_actual8=eight,edge_protected=protected)
        return final,actual,report,cache
    return final,actual,report
