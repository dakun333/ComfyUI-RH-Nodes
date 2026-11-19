"""Shared source geometry and whole-pixel finish used by CF1 protection."""
import time
import cv2
import numpy as np
try:
    from . import native_analysis as _analysis_native
except ImportError:
    _analysis_native = None
try:
    from .native_output import finish_edges as _native_finish
except ImportError:
    _native_finish = None

EDGE_THRESHOLD_1K = 4.0
EDGE_RADIUS_1K = 4.0
BLUR_SIGMA_1K = 1.0
TENSOR_SIGMA_1K = 2.0
MIN_DIRECTION_COHERENCE = 0.35
MIN_REMOVAL_GRADIENT_RMS_1K = 0.01
EPS = np.float32(1e-12)


def rgb_gradients(field, scale):
    smooth = cv2.GaussianBlur(np.asarray(field, np.float32), (0, 0), BLUR_SIGMA_1K*scale,
                             borderType=cv2.BORDER_REFLECT_101)
    gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0, scale=scale/32., borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1, scale=scale/32., borderType=cv2.BORDER_REFLECT_101)
    return gx, gy


def tensor_direction(gx, gy, scale):
    if _analysis_native is not None and gx.dtype==gy.dtype==np.float32 and gx.ndim==gy.ndim==3 and gx.shape==gy.shape and gx.shape[2]==3:
        jxx,jyy,jxy = _analysis_native.tensor_moments(gx,gy)
    else:
        jxx=np.mean(gx*gx,axis=2);jyy=np.mean(gy*gy,axis=2);jxy=np.mean(gx*gy,axis=2)
    sigma = TENSOR_SIGMA_1K*scale
    jxx=cv2.GaussianBlur(jxx,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)
    jyy=cv2.GaussianBlur(jyy,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)
    jxy=cv2.GaussianBlur(jxy,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)
    if _analysis_native is not None and jxx.dtype==jyy.dtype==jxy.dtype==np.float32:
        return _analysis_native.tensor_finish(jxx,jyy,jxy)
    x=jxx-jyy;y=np.float32(2)*jxy
    magnitude=np.sqrt(x*x+y*y);trace=jxx+jyy
    return x/np.maximum(magnitude,EPS),y/np.maximum(magnitude,EPS),magnitude/np.maximum(trace,EPS),np.sqrt(np.maximum(trace,0))


def finish(rgb,target,candidate8,protected,edge_report=None):
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
    report=dict(method='CE4_CG1_narrow_strong_edge_veto',edge_detector=edge_report,
        candidate_changed8_pixels=int(active_count),edge_changed8_pixels_restored=int(restored),
        unprotected_output_exact_CG1=True,protected_output_exact_input=True,
        no_channel_projection_or_colorization=True,finish_seconds=time.perf_counter()-started,
        limitation='A target stripe can also cross an object edge and remain in this band; outside-edge colorful false correction is not solved.')
    return effective,eight,report
