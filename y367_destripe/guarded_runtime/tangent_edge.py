"""TA4: veto only coherent edge-following corrections inside CE4's narrow band.

Original CG1 inference/encoding/palette is frozen. Direction is estimated from
continuous pre-color, pre-quantization RGB removal, not the colorful x16 image.
Near-edge crossing or direction-ambiguous correction is left exactly CG1.
"""
import os
import time
import cv2
import numpy as np
try:
    from . import native_analysis as _analysis_native
except ImportError:
    _analysis_native=None
from . import chroma_guard as cg
from . import edge_output as ce4
METHOD='TA4_CG1_near_edge_tangent_only_veto'
EDGE_RADIUS_1K=ce4.EDGE_RADIUS_1K
EDGE_THRESHOLD_1K=ce4.EDGE_THRESHOLD_1K
BLUR_SIGMA_1K=ce4.BLUR_SIGMA_1K
TENSOR_SIGMA_1K=2.0
MAX_TANGENT_ANGLE_DEGREES=30.0
MIN_DIRECTION_COHERENCE=0.35
MIN_REMOVAL_GRADIENT_RMS_1K=0.01
EPS=np.float32(1e-12)


def rgb_gradients(field,scale):
    smooth=cv2.GaussianBlur(np.asarray(field,np.float32),(0,0),BLUR_SIGMA_1K*scale,
        borderType=cv2.BORDER_REFLECT_101)
    gx=cv2.Scharr(smooth,cv2.CV_32F,1,0,scale=scale/32.,borderType=cv2.BORDER_REFLECT_101)
    gy=cv2.Scharr(smooth,cv2.CV_32F,0,1,scale=scale/32.,borderType=cv2.BORDER_REFLECT_101)
    return gx,gy


def tensor_direction(gx,gy,scale):
    """Unsigned normal direction in doubled-angle coordinates and coherence.

    RGB squared gradients are summed; no channel/hue-specific direction rule.
    A scalar sign flip or a RGB color-direction change cannot flip a line normal.
    """
    if _analysis_native is not None and gx.dtype==gy.dtype==np.float32 and gx.ndim==gy.ndim==3 and gx.shape==gy.shape and gx.shape[2]==3:
        jxx,jyy,jxy=_analysis_native.tensor_moments(gx,gy)
    else:
        jxx=np.mean(gx*gx,axis=2);jyy=np.mean(gy*gy,axis=2);jxy=np.mean(gx*gy,axis=2)
    sigma=TENSOR_SIGMA_1K*scale
    jxx=cv2.GaussianBlur(jxx,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)
    jyy=cv2.GaussianBlur(jyy,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)
    jxy=cv2.GaussianBlur(jxy,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)
    if _analysis_native is not None and jxx.dtype==jyy.dtype==jxy.dtype==np.float32:
        return _analysis_native.tensor_finish(jxx,jyy,jxy)
    x=jxx-jyy;y=np.float32(2)*jxy
    magnitude=np.sqrt(x*x+y*y);trace=jxx+jyy
    return x/np.maximum(magnitude,EPS),y/np.maximum(magnitude,EPS),magnitude/np.maximum(trace,EPS),np.sqrt(np.maximum(trace,0))


def tangent_protection(rgb,raw_removed,*,radius=EDGE_RADIUS_1K,threshold=EDGE_THRESHOLD_1K,
                       max_angle=MAX_TANGENT_ANGLE_DEGREES,
                       min_coherence=MIN_DIRECTION_COHERENCE,diagnostics=False):
    """Same native proximity as CE4, then alignment of local line tangents.

    Edge direction is propagated from the nearest strong INPUT edge. Removal
    direction uses the continuous unfiltered model correction. Normals aligned
    within max_angle imply tangents aligned as well. Large-angle/orthogonal
    crossing and ambiguous direction do NOT receive this extra output veto.
    """
    started=time.perf_counter();rgb=np.asarray(rgb,np.float32)
    removed=np.asarray(raw_removed,np.float32)
    if rgb.shape!=removed.shape or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<2:
        raise ValueError('Expected matching HxWx3 input/removal, sides >=2')
    if radius<0 or threshold<=0 or not 0<=max_angle<90 or not 0<=min_coherence<=1:
        raise ValueError('Invalid radius, gradient threshold, angle or coherence')
    if not np.isfinite(rgb).all() or not np.isfinite(removed).all():raise ValueError('Nonfinite input/removal')
    scale=min(rgb.shape[:2])/1024.
    ix,iy=rgb_gradients(rgb,scale)
    gradient=np.sqrt(np.mean(ix*ix+iy*iy,axis=2));seeds=gradient>=threshold
    shape=rgb.shape[:2]
    if seeds.any():
        # Unlabelled CE4 transform defines the band, preserving its exact geometry.
        binary=(~seeds).astype(np.uint8)
        distance=cv2.distanceTransform(binary,cv2.DIST_L2,5)
        near=distance<=radius*scale
        ex,ey,ecoherence,estrength=tensor_direction(ix,iy,scale)
        _,labels=cv2.distanceTransformWithLabels(binary,cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
        seed_labels=labels[seeds]
        if (seed_labels<=0).any() or np.unique(seed_labels).size!=int(seeds.sum()):
            raise ValueError('Nearest-edge pixel labels are not unique')
        n=int(labels.max())+1
        lx=np.zeros(n,np.float32);ly=np.zeros(n,np.float32);lc=np.zeros(n,np.float32)
        lx[seed_labels]=ex[seeds];ly[seed_labels]=ey[seeds];lc[seed_labels]=ecoherence[seeds]
        nearest_x=lx[labels];nearest_y=ly[labels];nearest_coherence=lc[labels]
        rx,ry=rgb_gradients(removed,scale)
        nx,ny,rcoherence,rstrength=tensor_direction(rx,ry,scale)
        cosine2=np.clip(nearest_x*nx+nearest_y*ny,-1,1)
        confident=(nearest_coherence>=min_coherence)&(rcoherence>=min_coherence)&(rstrength>=MIN_REMOVAL_GRADIENT_RMS_1K)
        aligned=cosine2>=np.float32(np.cos(np.deg2rad(2*max_angle)))-np.float32(1e-5)
        veto=near&confident&aligned
        crossing=near&confident&~aligned
        ambiguous=near&~confident
        assert not np.any(veto&~near)
    else:
        distance=np.full(shape,np.inf,np.float32);near=np.zeros(shape,bool);veto=near.copy()
        confident=near.copy();crossing=near.copy();ambiguous=near.copy()
        cosine2=np.zeros(shape,np.float32);nearest_coherence=cosine2.copy()
        rcoherence=cosine2.copy();rstrength=cosine2.copy()
    report=dict(method=METHOD,radius_1K_pixels=float(radius),radius_native_pixels=float(radius*scale),
        input_edge_gradient_threshold_RGB_levels_1K=float(threshold),maximum_tangent_angle_degrees=float(max_angle),
        minimum_direction_coherence=float(min_coherence),minimum_removed_gradient_RMS_1K=float(MIN_REMOVAL_GRADIENT_RMS_1K),
        removal_preblur_sigma_1K=float(BLUR_SIGMA_1K),direction_tensor_sigma_1K=float(TENSOR_SIGMA_1K),
        scale_convention='short side / 1024',strong_input_edge_seed_percent=float(seeds.mean()*100),
        CE4_near_edge_area_percent=float(near.mean()*100),tangent_veto_area_percent=float(veto.mean()*100),
        confidently_crossing_area_percent=float(crossing.mean()*100),ambiguous_near_edge_area_percent=float(ambiguous.mean()*100),
        orientation_uses_continuous_RGB_removal_before_color_gate_and_native8=True,
        orientation_from_x16_display=False,edge_direction_from_nearest_input_edge=True,
        crossing_or_ambiguous_left_CG1=True,veto_is_subset_of_CE4=True,
        references_read=False,ROI_read=False,extra_CNN_calls=0,extra_FFT_calls=0,
        seconds=time.perf_counter()-started)
    if diagnostics:
        return veto,report,dict(near=near,distance=distance,seeds=seeds,input_gradient=gradient,
            cosine2=cosine2,edge_coherence=nearest_coherence,removal_coherence=rcoherence,
            removal_gradient_RMS=rstrength,confident=confident,crossing=crossing,ambiguous=ambiguous)
    return veto,report


def finish(rgb,cg_target,cg8,veto,geometry_report=None):
    target,eight,report=ce4.finish(rgb,cg_target,cg8,veto,geometry_report)
    report.update(method=METHOD,only_tangent_near_edge_veto=True,
        limitation='Crossing permission is not correctness proof. Along-edge real stripes may still be blocked; colorful false correction away from edges is not solved.')
    return target,eight,report


def process(rgb,alpha,*,orientation='both',keep_edge_protection=False,maps=False):
    started=time.perf_counter()
    cg_target,cg8,report,cache=cg.process(rgb,alpha,orientation=orientation,
        keep_edge_protection=keep_edge_protection,maps=True)
    raw_removed=np.asarray(rgb,np.float32)-cache['raw_target']
    veto,geometry=tangent_protection(rgb,raw_removed)
    target,eight,guard=finish(rgb,cg_target,cg8,veto,geometry)
    report.update(method=METHOD,tangent_edge_output_veto=guard,
        added_deletion_passes=0,additional_CNN_calls=0,additional_FFT_calls=0,
        color_contract_unchanged=True,original_five_refreshes_each_axis=True,
        compute_and_encode_seconds=time.perf_counter()-started,
        references_read=False,ROI_read=False,training_this_round=False)
    if maps:
        cache.update(CG1_target=cg_target,CG1_actual8=cg8,tangent_edge_veto=veto)
        return target,eight,report,cache
    return target,eight,report
