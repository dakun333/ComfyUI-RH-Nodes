"""CF1: supplement TA4 with locally evidenced, source-following soft contours.

A 32px context is not an unconditional edge halo. Added vetoes must be tangent
to the nearest strong input edge AND to a coherent input contour at the pixel.
Large-angle crossings remain untouched; original CG1 vectors/encoding are frozen.
"""
import os
import time
import cv2,numpy as np
from . import chroma_guard as cg
from . import contour_geometry as ce
from . import contour_geometry as ta
METHOD='CF1_CG1_source_supported_soft_contour_veto'
EDGE_RADIUS_1K=4.0
CONTEXT_RADIUS_1K=32.0
SOURCE_BLUR_SIGMA_1K=3.0
SOURCE_MIN_GRADIENT_RMS_1K=0.5
SOURCE_MIN_COHERENCE=0.60
SOURCE_GRADIENT_DOMINANCE=4.0
MAX_TANGENT_ANGLE_DEGREES=30.0
EPS=np.float32(1e-12)

def prepare_contour_source(rgb,*,context_radius=CONTEXT_RADIUS_1K):
    """Input-only geometry, evaluated before or alongside unchanged restoration."""
    started=time.perf_counter();rgb=np.asarray(rgb,np.float32)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<2:raise ValueError('Expected HxWx3 input, sides >=2')
    if not np.isfinite(rgb).all():raise ValueError('Nonfinite input')
    if context_radius<EDGE_RADIUS_1K:raise ValueError('Invalid context radius')
    scale=min(rgb.shape[:2])/1024.;shape=rgb.shape[:2]
    ix,iy=ta.rgb_gradients(rgb,scale)
    gradient=np.sqrt(np.mean(ix*ix+iy*iy,axis=2));seeds=gradient>=ce.EDGE_THRESHOLD_1K
    if not seeds.any():
        return dict(shape=shape,context_radius=context_radius,no_strong_input_edge=True,seconds=time.perf_counter()-started)
    binary=(~seeds).astype(np.uint8)
    distance=cv2.distanceTransform(binary,cv2.DIST_L2,5)
    near=distance<=EDGE_RADIUS_1K*scale;context=distance<=context_radius*scale
    ex,ey,ec,es=ta.tensor_direction(ix,iy,scale)
    _,labels=cv2.distanceTransformWithLabels(binary,cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    seed_labels=labels[seeds];n=int(labels.max())+1
    lx=np.zeros(n,np.float32);ly=np.zeros(n,np.float32);lc=np.zeros(n,np.float32)
    lx[seed_labels]=ex[seeds];ly[seed_labels]=ey[seeds];lc[seed_labels]=ec[seeds]
    nx=lx[labels];ny=ly[labels];nc=lc[labels]
    smooth=cv2.GaussianBlur(rgb,(0,0),SOURCE_BLUR_SIGMA_1K*scale,borderType=cv2.BORDER_REFLECT_101)
    sx=cv2.Scharr(smooth,cv2.CV_32F,1,0,scale=scale/32.,borderType=cv2.BORDER_REFLECT_101)
    sy=cv2.Scharr(smooth,cv2.CV_32F,0,1,scale=scale/32.,borderType=cv2.BORDER_REFLECT_101)
    ux,uy,uc,us=ta.tensor_direction(sx,sy,scale)
    return dict(shape=shape,context_radius=context_radius,no_strong_input_edge=False,near=near,context=context,distance=distance,nx=nx,ny=ny,nc=nc,ux=ux,uy=uy,uc=uc,us=us,seconds=time.perf_counter()-started)

def contour_protection(rgb,raw_removed,*,source_threshold=SOURCE_MIN_GRADIENT_RMS_1K,context_radius=CONTEXT_RADIUS_1K,diagnostics=False,_source_geometry=None):
    started=time.perf_counter();rgb=np.asarray(rgb,np.float32);removed=np.asarray(raw_removed,np.float32)
    if rgb.shape!=removed.shape or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<2:
        raise ValueError('Expected matching HxWx3 input/removal, sides >=2')
    if not np.isfinite(rgb).all() or not np.isfinite(removed).all():raise ValueError('Nonfinite input/removal')
    if source_threshold<=0 or context_radius<EDGE_RADIUS_1K:raise ValueError('Invalid threshold/context radius')
    shape=rgb.shape[:2];scale=min(shape)/1024.
    source=prepare_contour_source(rgb,context_radius=context_radius) if _source_geometry is None else _source_geometry
    if source['shape']!=shape or source['context_radius']!=context_radius:raise ValueError('Input geometry shape/config mismatch')
    if source['no_strong_input_edge']:
        veto=np.zeros(shape,bool);d=dict(near=veto,context=veto,TA4_veto=veto,soft_veto=veto,added_soft_veto=veto,crossing=veto)
        report=dict(method=METHOD,seconds=time.perf_counter()-started,no_strong_input_edge=True,source_geometry_seconds=source['seconds'])
        return (veto,report,d) if diagnostics else (veto,report)
    near=source['near'];context=source['context'];distance=source['distance'];nx=source['nx'];ny=source['ny'];nc=source['nc'];ux=source['ux'];uy=source['uy'];uc=source['uc'];us=source['us']
    rx,ry=ta.rgb_gradients(removed,scale);dx,dy,dc,ds=ta.tensor_direction(rx,ry,scale)
    cos_ref=np.clip(nx*dx+ny*dy,-1,1)
    cosine_limit=np.float32(np.cos(np.deg2rad(2*MAX_TANGENT_ANGLE_DEGREES)))-np.float32(1e-5)
    confident=(nc>=ta.MIN_DIRECTION_COHERENCE)&(dc>=ta.MIN_DIRECTION_COHERENCE)&(ds>=ta.MIN_REMOVAL_GRADIENT_RMS_1K)
    tangent=cos_ref>=cosine_limit;ta4=near&confident&tangent
    cos_source_removal=np.clip(ux*dx+uy*dy,-1,1);cos_source_ref=np.clip(ux*nx+uy*ny,-1,1)
    source_support=(uc>=SOURCE_MIN_COHERENCE)&(us>=source_threshold)&(us>=SOURCE_GRADIENT_DOMINANCE*ds)
    source_aligned=(cos_source_removal>=cosine_limit)&(cos_source_ref>=cosine_limit)
    soft=context&confident&tangent&source_support&source_aligned;added=soft&~ta4;veto=ta4|soft;crossing=context&confident&~tangent
    assert not np.any(veto&crossing)
    report=dict(method=METHOD,TA4_radius_1K_pixels=EDGE_RADIUS_1K,context_radius_1K_pixels=float(context_radius),context_is_not_unconditional_protection=True,
        source_blur_sigma_1K=SOURCE_BLUR_SIGMA_1K,source_min_gradient_RMS_1K=float(source_threshold),source_min_coherence=SOURCE_MIN_COHERENCE,source_gradient_dominance=SOURCE_GRADIENT_DOMINANCE,
        maximum_tangent_angle_degrees=MAX_TANGENT_ANGLE_DEGREES,scale_convention='short side / 1024',base_TA4_veto_percent=float(ta4.mean()*100),added_source_contour_veto_percent=float(added.mean()*100),
        total_veto_percent=float(veto.mean()*100),confident_crossing_retained_percent=float(crossing.mean()*100),source_input_at_pixel_is_required=True,
        continuous_pre_color_pre_8bit_removal_orientation=True,x16_image_used=False,reference_read=False,ROI_read=False,additional_CNN_calls=0,additional_FFT_calls=0,seconds=time.perf_counter()-started,source_geometry_seconds=source['seconds'])
    if diagnostics:return veto,report,dict(near=near,context=context,distance=distance,TA4_veto=ta4,soft_veto=soft,added_soft_veto=added,crossing=crossing,source_support=source_support,source_aligned=source_aligned,
        source_gradient_RMS=us,source_coherence=uc,cosine_source_removal=cos_source_removal,cosine_reference=cos_ref,cosine_source_reference=cos_source_ref,removal_coherence=dc,removal_gradient_RMS=ds)
    return veto,report

def finish(rgb,cg_target,cg8,veto,geometry_report=None):
    target,eight,report=ce.finish(rgb,cg_target,cg8,veto,geometry_report)
    report.update(method=METHOD,original_CG1_vector_unchanged=True,TA4_protection_retained=True,
        limitation='Local source contour support is not semantic ground truth; true waves running along a real contour can still be protected. Outside this context false corrections are unchanged.')
    return target,eight,report

def process(rgb,alpha,*,orientation='both',keep_edge_protection=False,maps=False):
    started=time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='CPU2-source-geometry') as pool:
        future=pool.submit(prepare_contour_source,rgb)
        target,eight,report,cache=cg.process(rgb,alpha,orientation=orientation,keep_edge_protection=keep_edge_protection,maps=True)
        guard_started=time.perf_counter();source=future.result();wait_seconds=time.perf_counter()-guard_started
        veto,geometry=contour_protection(rgb,np.asarray(rgb,np.float32)-cache['raw_target'],_source_geometry=source)
        geometry['source_geometry_overlap_wait_seconds']=wait_seconds
        geometry['post_CG1_guard_wall_seconds']=time.perf_counter()-guard_started
        geometry['input_geometry_overlapped']=True
    final,actual,guard=finish(rgb,target,eight,veto,geometry)
    report.update(method=METHOD,source_contour_veto=guard,added_deletion_passes=0,additional_CNN_calls=0,additional_FFT_calls=0,
        color_contract_unchanged=True,original_five_refreshes_each_axis=True,compute_and_encode_seconds=time.perf_counter()-started,references_read=False,ROI_read=False,training_this_round=False)
    if maps:
        cache.update(CG1_target=target,CG1_actual8=eight,contour_veto=veto)
        return final,actual,report,cache
    return final,actual,report

