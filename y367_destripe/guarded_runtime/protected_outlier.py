"""PO1: ST1 + amplitude outlier AND proximity to original edge/contour protection.
No additional detection/restoration iteration; only scalar retreat of admitted RGB.
"""
import time
import numpy as np,cv2
from .native_outlier import magnitude,local_outlier_gain
from .native_transition import finish_retreat
from .codec import directed_round
from . import chroma_guard as cg
from .transition import process as st_process
CONFIGS={'PO1':(2.,3.)}

def outlier_retreat(rgb,st_target,st8,reference_target,protected,*,variant='PO1',maps=False):
    started=time.perf_counter()
    if variant not in CONFIGS:raise ValueError('Choose PO1')
    if rgb.dtype!=np.float32 or st_target.dtype!=np.float32 or st8.dtype!=np.uint8:raise ValueError('Native integer-valued FLOAT32 input, FLOAT32 target, actual8 required')
    if protected.shape!=rgb.shape[:2] or reference_target.shape!=rgb.shape:raise ValueError('Shape mismatch')
    scale=min(rgb.shape[:2])/1024.;radius=np.float32(6*scale);clearance=np.float32(8*scale)
    start_ratio,full_ratio=map(np.float32,CONFIGS[variant])
    current=magnitude(rgb,st_target)
    if protected.any():distance=cv2.distanceTransform((~protected).astype(np.uint8),cv2.DIST_L2,5)
    else:distance=np.full(protected.shape,np.inf,np.float32)
    base=magnitude(rgb,reference_target)
    offsets=np.unique(np.rint(np.arange(-6,7,dtype=np.float32)*np.float32(8*scale)).astype(np.int32))
    scaled,gain,reference,affected,counts=local_outlier_gain(rgb,st_target,st8,current,base,distance,offsets,radius,clearance,start_ratio,full_ratio)
    proposed=directed_round(scaled,affected,st8)
    final,q,guard_report,admitted=cg.guard_output(rgb,scaled,proposed)
    cancelled,changed,error=finish_retreat(rgb,st_target,st8,final,q,affected)
    assert error<1.00001
    report=dict(method='ST1_plus_protected_local_amplitude_outlier_retreat',variant=variant,radius_1k=6.,radius_native=float(radius),reference_radius_1k=48.,reference_clearance_1k=8.,reference_quantile=.9,reference_minimum_native_RGB_level=1.,minimum_reference_samples=12,outlier_ratio_onset=float(start_ratio),outlier_ratio_full=float(full_ratio),affected_FLOAT_pixels=int(affected.sum()),affected_FLOAT_percent=float(affected.mean()*100),changed_actual8_pixels=int(changed),changed_actual8_percent=float(changed/affected.size*100),rounding_overshoot_pixels_reverted=int(cancelled),new_quantization_rejects=int((affected&~admitted).sum()),seconds=time.perf_counter()-started,extra_CNN_calls=0,extra_FFT_calls=0,extra_restoration_passes=0,reference_comes_from_pre_ST1_admitted_output=True,cancelled_and_natural_zero_pixels_excluded_from_reference=True,reference_not_a_clean_image_or_truth_label=True,near_protection_ALONE_not_sufficient=True,high_amplitude_ALONE_not_sufficient=True,only_originally_admitted_side=True,whole_RGB_scalar_gain_only=True,outside_affected_FLOAT_actual8_exact=True,no_actual8_component_more_aggressive_than_ST1=True,actual8_palette_chroma_guard_retained=True,limitation='High and near protection is a risk indicator, not proof of false removal; locally strong real stripes can also be retreated. Sparse references leave a pixel unchanged.')
    if maps:return final,q,report,dict(gain=gain,reference=reference,affected=affected,distance=distance,reference_count=counts,current_magnitude=current,admitted=admitted)
    return final,q,report

def process(rgb,alpha,*,method='CF1',variant='PO1',orientation='both',keep_edge_protection=False,maps=False):
    started=time.perf_counter()
    t,eight,report,cache=st_process(rgb,alpha,method=method,width_1k=6.,orientation=orientation,keep_edge_protection=keep_edge_protection,maps=True)
    protected=cache['contour_veto']
    out,q,orep,d=outlier_retreat(rgb,t,eight,cache['pre_transition_target'],protected,variant=variant,maps=True)
    report.update(method=method+'_ST1_'+variant,protected_outlier=orep,compute_and_encode_seconds=time.perf_counter()-started,quality_policy_change='ST1 plus local amplitude outlier AND original protection proximity')
    if maps:
        cache.update(pre_outlier_target=t,pre_outlier_actual8=eight,outlier_gain=d['gain'],outlier_reference=d['reference'],outlier_affected=d['affected'])
        return out,q,report,cache
    return out,q,report
