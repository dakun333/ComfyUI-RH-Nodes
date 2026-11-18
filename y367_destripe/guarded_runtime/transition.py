"""ST1/ST2: attenuate only policy-created discontinuities on the admitted side.
The original model, five passes, protected pixels and unmodified output bytes stay intact.
"""
import time
import numpy as np,cv2
from . import chroma_guard as cg
from . import contour_edge, edge_output
from .codec import directed_round
from .native_transition import boundary_seeds,seed_lookup,gain_and_target,finish_retreat

def transition(rgb,target,eight,raw_removed,forced,*,width_1k=6.,seed_policy=None,maps=False):
    started=time.perf_counter()
    if width_1k<=0:raise ValueError('Positive transition width required')
    if rgb.dtype!=np.float32 or target.dtype!=np.float32 or eight.dtype!=np.uint8:raise ValueError('Native FLOAT32 / actual8 required')
    if raw_removed.shape!=rgb.shape or forced.shape!=rgb.shape[:2]:raise ValueError('Shape mismatch')
    seed_policy=forced if seed_policy is None else seed_policy
    seeds,severity=boundary_seeds(rgb,raw_removed,target,forced,seed_policy)
    if not seeds.any():
        report=dict(width_1k=float(width_1k),seed_pixels=0,affected_FLOAT_pixels=0,changed_actual8_pixels=0,seconds=time.perf_counter()-started)
        return (target,eight,report,dict(gain=np.ones(forced.shape,np.float32),affected=np.zeros(forced.shape,bool),seeds=seeds)) if maps else (target,eight,report)
    distance,labels=cv2.distanceTransformWithLabels(1-seeds,cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    colors,weights=seed_lookup(rgb,seeds,labels,severity,int(labels.max())+1)
    width=width_1k*min(rgb.shape[:2])/1024.
    gain,scaled,affected=gain_and_target(rgb,target,forced,distance,labels,colors,weights,width)
    # Re-encode only modified runs. All other actual8 bytes retain the frozen baseline.
    proposed=directed_round(scaled,affected,eight)
    guarded,q,guard_report,admitted=cg.guard_output(rgb,scaled,proposed)
    cancelled,changed,error=finish_retreat(rgb,target,eight,guarded,q,affected)
    assert error<1.00001
    report=dict(method='ST_single_sided_forced_boundary_transition',width_1k=float(width_1k),width_native=float(width),seed_pixels=int(seeds.sum()),affected_FLOAT_pixels=int(affected.sum()),affected_FLOAT_percent=float(affected.mean()*100),changed_actual8_pixels=int(changed),changed_actual8_percent=float(changed/affected.size*100),monotone_actual8_retreat=True,rounding_overshoot_pixels_reverted=int(cancelled),new_quantization_rejects=int((affected&~admitted).sum()),seconds=time.perf_counter()-started,extra_CNN_calls=0,extra_FFT_calls=0,source_and_raw_continuity_at_seed=True,only_originally_admitted_side=True,protected_or_rejected_pixels_unmodified=True,outside_transition_FLOAT_and_actual8_exact=True,whole_RGB_scalar_gain_only=True,actual8_palette_chroma_guard_retained=True,additional_restoration_pass=False,limitation='Amplitude retreat, not correction truth: can retain real stripes inside the transition band; new 8-bit admissions can be sparse.')
    if maps:return guarded,q,report,dict(gain=gain,affected=affected,seeds=seeds,distance=distance,admitted=admitted)
    return guarded,q,report

def forced_from_cache(rgb,cache,veto):
    raw=rgb-cache['raw_target'];prior8=cache['previous_actual8']
    # Quantization-only natural zero does not become a transition seed.
    rejected_integer=(~cache['output_admitted'])&np.any(prior8.astype(np.float32)!=rgb,axis=2)
    forced=veto|(~cache['color_allowed'])|(~cg.chroma_mask(raw))|rejected_integer
    return raw,forced

def process(rgb,alpha,*,method='CF1',width_1k=6.,orientation='both',keep_edge_protection=False,maps=False):
    started=time.perf_counter()
    if method=='CF1':
        original = contour_edge.process
    elif method=='CE4':
        original = edge_output.process
    else:raise ValueError('Choose CF1 or CE4')
    target,eight,report,cache=original(rgb,alpha,orientation=orientation,keep_edge_protection=keep_edge_protection,maps=True)
    veto=cache['contour_veto'] if method=='CF1' else cache['edge_protected']
    raw,forced=forced_from_cache(rgb,cache,veto)
    seed_policy=veto|(~cache['color_allowed'])|(~cg.admission_mask(raw))
    final,q,tr,d=transition(rgb,target,eight,raw,forced,width_1k=width_1k,seed_policy=seed_policy,maps=True)
    report.update(method=method+'_ST',boundary_transition=tr,compute_and_encode_seconds=time.perf_counter()-started,quality_policy_changed_only_by_local_amplitude_retreat=True)
    if maps:
        cache.update(pre_transition_target=target,pre_transition_actual8=eight,transition_gain=d['gain'],transition_affected=d['affected'],forced_zero=forced)
        return final,q,report,cache
    return final,q,report