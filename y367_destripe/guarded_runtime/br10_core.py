"""Y367-sized response network; reconstruct on original analysis pixels. Five passes."""
import os
from . import cpu_runtime as cpu4
from pathlib import Path
import sys,time,json
import numpy as np,cv2
ROOT=Path(__file__).absolute().parent;MODEL=ROOT/'model.npz'
from . import envelope as e
try:
 from . import native_pipeline as _pipeline_native
except ImportError:
 _pipeline_native=None
from .robust_features import source_context, features
from .component_evidence import component_evidence
from .codec import encode_directed8
from .recurrence import recurrence
try:
 from . import native_iteration as _iteration_native
except ImportError:
 _iteration_native=None
ITERATION_FUSION=True
FIXED_POINT_REUSE=True


CONFIG=dict(score_low=.25,score_full=.60,amplitude_low=.10,amplitude_full=.35,permit_power=.5,max_additional=1.0,minimum_additional=.005,phase_low=.65,phase_full=.95,recurrence_along=.80,confidence_floor=.02,permit_gain=1.4)

def ramp(q,lo,hi):return np.clip((q-lo)/(hi-lo),0,1).astype(np.float32)
def lab_j(geom):
 guide=np.clip((geom[...,:3]+.5)*255,0,255);J=np.empty((*guide.shape[:2],3,3),np.float32)
 for k in range(3):
  lo=guide.copy();hi=guide.copy();lo[...,k]=np.maximum(0,lo[...,k]-.5);hi[...,k]=np.minimum(255,hi[...,k]+.5)
  J[...,k]=(e._lab(hi)-e._lab(lo))/(hi[...,k]-lo[...,k])[...,None]
 return J

def fine_transform(f):
 w,h=f['analysis_shape'];geom=cv2.resize(f['geom'],(w,h),interpolation=cv2.INTER_LINEAR)
 return geom,e.perceptual_transform(geom)

def reconstruct_response(rgb,alpha,coeff,f,f0,fine,phases=None,*,noise_amp=None,fit_magnitude=False,return_layer=False):
 """Coefficients are regional; measured complex amplitudes keep analysis resolution.
 No new FFT, CNN call, deletion pass, mask rule or external reference.
 """
 geom,transform=fine;w,h=f['analysis_shape'];co=cv2.resize(coeff,(w,h),interpolation=cv2.INTER_LINEAR)
 raw=e.decode_coeff(co,f['fine_amp'])
 if fit_magnitude:
  if noise_amp is None:raise ValueError('Measured residual is required for amplitude fit')
  J,inv=transform;rr=e.matvec(J,raw);aa=e.matvec(J,noise_amp)
  # Unit-phase alignment only: keep the learned response magnitude/confidence. Do not normalize a tiny rejected proposal into a full deletion.
  for sl in (slice(0,1),slice(1,3)):
   cross=np.sum(aa[...,sl]*np.conj(rr[...,sl]),axis=2);gain=cross/np.maximum(abs(cross),1e-12);rr[...,sl]*=gain[...,None]
  raw=e.matvec(inv,rr).astype(np.complex64)
 prediction,projection=e.perceptual_passivity(noise_amp if fit_magnitude else f['fine_amp'],raw,geom,transform=transform,transformed_amp=aa if fit_magnitude else None)
 if phases is not None:
  J,inv=transform;pred=e.matvec(J,prediction)
  for phase,sl in zip(phases,(slice(0,1),slice(1,3))):pred[...,sl]*=cv2.resize(phase,(w,h),interpolation=cv2.INTER_LINEAR)[...,None]
  prediction=e.matvec(inv,pred).astype(np.complex64)
 if fit_magnitude:
  # The natural source component has already been removed from the measured band;
  # existing source/recurrence authorization is applied by the caller. Do not also
  # multiply this residual by the distance-style original edge halo.
  H,W=rgb.shape[:2];phase=e._native_phase(W,f0['configuration']['cycles']);layer=e.reconstruct_layer(prediction,(W,H),phase);layer[alpha<.95]=0
  if return_layer:return layer,projection
  return e.compose(rgb,layer,alpha,f0['configuration']['limit']),projection
 return e.reconstruct(rgb,alpha,prediction,f0),projection

def prepare(rgb,alpha,orientation='both',weights_path=None,*,edge_protection=True):
 start=time.perf_counter();rgb=np.asarray(rgb,np.float32);alpha=np.asarray(alpha,np.float32)
 if rgb.ndim!=3 or rgb.shape[2]!=3 or alpha.shape!=rgb.shape[:2] or min(rgb.shape[:2])<2:raise ValueError('Invalid RGB/alpha dimensions')
 if not np.isfinite(rgb).all() or not np.isfinite(alpha).all() or rgb.min()<0 or rgb.max()>255 or alpha.min()<0 or alpha.max()>1:raise ValueError('Nonfinite or invalid input range')
 if orientation not in ('vertical','horizontal','both'):raise ValueError('Invalid direction')
 with np.load(Path(weights_path) if weights_path is not None else MODEL) as n:
  weights={k:n[k] for k in n.files if k!='metadata'};model_meta=json.loads(str(n['metadata']))
 def prepare_branch(direction):
  if e._hot is not None:e._hot.configure_threads()
  trans=direction=='horizontal';a=np.ascontiguousarray(rgb.transpose(1,0,2)) if trans else rgb;al=np.ascontiguousarray(alpha.T) if trans else alpha
  analysis_alpha=(al>=.95).astype(np.float32)
  f=e.extract(a,analysis_alpha,dict(passivity=True,edge_radius=.5,edge_protection=edge_protection),cache_image=True)
  if 48/f['analysis_shape'][0]>=.4:raise ValueError('Aspect too narrow')
  comp,image,companion=source_context(a,f);co=e.infer_numpy(features(f['amp'],f['geom'],comp),weights);raw=e.decode_coeff(co,f['amp']);transform=e.perceptual_transform(f['geom']);pred,rep=e.perceptual_passivity(f['amp'],raw,f['geom'],transform=transform)
  support,_,detail=component_evidence(a,al,f,co,raw,diagnostics=True,cache=(image,companion));fine=fine_transform(f);decoded,fine_projection=reconstruct_response(a,al,co,f,f,fine);base=(a-decoded)*support[...,None]
  base[al<.95]=0
  rec,rd=recurrence(a,f,amplitude=(CONFIG['amplitude_low'],CONFIG['amplitude_full']),score_range=(CONFIG['score_low'],CONFIG['score_full']),along=CONFIG['recurrence_along'])
  permit=np.clip(ramp(rec,CONFIG['confidence_floor'],1.)**CONFIG['permit_power']*ramp(detail['support_grid'],.45,.95)*CONFIG['permit_gain'],0,1)
  pn=cv2.resize(permit,(a.shape[1],a.shape[0]),interpolation=cv2.INTER_LINEAR);pn[al<.95]=0
  decoded_matched,_=reconstruct_response(a,al,co,f,f,fine,noise_amp=f['fine_amp']-detail['fine_structure_amp'],fit_magnitude=True)
  matched=(a-decoded_matched)*support[...,None];base=base+(matched-base)*pn[...,None];base[al<.95]=0
  native_base=np.ascontiguousarray(base.transpose(1,0,2)) if trans else base
  branch=dict(direction=direction,trans=trans,rgb=a,alpha=al,analysis_alpha=analysis_alpha,f=f,base=base,structure_amp=detail['structure_amp'],original_noise=f['amp']-detail['structure_amp'],fine=fine,fine_structure_amp=detail['fine_structure_amp'],permit=pn,permit_grid=permit,J=transform[0],J_inv=transform[1])
  return branch,native_base
 directions=['vertical','horizontal'] if orientation=='both' else [orientation]
 if len(directions)==2:
  from concurrent.futures import ThreadPoolExecutor
  with ThreadPoolExecutor(max_workers=2,thread_name_prefix='Y367-prepare') as pool:items=list(pool.map(prepare_branch,directions))
 else:items=[prepare_branch(directions[0])]
 branches=[x[0] for x in items];layers=[x[1] for x in items]
 baseline=e.compose(rgb,sum(layers),alpha,8.)
 return baseline,dict(rgb=rgb,alpha=alpha,branches=branches,weights=weights,model_meta=model_meta,prepare_seconds=time.perf_counter()-start)

def _finish_branch(b,weights,method,steps):
 if e._hot is not None:e._hot.configure_threads()
 if method!='refresh':raise ValueError('BR10 only supports frozen-count refresh')
 a=b['rgb'];al=b['alpha'];f0=b['f'];J=b['J'];orig=e.matvec(J,b['original_noise']);orig_power=[np.sum(abs(orig[...,sl])**2,axis=2) for sl in (slice(0,1),slice(1,3))];extra=np.zeros_like(a);track=[];executed=0;reused=0
 fused=ITERATION_FUSION and _iteration_native is not None and e._native is not None
 for k in range(steps):
  current=(e._hot.compose_added(a,b['base'],extra,al,8.) if cpu4.HOT_KERNELS and e._hot is not None else e._native.compose_added(a,b['base'],extra,al,8.)) if e._native is not None else e.compose(a,b['base']+extra,al,8.)
  f=e.extract(current,b['analysis_alpha'],dict(passivity=True,edge_radius=.5),amp_only=True,cache_image=True)
  comp,_,_=source_context(current,f);co=e.infer_numpy(features(f['amp'],f0['geom'],comp),weights)
  left=e.matvec(J,f['amp']-b['structure_amp']);phases=[]
  for channel,sl in enumerate((slice(0,1),slice(1,3))):
   oo=orig[...,sl];rr=left[...,sl];num=np.real(np.sum(rr*np.conj(oo),axis=2));den=np.sqrt(np.sum(abs(rr)**2,axis=2)*orig_power[channel])+1e-12
   phases.append(ramp(num/den,CONFIG['phase_low'],CONFIG['phase_full']))
  before=extra.copy() if FIXED_POINT_REUSE and not fused else None
  decoded,projection=reconstruct_response(a,al,co,f,f0,b['fine'],phases,noise_amp=f['fine_amp']-b['fine_structure_amp'],fit_magnitude=True,return_layer=fused)
  if fused:
   accumulate=e._hot.compose_accumulate if cpu4.HOT_KERNELS and e._hot is not None else _iteration_native.compose_accumulate
   unchanged=accumulate(extra,a,decoded,b['permit'],al,f0['configuration']['limit'],CONFIG['max_additional'],CONFIG['minimum_additional'])
  elif e._native is not None:e._native.add_difference_and_cap_extra(extra,a,decoded,b['permit'],al,CONFIG['max_additional'],CONFIG['minimum_additional'])
  else:
   native=(a-decoded)*b['permit'][...,None]
   extra+=native;mag=np.max(abs(extra),axis=2);extra*=np.minimum(1,CONFIG['max_additional']/np.maximum(mag,1e-7))[...,None]
   extra[np.max(abs(extra),axis=2)<CONFIG['minimum_additional']]=0;extra[al<.95]=0
  if cpu4.HOT_KERNELS and decoded.dtype==np.float32 and decoded.flags.c_contiguous and (decoded.flags.owndata or (fused and cpu4.CARRIER_FUSION and e._hot is not None)):
   # Disposable reconstruction buffer; retain NumPy's historical reduction order.
   np.multiply(extra,extra,out=decoded);extra_rms=float(np.sqrt(np.mean(decoded)))
   np.abs(extra,out=decoded);extra_max=float(decoded.max())
  else:extra_rms=float(np.sqrt(np.mean(extra**2)));extra_max=float(abs(extra).max())
  track.append(dict(step=k+1,phase_mean=[float(p.mean()) for p in phases],projection=projection,projection_resolution='original_analysis_pixels',extra_RMS=extra_rms,extra_max=extra_max))
  executed+=1
  if FIXED_POINT_REUSE:
   if not fused:unchanged=np.array_equal(before.view(np.uint32),extra.view(np.uint32))
   if unchanged:
    # The full recurrence state is identical, not merely similar. All other
    # inputs/weights/geometry are frozen, so the next deterministic map is the
    # same. Preserve the logical five-step report, reuse the stable result.
    for j in range(k+1,steps):
     reused_step=track[-1].copy();reused_step['step']=j+1;track.append(reused_step)
    reused=steps-k-1
    break
 if b['trans']:extra=np.ascontiguousarray(extra.transpose(1,0,2))
 return extra,dict(direction=b['direction'],permit_mean=float(b['permit'].mean()),steps=track,executed_refreshes=executed,exact_fixed_point_reused_refreshes=reused,fused_compose_accumulate=bool(fused),approximate_early_stop=False)

def finish(cache,baseline,method='noise',steps=1):
 start=time.perf_counter()
 if len(cache['branches'])==2:
  from concurrent.futures import ThreadPoolExecutor
  with ThreadPoolExecutor(max_workers=2,thread_name_prefix='rx5-axis') as pool:
   items=list(pool.map(lambda b:_finish_branch(b,cache['weights'],method,steps),cache['branches']))
 else:items=[_finish_branch(b,cache['weights'],method,steps) for b in cache['branches']]
 extras=[q[0] for q in items];reports=[q[1] for q in items]
 if e._native is not None:
  target,total_max,additional_max=e._native.finish_total(cache['rgb'],cache['alpha'],baseline,extras[0],extras[1] if len(extras)==2 else extras[0],len(extras)==2,CONFIG['max_additional'],8.)
  assert np.isfinite(target).all() and total_max<=8.0002 and additional_max<=CONFIG['max_additional']+1e-4
 else:
  raw=sum(extras);mag=np.max(abs(raw),axis=2);raw*=np.minimum(1,CONFIG['max_additional']/np.maximum(mag,1e-7))[...,None]
  active=np.any(raw!=0,axis=2);target=baseline.copy();composed=e.compose(cache['rgb'],cache['rgb']-baseline+raw,cache['alpha'],8.);target[active]=composed[active]
  # Color gamut limiting can reduce the requested correction; enforce the additive cap on the actual final change.
  delta=target-baseline;mag=np.max(abs(delta),axis=2);target=baseline+delta*np.minimum(1,CONFIG['max_additional']/np.maximum(mag,1e-7))[...,None]
  target[~active]=baseline[~active];target[cache['alpha']<.95]=cache['rgb'][cache['alpha']<.95]
  assert np.isfinite(target).all() and np.max(abs(target-cache['rgb']))<=8.0002 and np.max(abs(target-baseline))<=CONFIG['max_additional']+1e-4
 return target,dict(method=method,steps=steps,seconds=time.perf_counter()-start,configuration=CONFIG,branches=reports)

def encode(rgb,baseline,target,prior8=None):
 if _pipeline_native is not None and rgb.dtype==baseline.dtype==target.dtype==np.float32 and rgb.shape==baseline.shape==target.shape and rgb.ndim==3 and rgb.shape[2]==3 and (prior8 is None or (prior8.dtype==np.uint8 and prior8.shape==target.shape)):
  eight,error_max=_pipeline_native.encode_pair(rgb,baseline,target,prior8)
  assert error_max<1.00001
  return eight
 if prior8 is None:
  prior8=encode_directed8(rgb,baseline);same=np.all(baseline==rgb,axis=2);prior8[same]=np.uint8(np.clip(np.rint(rgb[same]),0,255))
 eight=encode_directed8(rgb,target);same=np.all(target==baseline,axis=2);eight[same]=prior8[same]
 assert np.max(abs(eight.astype(np.float32)-target))<1.00001
 return eight

def apply(rgb,alpha,*,orientation='both',maps=False,weights_path=None,edge_protection=True):
 """Explicit RGB-response weights plus five evidence-limited residual passes, fixed config for all inputs.

 References, masks from the experiments, filenames and source captions are not read.
 The passive carrier/recurrence controls are not a pixelwise clean-image certificate.
 """
 start=time.perf_counter();base,cache=prepare(rgb,alpha,orientation,weights_path,edge_protection=edge_protection)
 target,report=finish(cache,base,'refresh',5)
 report.update(implementation='Y367_BR10_confidence_preserving_phase_alignment',native_CPU_kernels=e._native is not None,model_unchanged=(Path(weights_path).read_bytes()==MODEL.read_bytes()) if weights_path is not None else True,algorithm_unchanged=False, numerical_equivalence_scope='New analysis-resolution reconstruction; not claimed equivalent to Y367',method='Y367-BR10-confidence-preserving-phase-alignment',orientation=orientation,seconds=time.perf_counter()-start,prepare_seconds=cache['prepare_seconds'],refresh_seconds=report['seconds'],model_sha256=__import__('hashlib').sha256((Path(weights_path) if weights_path is not None else MODEL).read_bytes()).hexdigest(),additional_FLOAT_limit=1.0,total_FLOAT_limit=8.0,axis_workers=2 if orientation=='both' else 1,references_read=False,independent_new_weights_fitted=(cache['model_meta'].get('method','').startswith(('BR2_','BR3_','BR5_')) if weights_path is not None else False),new_training_pair_scenes=(cache['model_meta'].get('real_pairs',0) if (cache['model_meta'].get('method','').startswith(('BR2_','BR3_','BR5_')) if weights_path is not None else False) else 0),final_full_data_fit=False,base_model='Original Y367 response checkpoint; five full-analysis phase-aligned residual passes',limitation='The learned response magnitude is retained, but source/repeat evidence remains approximate. Real native8 visual regressions are required; not a clean-image certificate.')
 return (target,report,dict(baseline=base)) if maps else (target,report)
