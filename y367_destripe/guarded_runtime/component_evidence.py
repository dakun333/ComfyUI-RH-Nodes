"""Classify nonrepeated, locally straight source features, not edge neighborhoods.
Periodic pulses/stripes (including perfectly straight stripes) are retained.
The detected feature's narrow-band contribution is estimated with the identical
carrier operator, then only its aligned portion of the old RGB proposal is vetoed.
"""
import numpy as np,cv2,time
from scipy.ndimage import maximum_filter1d
from .exact_median import median_filter
from .carrier_tools import _shift, _companion_amp

COMPONENT_CONFIG=dict(median_periods=2.,spike_ratio=2.2,spike_floor=.75,
                      peer_peak_ratio=.50,peer_radius=.24,peer_count=2,
                      tensor_across=.16,tensor_along=1.4,
                      coherence_start=.75,coherence_full=.88,
                      vertical_fraction=.85,accept_low=.10,accept_high=.80)


def component_evidence(rgb,alpha,f,coeff,raw_prediction,*,config=None,mode='straight',diagnostics=False,cache=None):
 from . import envelope as base
 start=time.perf_counter();c=dict(COMPONENT_CONFIG);c.update(config or {});w,h=f['analysis_shape'];gw,gh=f['grid_shape'];p=w/f['configuration']['cycles']
 image=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA) if cache is None else cache[0];k=max(3,round(p*c['median_periods']));k+=1-k%2
 companion=median_filter(image,size=(1,k,1),mode='reflect').astype(np.float32) if cache is None else cache[1];r=image-companion
 magnitude=np.sqrt(np.mean(r*r,axis=2));typical=median_filter(magnitude,size=(1,k),mode='reflect')
 spike=magnitude>c['spike_ratio']*typical+c['spike_floor']
 # SIGNED RGB repetition, not absolute magnitude: the two opposite lobes
 # of one natural contour must not count as two repetitions of a stripe.
 peer_count=np.zeros((h,w),np.int8);norm=np.sum(r*r,axis=2)+1e-6
 search=range(-max(1,round(p*c['peer_radius'])),max(1,round(p*c['peer_radius']))+1)
 for offset in (-2.,-1.,1.,2.):
  best=np.zeros((h,w),np.float32)
  for jitter in search:
   peer=_shift(r,offset*p+jitter);alignment=np.sum(peer*r,axis=2)/norm
   best=np.maximum(best,alignment)
  peer_count+=(best>c['peer_peak_ratio'])&(magnitude>c['spike_floor'])
 isolated=spike&(peer_count<c['peer_count'])
 # Tensor is derived from ORIGINAL residual pixels, not the Fourier ringing.
 g=cv2.GaussianBlur(magnitude,(0,0),max(.5,p*.08));gx=np.gradient(g,axis=1);gy=np.gradient(g,axis=0)
 smooth=lambda q:cv2.GaussianBlur(q,(0,0),max(.5,p*c['tensor_across']),sigmaY=max(.5,p*c['tensor_along']))
 xx=smooth(gx*gx);yy=smooth(gy*gy);xy=smooth(gx*gy);trace=xx+yy
 coherence=np.sqrt((xx-yy)**2+4*xy**2)/(trace+1e-8);vertical=xx/(trace+1e-8)
 shape=np.clip((coherence-c['coherence_start'])/(c['coherence_full']-c['coherence_start']),0,1)*(vertical>=c['vertical_fraction'])
 selected=isolated.astype(np.float32)*(shape if mode=='straight' else 1.)
 structure=companion+r*selected[...,None]
 structure_amp,fine_structure_amp=_companion_amp(structure,f,rgb.shape[1],full_output=True);structure_prediction=base.decode_coeff(coeff,structure_amp)
 denominator=np.sum(np.abs(raw_prediction)**2,axis=2)+1e-10
 score=np.real(np.sum((raw_prediction-structure_prediction)*np.conj(raw_prediction),axis=2))/denominator
 support=np.clip((score-c['accept_low'])/(c['accept_high']-c['accept_low']),0,1).astype(np.float32)
 native=cv2.resize(support,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)
 report=dict(mode=mode,configuration=c,seconds=time.perf_counter()-start,
             spike_fraction=float(spike.mean()),isolated_fraction=float(isolated.mean()),classified_straight_fraction=float((selected>0).mean()),
             full_grid_fraction=float((support==1).mean()),rejected_grid_fraction=float((support==0).mean()),
             mechanism='original nonrepeated source component + predicted aligned carrier leakage; no distance halo or full edge core',
             caution='Source decomposition is approximate. Some near-feature true stripes may still be attenuated; periodic real texture remains ambiguous.')
 detail=dict(score=score,support_grid=support)
 if diagnostics:detail.update(selected_source=selected,structure=structure,companion=companion,isolated=isolated,coherence=coherence,peer_count=peer_count,structure_amp=structure_amp,fine_structure_amp=fine_structure_amp)
 return native,report,detail

