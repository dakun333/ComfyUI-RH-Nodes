"""Evidence-only period bounds selection; fixed 3–96px probe (2026-10-23)."""
from dataclasses import replace
import math

import cv2
import numpy as np

from .core import (
    Parameters, Prior, Settings, estimate_color, guided_detail,
    shade_tangent, separate_scalar,
)


def prior_for(low=22., high=80.):
    return Prior(Parameters(period_min=float(low), period_max=float(high)))


def estimate_scale(rgb, include_near_white=False):
    """Estimate a central-energy interval; insufficient evidence means abstain."""
    if not isinstance(include_near_white, bool):raise ValueError('include_near_white must be boolean')
    a=np.asarray(rgb)
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<2:raise ValueError('Expected uint8 RGB')
    src=a.astype(np.float32);settings=Settings(include_near_white=include_near_white);probe=prior_for(3,96)
    color,evidence=estimate_color(src,probe,settings)
    info={'method':'shared_color_spectral_scale','probe_range':[3,96],'color_evidence':evidence,'uses_resolution_ratio':False,'include_near_white':include_near_white}
    if color is None:return None,{**info,'status':'abstained_no_color_evidence'}
    tangent=shade_tangent(cv2.GaussianBlur(src,(0,0),.3*math.sqrt(3*96))).astype(np.float32)
    scalar=separate_scalar(guided_detail(src,probe.base),tangent,color,.15)
    scalar=cv2.GaussianBlur(scalar,(1,0),sigmaX=0,sigmaY=1.,borderType=cv2.BORDER_REFLECT)
    low=cv2.GaussianBlur(src,(0,0),2.);gy,gx=np.gradient(low,axis=(0,1))
    h,w=a.shape[:2]
    size=min(256,2**int(math.floor(math.log2(max(1,min(h,w))))))
    if size<64:return None,{**info,'status':'abstained_small_frame'}
    attempts=[]
    while size>=64:
        xx=np.arange(size,dtype=float);xx-=xx.mean();norm=np.sum(xx*xx)
        window=np.hanning(size);spectra=[];tiles=[]
        freq=np.fft.rfftfreq(size);search=(freq>=1/96)&(freq<=1/3)
        for y in range(0,h-size+1,max(32,size//2)):
            for x in range(0,w-size+1,max(32,size//2)):
                sl=np.s_[y:y+size,x:x+size];mean=np.mean(low[sl],axis=(0,1))
                gradient=float(np.sqrt(np.mean(gx[sl]**2+gy[sl]**2)))
                if mean.min()<5 or (not include_near_white and mean.max()>249) or gradient>1.8:continue
                z=scalar[sl].astype(float);z-=z.mean(1,keepdims=True);z-=(z@xx)[:,None]*xx[None,:]/norm
                power=np.mean(abs(np.fft.rfft(z*window[None,:],axis=1))**2,axis=0)/np.sum(window**2)
                floor=float(np.median(power[freq>=.35]));power=np.maximum(power-2*floor,0)
                total=float(np.sum(power[search]))
                if total<.01:continue
                spectra.append(power/total);tiles.append({'x':x,'y':y,'energy':total,'gradient':gradient})
        attempts.append({'tile_size':size,'valid_count':len(tiles)})
        if len(spectra)>=6:break
        size//=2
    info['window_attempts']=attempts
    if len(spectra)<6:return None,{**info,'status':'abstained_insufficient_spectrum_tiles','spectral_tile_count':len(spectra)}
    p=np.median(np.array(spectra),axis=0);p[~search]=0
    total=float(p.sum())
    if total<=0:return None,{**info,'status':'abstained_no_spectral_energy'}
    p/=total;cdf=np.cumsum(p)
    def quantile(q):return float(freq[min(np.searchsorted(cdf,q),len(freq)-1)])
    f10,f50,f90=quantile(.10),quantile(.50),quantile(.90)
    lower=max(3.,.8/f90);upper=min(128.,1.2/f10)
    if upper<=lower:return None,{**info,'status':'abstained_degenerate_interval'}
    info.update(status='estimated_not_truth',spectral_tile_count=len(tiles),spectral_tile_size=size,spectral_tiles=tiles,median_wavelength=1/f50,central_energy_wavelengths=[1/f90,1/f10],selected_range=[lower,upper],frequency=freq.tolist(),normalized_spectrum=p.tolist())
    return prior_for(lower,upper),info
