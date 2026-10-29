"""Alpha-aware evidence and restoration without baking a matte into RGB."""
from dataclasses import asdict
import math

import cv2
import numpy as np

from .core import (Prior, Parameters, Settings, canonical, shade_tangent,
                   separate_scalar, scalar_filter)


def validate_alpha(alpha, shape):
    a = np.asarray(alpha, dtype=np.float32)
    if a.shape != shape or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError('alpha must be a finite HxW array in 0-1 matching RGB')
    return a


def _weighted(values, weights, operator):
    denominator = operator(weights)
    scale = weights if values.ndim == 2 else weights[:, :, None]
    norm = denominator if values.ndim == 2 else denominator[:, :, None]
    return operator(values * scale) / np.maximum(norm, 1e-6)


def _gaussian(values, weights, sx, sy=None):
    sy = sx if sy is None else sy
    return _weighted(values, weights, lambda z: cv2.GaussianBlur(
        z, (0, 0), sx, sigmaY=sy, borderType=cv2.BORDER_REFLECT))


def _weights(alpha):
    peak = float(alpha.max())
    threshold = max(.1, .9 * peak)
    evidence = (alpha >= threshold).astype(np.float32)
    application = np.where(alpha >= .05, alpha, 0).astype(np.float32)
    return evidence, application


def guided_detail_alpha(src, alpha, params):
    radius = max(2, int(round(.3 * params.period_max)))
    average = lambda z: cv2.boxFilter(z, -1, (2*radius+1, 1), borderType=cv2.BORDER_REFLECT)
    _, weights = _weights(alpha)
    mean = _weighted(src - 128., weights, average)
    variance = np.maximum(_weighted((src-128.)**2, weights, average) - mean*mean, 0)
    edge = np.max(variance, axis=2)
    coefficient = edge / (edge + params.guide_sigma**2)
    base = (_weighted(coefficient, weights, average)[:, :, None]*(src-128.)
            + _weighted((1-coefficient[:, :, None])*mean, weights, average))
    return np.where(weights[:, :, None] > 0, src-128.-base, 0).astype(np.float32)


def estimate_color_alpha(src, prior, settings, alpha):
    evidence_weights, app_weights = _weights(alpha)
    scale = math.sqrt(prior.base.period_min*prior.base.period_max)
    residual = guided_detail_alpha(src, alpha, prior.base)
    f = np.stack([scalar_filter(residual[:, :, c], prior, settings) for c in range(3)], axis=2)
    size = max(40, round(2*scale)); tile_height = 16; h,w = alpha.shape
    low = _gaussian(src, app_weights, .12*scale)
    gy,gx = np.gradient(low, axis=(0,1))
    tangent = shade_tangent(low).astype(np.float32)
    gx = gx - tangent*np.sum(gx*tangent, axis=2, keepdims=True)
    gy = gy - tangent*np.sum(gy*tangent, axis=2, keepdims=True)
    interior = cv2.erode(evidence_weights, np.ones((5,5),np.uint8)) > 0
    tiles = []; rejected = {'alpha_coverage':0,'brightness':0,'gradient':0,'amplitude':0,'rank':0,'shade_alignment':0}
    for y in range(0,h-tile_height+1,tile_height):
        for x in range(0,w-size+1,max(20,size//2)):
            sl = np.s_[y:y+tile_height,x:x+size]; valid = interior[sl]
            coverage = float(valid.mean())
            if coverage < .8:
                rejected['alpha_coverage'] += 1; continue
            mean = low[sl][valid].mean(0).astype(float)
            gradient = float(np.sqrt(np.mean((gx[sl]**2+gy[sl]**2)[valid])))
            if mean.min()<5 or (not settings.include_near_white and mean.max()>249):
                rejected['brightness']+=1;continue
            if gradient>settings.max_gradient:
                rejected['gradient']+=1;continue
            z=f[sl][valid].astype(float);z-=np.median(z,axis=0)
            amplitude=np.linalg.norm(z,axis=1)
            weights=np.minimum(1,2*np.median(amplitude)/np.maximum(amplitude,1e-9))
            cov=(z*weights[:,None]).T@z/max(weights.sum(),1e-9)
            vals,vecs=np.linalg.eigh(cov);direction=canonical(vecs[:,-1])
            rank=float(vals[-1]/max(vals.sum(),1e-12))
            separation=float(1-(direction@shade_tangent(mean))**2)
            if vals[-1]<.025:rejected['amplitude']+=1;continue
            if rank<settings.min_tile_rank:rejected['rank']+=1;continue
            if separation<settings.min_color_separation:rejected['shade_alignment']+=1;continue
            tiles.append(dict(x=x,y=y,rank=rank,separation=separation,direction=direction.tolist(),rms=float(np.sqrt(np.trace(cov)/3)),alpha_coverage=coverage))
    info=dict(tile_size=size,tile_height=tile_height,candidate_tile_count=len(tiles),tiles=tiles,alpha_aware=True,rejections=rejected)
    if len(tiles)<settings.min_support:return None,{**info,'reason':'insufficient_color_tiles'}
    directions=np.array([t['direction'] for t in tiles]);dots=abs(directions@directions.T)
    inlier=dots[np.argmax((dots>=settings.cluster_cosine).sum(1))]>=settings.cluster_cosine
    for _ in range(2):
        _,v=np.linalg.eigh(directions[inlier].T@directions[inlier]);color=canonical(v[:,-1])
        inlier=abs(directions@color)>=settings.cluster_cosine
    count=int(inlier.sum());info.update(support_tile_count=count,support_fraction=count/len(tiles),color_direction=color.tolist(),supporting_tiles=[t for t,k in zip(tiles,inlier) if k])
    if count<settings.min_support:return None,{**info,'reason':'insufficient_consensus'}
    return color.astype(np.float32), info


def estimate_scale_alpha(rgb, alpha, include_near_white=False):
    src=np.asarray(rgb,dtype=np.float32);alpha=validate_alpha(alpha,src.shape[:2])
    settings=Settings(include_near_white=include_near_white);settings.validate()
    probe=Prior(Parameters(period_min=3.,period_max=96.))
    color,evidence=estimate_color_alpha(src,probe,settings,alpha)
    info=dict(method='alpha_aware_shared_color_spectral_scale',probe_range=[3,96],color_evidence=evidence,uses_resolution_ratio=False,include_near_white=include_near_white,alpha_aware=True)
    if color is None:return None,{**info,'status':'abstained_no_color_evidence'}
    ew,aw=_weights(alpha);tangent=shade_tangent(_gaussian(src,aw,.3*math.sqrt(3*96))).astype(np.float32)
    scalar=separate_scalar(guided_detail_alpha(src,alpha,probe.base),tangent,color,.15)
    scalar=_gaussian(scalar,aw,.65,1.)
    low=_gaussian(src,aw,2.);gy,gx=np.gradient(low,axis=(0,1))
    tangent_low=shade_tangent(low).astype(np.float32)
    gx=gx-tangent_low*np.sum(gx*tangent_low,axis=2,keepdims=True)
    gy=gy-tangent_low*np.sum(gy*tangent_low,axis=2,keepdims=True)
    valid=cv2.erode(ew,np.ones((5,5),np.uint8))>0
    h,w=alpha.shape;attempts=[];spectra=[];tiles=[]
    for width in [256,128,64]:
        height=max(16,width//4)
        if width>w or height>h:continue
        xx=np.arange(width,dtype=float);xx-=xx.mean();norm=np.sum(xx*xx);window=np.hanning(width)
        freq=np.fft.rfftfreq(width);search=(freq>=1/96)&(freq<=1/3)
        spectra=[];tiles=[]
        for y in range(0,h-height+1,max(8,height//2)):
            for x in range(0,w-width+1,max(32,width//2)):
                sl=np.s_[y:y+height,x:x+width];v=valid[sl];rows=v.all(axis=1)
                if rows.sum()<max(8,height//2):continue
                mean=low[sl][v].mean(0);gradient=float(np.sqrt(np.mean((gx[sl]**2+gy[sl]**2)[v])))
                if mean.min()<5 or (not include_near_white and mean.max()>249) or gradient>1.8:continue
                z=scalar[sl][rows].astype(float);z-=z.mean(1,keepdims=True);z-=(z@xx)[:,None]*xx[None,:]/norm
                power=np.mean(abs(np.fft.rfft(z*window[None,:],axis=1))**2,axis=0)/np.sum(window**2)
                floor=float(np.median(power[freq>=.35]));power=np.maximum(power-2*floor,0);total=float(power[search].sum())
                if total<.01:continue
                spectra.append(power/total);tiles.append(dict(x=x,y=y,energy=total,gradient=gradient,valid_rows=int(rows.sum())))
        attempts.append(dict(tile_width=width,tile_height=height,valid_count=len(tiles)))
        if len(spectra)>=6:break
    info['window_attempts']=attempts
    if len(spectra)<6:return None,{**info,'status':'abstained_insufficient_spectrum_tiles','spectral_tile_count':len(spectra)}
    p=np.median(np.asarray(spectra),axis=0);p[~search]=0;total=float(p.sum())
    if total<=0:return None,{**info,'status':'abstained_no_spectral_energy'}
    p/=total;cdf=np.cumsum(p)
    q=lambda value:float(freq[min(np.searchsorted(cdf,value),len(freq)-1)])
    f10,f50,f90=q(.1),q(.5),q(.9);lower=max(3.,.8/f90);upper=min(128.,1.2/f10)
    if upper<=lower:return None,{**info,'status':'abstained_degenerate_interval'}
    info.update(status='estimated_not_truth',spectral_tile_count=len(tiles),spectral_tile_size=width,spectral_tile_height=height,spectral_tiles=tiles,median_wavelength=1/f50,central_energy_wavelengths=[1/f90,1/f10],selected_range=[lower,upper])
    return Prior(Parameters(period_min=lower,period_max=upper)),info


def restore_alpha(rgb, alpha, prior=Prior(), settings=Settings(), across_sigma=.65):
    prior.validate();settings.validate();src=np.asarray(rgb,dtype=np.float32)
    alpha=validate_alpha(alpha,src.shape[:2]);color,evidence=estimate_color_alpha(src,prior,settings,alpha)
    info=dict(method='alpha_aware_shared_color_broadband',settings=asdict(settings),evidence=evidence,alpha_aware=True,alpha_modified=False,hidden_rgb_excluded=True)
    if color is None:return src.copy(),{**info,'status':'abstained'},dict(field=np.zeros_like(src))
    scale=math.sqrt(prior.base.period_min*prior.base.period_max);_,weights=_weights(alpha)
    tangent=shade_tangent(_gaussian(src,weights,.3*scale)).astype(np.float32);output=src.copy()
    for gain in settings.gains:
        scalar=separate_scalar(guided_detail_alpha(output,alpha,prior.base),tangent,color,settings.denominator_floor)
        scalar=_gaussian(scalar,weights,across_sigma,settings.along_ratio*scale)
        output-=np.where(weights[:,:,None]>0,gain*scalar[:,:,None]*color,0)
    if not np.isfinite(output).all():raise FloatingPointError('Invalid transparent output')
    field=src-output;info.update(status='processed_not_quality_approved',correction_rms=float(np.sqrt(np.mean(field**2))))
    return output,info,dict(field=field,color=color)
