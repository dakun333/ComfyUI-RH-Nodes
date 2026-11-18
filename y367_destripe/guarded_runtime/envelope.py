"""Phase-equivariant, input-only vertical-stripe envelope operator.
Offline training lives in train_envelope.py. This module has no data/reference/
M5 imports. NumPy CNN inference avoids PyTorch/CUDA cold-start dependencies.
"""
import os
from . import cpu_runtime as cpu4
import json,time
from functools import lru_cache
from pathlib import Path
import numpy as np
import cv2
from scipy.fft import fft2,ifft2,fftfreq
from scipy.ndimage import maximum_filter1d
try:
    from . import native_kernels as _native
except ImportError:
    _native=None
try:
    from . import native_hot as _hot
except ImportError:
    _hot=None
CONFIG=dict(cycles=48.,across=.30,along=.30,analysis=640,grid=128,edge=30.,gate=.5,limit=8.)
try:
    from . import native_analysis as _analysis_native
except ImportError:
    _analysis_native=None

def matvec(m,a):
    if _analysis_native is not None and m.dtype==np.float32 and a.dtype==np.complex64 and m.ndim==4 and a.ndim==3 and m.shape[:2]==a.shape[:2] and m.shape[2:]==(3,3) and a.shape[2]==3:
        return _hot.real_complex_matvec(m,a) if cpu4.HOT_KERNELS and _hot is not None else _analysis_native.real_complex_matvec(m,a)
    return np.einsum('...ij,...j->...i',m,a)


def carrier_real_inplace(amp,phase):
    # resize_complex produced an owned disposable buffer. Keep the original
    # NumPy complex multiply (including its rounding), but reuse that buffer.
    np.multiply(amp,phase[None,:,None],out=amp)
    return amp.real

def resize_complex(z,shape):
    w,h=shape
    real=cv2.resize(z.real,(w,h),interpolation=cv2.INTER_LINEAR)
    imag=cv2.resize(z.imag,(w,h),interpolation=cv2.INTER_LINEAR)
    if _native is not None and real.dtype==imag.dtype==np.float32:
        return _hot.pack_complex(real,imag) if cpu4.HOT_KERNELS and _hot is not None else _native.pack_complex(real,imag)
    return (real+1j*imag).astype(np.complex64)

@lru_cache(maxsize=8)
def _analysis_plan(w,h,original_width,cycles,across,along):
    p=w/cycles;pad=max(8,round(p*3));ph=h+2*pad;pw=w+2*pad
    fy=fftfreq(ph).astype(np.float32)[:,None];fx=fftfreq(pw).astype(np.float32)[None,:];f=cycles/w
    filt=np.exp(-.5*((fx-f)/(f*across))**2-.5*(fy/(f*along))**2)*(fx>0)
    coord=(np.arange(w,dtype=np.float64)+.5)/w-.5/original_width
    carrier=np.exp(-2j*np.pi*cycles*coord).astype(np.complex64)
    filt.setflags(write=False);carrier.setflags(write=False)
    return pad,filt,carrier

def _analytic_band(z,F,pad,h,w):
    if cpu4.FFT_LAYOUT and z.dtype==np.float32 and z.ndim==3 and z.shape[2]==3:
        planar=np.ascontiguousarray(z.transpose(2,0,1))
        spectral=fft2(planar,axes=(-2,-1),workers=1);spectral*=F[None]
        inverse=ifft2(spectral,axes=(-2,-1),workers=1,overwrite_x=True)
        return np.ascontiguousarray((2*inverse[:,pad:pad+h,pad:pad+w]).transpose(1,2,0))
    return 2*ifft2(fft2(z,axes=(0,1),workers=1)*F[...,None],axes=(0,1),workers=1)[pad:pad+h,pad:pad+w]

def analysis_amp(image,original_width,grid,configuration,*,full_output=False):
    c=configuration;h,w=image.shape[:2];p=w/c['cycles'];low=cv2.GaussianBlur(image,(0,0),p*.5)
    pad,F,carrier=_analysis_plan(w,h,original_width,c['cycles'],c['across'],c['along'])
    z=np.pad(image-low,((pad,pad),(pad,pad),(0,0)),mode='reflect')
    analytic=_analytic_band(z,F,pad,h,w)
    full=(analytic*carrier[None,:,None]/(1-np.exp(-.5*np.pi*np.pi))).astype(np.complex64)
    down=resize_complex(full,grid)
    return (down,full) if full_output else down

def extract(rgb,alpha,config=None,*,amp_only=False,cache_image=False):
    c=dict(CONFIG);c.update(config or {});H,W=rgb.shape[:2];sc=min(1.,c['analysis']/max(H,W));w=max(32,round(W*sc));h=max(32,round(H*sc))
    im=cv2.resize(np.asarray(rgb,np.float32),(w,h),interpolation=cv2.INTER_AREA)
    p=w/c['cycles'];g=cv2.GaussianBlur(im,(0,0),p*.50)
    pad,F,carrier=_analysis_plan(w,h,W,c['cycles'],c['across'],c['along'])
    z=np.pad(im-g,((pad,pad),(pad,pad),(0,0)),mode='reflect')
    analytic=_analytic_band(z,F,pad,h,w)
    amp=analytic*carrier[None,:,None]/(1-np.exp(-.5*np.pi*np.pi))
    fine_amp=amp.astype(np.complex64,copy=False)
    gs=min(1.,c['grid']/max(h,w));gw=max(16,round(w*gs));gh=max(16,round(h*gs));amp=resize_complex(amp,(gw,gh))
    result=dict(amp=amp,fine_amp=fine_amp,analysis_shape=[w,h],grid_shape=[gw,gh],configuration=c)
    if cache_image and cpu4.ANALYSIS_REUSE:result.update(_analysis_image=im,_analysis_source=rgb)
    if amp_only:return result # Refresh consumes original geom/gate, not these discarded fresh arrays.
    al=cv2.resize(np.asarray(alpha,np.float32),(w,h),interpolation=cv2.INTER_AREA)
    gx=np.gradient(g,axis=1)*p;gy=np.gradient(g,axis=0)*p
    geom=np.concatenate([g/255-.5,np.clip(gx/30,-3,3),np.clip(gy/30,-3,3)],2)
    geom=np.concatenate([cv2.resize(geom[...,k:k+3],(gw,gh),interpolation=cv2.INTER_AREA) for k in (0,3,6)],2)
    edge=np.sqrt(np.mean(gx*gx+gy*gy,axis=2))
    if c.get('edge_protection',True):
        if c.get('edge_radius',0)>0:edge=maximum_filter1d(edge,2*round(p*c['edge_radius'])+1,axis=1,mode='reflect')
        gate=((al>=.995).astype(np.float32)/(1+(edge/c['edge'])**4))**c['gate']
    else:
        # Remove only BR10's geometric/strong-edge halo. Preserve opacity protection.
        gate=(al>=.995).astype(np.float32)**c['gate']
    gate=cv2.GaussianBlur(gate,(0,0),max(.5,p*.06))
    result.update(geom=geom.astype(np.float32),gate=gate.astype(np.float32))
    return result

def features(amp,geom):
    # All amplitude features are invariant to a COMMON arbitrary phase rotation.
    a=np.abs(amp)/2;terms=[geom,np.clip(a,0,5)]
    cross=[]
    for i,j in ((0,1),(0,2),(1,2)):
        z=amp[...,i]*np.conj(amp[...,j])/4
        cross.extend([np.clip(z.real,-5,5),np.clip(z.imag,-5,5)])
    terms.append(np.stack(cross,2))
    return np.concatenate(terms,2).astype(np.float32)

def decode_coeff(coeff,amp):
    if cpu4.HOT_KERNELS and _hot is not None and coeff.dtype==np.float32 and amp.dtype==np.complex64 and coeff.shape==(*amp.shape[:2],18) and amp.ndim==3 and amp.shape[2]==3:
        return _hot.decode_coeff(coeff,amp)
    a=coeff[...,:9].reshape(*coeff.shape[:2],3,3);b=coeff[...,9:].reshape(*coeff.shape[:2],3,3)
    return np.einsum('...ij,...j->...i',a+1j*b,amp,optimize=True).astype(np.complex64)

def _lab(rgb):
    a=np.clip(rgb/255,0,1);a=np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
    lms=a@np.array([[.4122214708,.5363325363,.0514459929],[.2119034982,.6806995451,.1073969566],[.0883024619,.2817188376,.6299787005]],np.float32).T
    return np.cbrt(lms)@np.array([[.2104542553,.7936177850,-.0040720468],[1.9779984951,-2.4285922050,.4505937099],[.0259040371,.7827717662,-.8086757660]],np.float32).T

def perceptual_transform(geom):
    guide=np.clip((geom[...,:3]+.5)*255,0,255);J=np.empty((*guide.shape[:2],3,3),np.float32)
    for k in range(3):
        lo=guide.copy();hi=guide.copy();lo[...,k]=np.maximum(0,lo[...,k]-.5);hi[...,k]=np.minimum(255,hi[...,k]+.5)
        J[...,k]=(_lab(hi)-_lab(lo))/(hi[...,k]-lo[...,k])[...,None]
    if cpu4.INVERSE_BATCH and _hot is not None and J.dtype==np.float32:
        inverse,bad=_hot.invert3x3(J)
        if not bad:return J,inverse
    return J,np.linalg.inv(J)

def perceptual_passivity(amp,pred,geom,*,transform=None,transformed_amp=None):
    # Original passive-response ball and arithmetic, only the fixed transform is reused.
    J,inv=perceptual_transform(geom) if transform is None else transform
    a=transformed_amp if cpu4.TRANSFORM_REUSE and transformed_amp is not None else matvec(J,amp);r=matvec(J,pred);clean=a-r
    result=clean.copy();fractions=[]
    for sl in (slice(0,1),slice(1,3)):
        center=a[...,sl]/2;delta=clean[...,sl]-center
        radius=np.sqrt(np.sum(np.abs(a[...,sl])**2,axis=2))/2
        length=np.sqrt(np.sum(np.abs(delta)**2,axis=2))
        scale=np.minimum(1,radius/np.maximum(length,1e-12))
        result[...,sl]=center+delta*scale[...,None];fractions.append(float(np.mean(scale<.99999)))
    fixed=matvec(inv,a-result).astype(np.complex64)
    return fixed,dict(L_projected_fraction=fractions[0],C_projected_fraction=fractions[1])

def _conv(x,w,b,d=1):
    # Exported torch kernels: O,I,KH,KW. No im2col tensor proportional to RF area.
    kh,kw=w.shape[2:];pad=d*(kh//2);z=np.pad(x,((pad,pad),(pad,pad),(0,0)),mode='constant');H,W=x.shape[:2]
    out=np.zeros((H*W,w.shape[0]),np.float32)
    for iy in range(kh):
        for ix in range(kw):
            a=np.ascontiguousarray(z[iy*d:iy*d+H,ix*d:ix*d+W]).reshape(H*W,-1)
            out+=a@w[:,:,iy,ix].T
    out+=b
    return out.reshape(H,W,-1)
def _silu(x):return x/(1+np.exp(np.clip(-x,-40,40)))

def infer_numpy(feat,weights):
    x=_silu(_conv(feat,weights['stem.weight'],weights['stem.bias']))
    for i,d in enumerate((1,2,4,8)):
        z=_silu(_conv(x,weights[f'blocks.{i}.c1.weight'],weights[f'blocks.{i}.c1.bias'],d))
        x=x+.1*_conv(z,weights[f'blocks.{i}.c2.weight'],weights[f'blocks.{i}.c2.bias'],d)
    y=_conv(x,weights['head.weight'],weights['head.bias'])
    return 1.5*np.tanh(y)

def compose(rgb,layer,alpha,limit=8.):
    if _native is not None and rgb.dtype==layer.dtype==alpha.dtype==np.float32:return _native.compose(rgb,layer,alpha,np.float32(limit))
    lim=np.ones_like(layer)
    np.divide(rgb,layer,out=lim,where=layer>1e-7);np.divide(255-rgb,-layer,out=lim,where=layer<-1e-7)
    small=np.minimum(np.minimum(lim[...,0],lim[...,1]),lim[...,2]);mag=np.maximum(np.maximum(abs(layer[...,0]),abs(layer[...,1])),abs(layer[...,2]))
    gain=np.minimum(1,np.clip(small,0,1));gain=np.minimum(gain,limit/np.maximum(mag,1e-7))
    out=np.clip(rgb-layer*gain[...,None],0,255).astype(np.float32);out[alpha<.95]=rgb[alpha<.95]
    return out

@lru_cache(maxsize=8)
def _native_phase(W,cycles):
    phase=np.exp(2j*np.pi*cycles*np.arange(W,dtype=np.float64)/W).astype(np.complex64);phase.setflags(write=False);return phase

def reconstruct_layer(amp,shape,phase):
    if cpu4.CARRIER_FUSION and _hot is not None and amp.dtype==np.complex64 and amp.ndim==3 and amp.shape[2]==3 and phase.dtype==np.complex64:
        real=cv2.resize(amp.real,shape,interpolation=cv2.INTER_LINEAR)
        imag=cv2.resize(amp.imag,shape,interpolation=cv2.INTER_LINEAR)
        return _hot.carrier_from_channels(real,imag,phase)
    return carrier_real_inplace(resize_complex(amp,shape),phase)

def reconstruct(rgb,alpha,amp,f):
    H,W=rgb.shape[:2];phase=_native_phase(W,f['configuration']['cycles'])
    layer=reconstruct_layer(amp,(W,H),phase)
    if '_native_gate' not in f:f['_native_gate']=cv2.resize(f['gate'],(W,H),interpolation=cv2.INTER_LINEAR)
    gate=f['_native_gate'];layer*=gate[...,None];layer[alpha<.95]=0
    return compose(rgb,layer,alpha,f['configuration']['limit'])

def apply(rgb,alpha,weights_path,config=None):
    start=time.perf_counter();f=extract(rgb,alpha,config);extract_s=time.perf_counter()-start
    with np.load(weights_path) as n:weights={k:n[k] for k in n.files if k!='metadata'};meta=json.loads(str(n['metadata']))
    t=time.perf_counter();feat=features(f['amp'],f['geom']);co=infer_numpy(feat,weights);pred=decode_coeff(co,f['amp']);projection=None
    if f['configuration'].get('passivity',False):pred,projection=perceptual_passivity(f['amp'],pred,f['geom'])
    cnn_s=time.perf_counter()-t
    t=time.perf_counter();out=reconstruct(np.asarray(rgb,np.float32),alpha,pred,f)
    return out,dict(seconds=time.perf_counter()-start,extract_seconds=extract_s,cnn_seconds=cnn_s,reconstruct_seconds=time.perf_counter()-t,configuration=f['configuration'],projection=projection,analysis_shape=f['analysis_shape'],grid_shape=f['grid_shape'],model=meta,max_change=float(np.max(np.abs(out-rgb))))
