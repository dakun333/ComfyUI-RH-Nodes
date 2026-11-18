"""Same float32 arithmetic and exact horizontal rank median, compiled without fastmath.
No learned parameter, new detection rule, or residual early stop lives here.
Numba is a CPU implementation dependency, not another image model.
"""
import numpy as np
from numba import njit
from . import cpu_runtime as cpu4
try:
    from . import native_hot as _hot
except ImportError:
    _hot=None

@njit(cache=True,nogil=True,fastmath=False)
def compose(rgb,layer,alpha,limit):
    out=np.empty_like(rgb);H,W=alpha.shape
    zero=np.float32(0);one=np.float32(1);full=np.float32(255);eps=np.float32(1e-7);threshold=np.float32(.95);limit=np.float32(limit)
    for y in range(H):
        for x in range(W):
            lo=one;mag=zero
            for k in range(3):
                v=layer[y,x,k];b=rgb[y,x,k];q=one
                if v>eps:q=np.float32(b/v)
                if v<-eps:q=np.float32(np.float32(full-b)/np.float32(-v))
                if q<lo:lo=q
                if abs(v)>mag:mag=abs(v)
            if lo<zero:lo=zero
            if lo>one:lo=one
            if mag<eps:mag=eps
            z=np.float32(limit/mag)
            if z<lo:lo=z
            for k in range(3):
                v=np.float32(rgb[y,x,k]-np.float32(layer[y,x,k]*lo))
                if v<zero:v=zero
                if v>full:v=full
                if alpha[y,x]<threshold:v=rgb[y,x,k]
                out[y,x,k]=v
    return out

@njit(cache=True,nogil=True,fastmath=False)
def _median_rank(padded,width,k):
    H,_,C=padded.shape;out=np.empty((H,width,C),np.float32);middle=k//2
    for y in range(H):
        for c in range(C):
            values=np.empty(k,np.float32)
            for i in range(k):
                v=padded[y,i,c];j=i
                while j>0 and values[j-1]>v:
                    values[j]=values[j-1];j-=1
                values[j]=v
            out[y,0,c]=values[middle]
            for x in range(1,width):
                gone=padded[y,x-1,c];v=padded[y,x+k-1,c];i=0
                while i<k-1 and values[i]!=gone:i+=1
                for j in range(i,k-1):values[j]=values[j+1]
                j=k-1
                while j>0 and values[j-1]>v:
                    values[j]=values[j-1];j-=1
                values[j]=v;out[y,x,c]=values[middle]
    return out

def horizontal_median(image,k):
    # SciPy reflect == half-sample symmetric; all production kernels are odd, horizontal.
    if k<1 or k%2!=1 or image.dtype!=np.float32 or image.ndim not in (2,3):raise ValueError('Unsupported exact median shape/type')
    a=image[...,None] if image.ndim==2 else image;pad=k//2
    padded=np.ascontiguousarray(np.pad(a,((0,0),(pad,pad),(0,0)),mode='symmetric'))
    out=_hot._median_rank(padded,a.shape[1],k) if cpu4.HOT_KERNELS and _hot is not None else _median_rank(padded,a.shape[1],k)
    return out[...,0] if image.ndim==2 else out

@njit(cache=True,nogil=True,fastmath=False)
def add_and_cap_extra(extra,native,alpha,maximum,minimum):
    H,W=alpha.shape;one=np.float32(1);eps=np.float32(1e-7);maximum=np.float32(maximum);minimum=np.float32(minimum);threshold=np.float32(.95)
    for y in range(H):
        for x in range(W):
            mag=np.float32(0)
            for k in range(3):
                extra[y,x,k]=np.float32(extra[y,x,k]+native[y,x,k])
                if abs(extra[y,x,k])>mag:mag=abs(extra[y,x,k])
            if mag<eps:mag=eps
            scale=np.float32(maximum/mag)
            if scale>one:scale=one
            mag=np.float32(0)
            for k in range(3):
                extra[y,x,k]=np.float32(extra[y,x,k]*scale)
                if abs(extra[y,x,k])>mag:mag=abs(extra[y,x,k])
            if mag<minimum or alpha[y,x]<threshold:
                for k in range(3):extra[y,x,k]=np.float32(0)
    return extra

@njit(cache=True,nogil=True,fastmath=False)
def pack_complex(real,imag):
    """Assemble exactly real + complex64(1j)*imag, including signed-zero behavior."""
    out=np.empty(real.shape,np.complex64);rr=real.ravel();ii=imag.ravel();q=out.ravel()
    zero=np.float32(0);negative_zero=np.float32(-0.)
    for j in range(rr.size):
        r=rr[j];i=ii[j]
        if r==zero:
            if np.signbit(r) and np.signbit(i):r=negative_zero
            else:r=zero
        if i==zero:i=zero
        q[j]=np.complex64(complex(r,i))
    return out

@njit(cache=True,nogil=True,fastmath=False)
def compose_added(rgb,base,extra,alpha,limit):
    """Identical compose(rgb, float32(base+extra)) without that native-size temporary."""
    out=np.empty_like(rgb);H,W=alpha.shape
    zero=np.float32(0);one=np.float32(1);full=np.float32(255);eps=np.float32(1e-7);threshold=np.float32(.95);limit=np.float32(limit)
    for y in range(H):
        for x in range(W):
            lo=one;mag=zero
            for k in range(3):
                v=np.float32(base[y,x,k]+extra[y,x,k]);b=rgb[y,x,k];q=one
                if v>eps:q=np.float32(b/v)
                if v<-eps:q=np.float32(np.float32(full-b)/np.float32(-v))
                if q<lo:lo=q
                if abs(v)>mag:mag=abs(v)
            if lo<zero:lo=zero
            if lo>one:lo=one
            if mag<eps:mag=eps
            z=np.float32(limit/mag)
            if z<lo:lo=z
            for k in range(3):
                layer=np.float32(base[y,x,k]+extra[y,x,k])
                v=np.float32(rgb[y,x,k]-np.float32(layer*lo))
                if v<zero:v=zero
                if v>full:v=full
                if alpha[y,x]<threshold:v=rgb[y,x,k]
                out[y,x,k]=v
    return out

@njit(cache=True,nogil=True,fastmath=False)
def add_difference_and_cap_extra(extra,rgb,decoded,permit,alpha,maximum,minimum):
    """Original float32 (rgb-decoded)*permit, addition and cap; avoid native temporary."""
    H,W=alpha.shape;one=np.float32(1);eps=np.float32(1e-7);maximum=np.float32(maximum);minimum=np.float32(minimum);threshold=np.float32(.95)
    for y in range(H):
        for x in range(W):
            mag=np.float32(0)
            for k in range(3):
                native=np.float32(np.float32(rgb[y,x,k]-decoded[y,x,k])*permit[y,x])
                extra[y,x,k]=np.float32(extra[y,x,k]+native)
                if abs(extra[y,x,k])>mag:mag=abs(extra[y,x,k])
            if mag<eps:mag=eps
            scale=np.float32(maximum/mag)
            if scale>one:scale=one
            mag=np.float32(0)
            for k in range(3):
                extra[y,x,k]=np.float32(extra[y,x,k]*scale)
                if abs(extra[y,x,k])>mag:mag=abs(extra[y,x,k])
            if mag<minimum or alpha[y,x]<threshold:
                for k in range(3):extra[y,x,k]=np.float32(0)
    return extra

@njit(cache=True,nogil=True,fastmath=False)
def finish_total(rgb,alpha,baseline,vertical,horizontal,two_axes,maximum,limit):
    """Same ordered axis sum, two additive caps and compose; no full-size temporaries."""
    out=np.empty_like(baseline);H,W=alpha.shape;values=np.empty(3,np.float32);layer=np.empty(3,np.float32);composed=np.empty(3,np.float32)
    zero=np.float32(0);one=np.float32(1);full=np.float32(255);eps=np.float32(1e-7);threshold=np.float32(.95);maximum=np.float32(maximum);limit=np.float32(limit)
    total_max=zero;additional_max=zero
    for y in range(H):
        for x in range(W):
            mag=zero
            for k in range(3):
                v=np.float32(zero+vertical[y,x,k])
                if two_axes:v=np.float32(v+horizontal[y,x,k])
                values[k]=v;mag=max(mag,abs(v))
            scale=np.float32(maximum/max(mag,eps));scale=min(one,scale);active=False
            for k in range(3):
                values[k]=np.float32(values[k]*scale)
                if values[k]!=zero:active=True
                layer[k]=np.float32(np.float32(rgb[y,x,k]-baseline[y,x,k])+values[k])
            lo=one;mag=zero
            for k in range(3):
                v=layer[k];b=rgb[y,x,k];q=one
                if v>eps:q=np.float32(b/v)
                if v<-eps:q=np.float32(np.float32(full-b)/np.float32(-v))
                if q<lo:lo=q
                if abs(v)>mag:mag=abs(v)
            lo=min(max(lo,zero),one);mag=max(mag,eps);z=np.float32(limit/mag)
            if z<lo:lo=z
            delta_mag=zero
            for k in range(3):
                v=np.float32(rgb[y,x,k]-np.float32(layer[k]*lo));v=min(max(v,zero),full)
                if alpha[y,x]<threshold:v=rgb[y,x,k]
                composed[k]=v if active else baseline[y,x,k]
                values[k]=np.float32(composed[k]-baseline[y,x,k]);delta_mag=max(delta_mag,abs(values[k]))
            delta_scale=min(one,np.float32(maximum/max(delta_mag,eps)))
            for k in range(3):
                v=np.float32(baseline[y,x,k]+np.float32(values[k]*delta_scale))
                if not active:v=baseline[y,x,k]
                if alpha[y,x]<threshold:v=rgb[y,x,k]
                out[y,x,k]=v;total_max=max(total_max,abs(np.float32(v-rgb[y,x,k])));additional_max=max(additional_max,abs(np.float32(v-baseline[y,x,k])))
    return out,total_max,additional_max
