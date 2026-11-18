"""CPU4: independent rows in the original strict FP32 kernels. No fastmath.
Parallelism changes execution only; thresholds, coefficients and passes are unchanged.
Two axis workers use up to four native threads each, not full-machine nested teams.
"""
import os
import numpy as np
from numba import njit,prange,set_num_threads,config
_THREAD_COUNT=min(4,config.NUMBA_NUM_THREADS)
def configure_threads(count=None):
    global _THREAD_COUNT
    if count is not None:
        if not isinstance(count,int) or count<1 or count>config.NUMBA_NUM_THREADS:
            raise ValueError('Native CPU thread count outside initialized Numba limit')
        _THREAD_COUNT=count
    set_num_threads(_THREAD_COUNT)
    return _THREAD_COUNT

@njit(cache=True,nogil=True,fastmath=False,parallel=True)
def real_complex_matvec(m,a):
    H,W=a.shape[:2];out=np.empty(a.shape,np.complex64)
    for y in prange(H):
        for x in range(W):
            for i in range(3):
                q=np.complex64(0)
                for j in range(3):
                    q=np.complex64(q+np.complex64(np.complex64(m[y,x,i,j])*a[y,x,j]))
                out[y,x,i]=q
    return out

@njit(cache=True,nogil=True,fastmath=False,parallel=True)
def _median_rank(padded,width,k):
    H,_,C=padded.shape;out=np.empty((H,width,C),np.float32);middle=k//2
    for y in prange(H):
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

@njit(cache=True,nogil=True,fastmath=False,parallel=True)
def pack_complex(real,imag):
    """Assemble exactly real + complex64(1j)*imag, including signed-zero behavior."""
    out=np.empty(real.shape,np.complex64);rr=real.ravel();ii=imag.ravel();q=out.ravel()
    zero=np.float32(0);negative_zero=np.float32(-0.)
    for j in prange(rr.size):
        r=rr[j];i=ii[j]
        if r==zero:
            if np.signbit(r) and np.signbit(i):r=negative_zero
            else:r=zero
        if i==zero:i=zero
        q[j]=np.complex64(complex(r,i))
    return out

@njit(cache=True,nogil=True,fastmath=False,parallel=True)
def compose_added(rgb,base,extra,alpha,limit):
    """Identical compose(rgb, float32(base+extra)) without that native-size temporary."""
    out=np.empty_like(rgb);H,W=alpha.shape
    zero=np.float32(0);one=np.float32(1);full=np.float32(255);eps=np.float32(1e-7);threshold=np.float32(.95);limit=np.float32(limit)
    for y in prange(H):
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

@njit(cache=True,nogil=True,fastmath=False,parallel=True)
def compose_accumulate(extra,rgb,layer,permit,alpha,compose_limit,maximum,minimum):
    H,W=alpha.shape
    zero=np.float32(0);one=np.float32(1);full=np.float32(255)
    eps=np.float32(1e-7);threshold=np.float32(.95)
    limit=np.float32(compose_limit);maximum=np.float32(maximum);minimum=np.float32(minimum)
    row_changed=np.empty(H,np.bool_)
    for y in prange(H):
        any_change=False
        for x in range(W):
            if not any_change:
                old0=extra[y,x,0];old1=extra[y,x,1];old2=extra[y,x,2]
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
            em=zero
            for k in range(3):
                decoded=np.float32(rgb[y,x,k]-np.float32(layer[y,x,k]*lo))
                if decoded<zero:decoded=zero
                if decoded>full:decoded=full
                if alpha[y,x]<threshold:decoded=rgb[y,x,k]
                native=np.float32(np.float32(rgb[y,x,k]-decoded)*permit[y,x])
                extra[y,x,k]=np.float32(extra[y,x,k]+native)
                if abs(extra[y,x,k])>em:em=abs(extra[y,x,k])
            if em<eps:em=eps
            scale=np.float32(maximum/em)
            if scale>one:scale=one
            em=zero
            for k in range(3):
                extra[y,x,k]=np.float32(extra[y,x,k]*scale)
                if abs(extra[y,x,k])>em:em=abs(extra[y,x,k])
            if em<minimum or alpha[y,x]<threshold:
                for k in range(3):extra[y,x,k]=zero
            if not any_change:
                v0=extra[y,x,0];v1=extra[y,x,1];v2=extra[y,x,2]
                same=(v0==old0 and v1==old1 and v2==old2 and np.signbit(v0)==np.signbit(old0) and np.signbit(v1)==np.signbit(old1) and np.signbit(v2)==np.signbit(old2))
                if not same:any_change=True
        row_changed[y]=any_change
    return not np.any(row_changed)

@njit(cache=True,nogil=True,fastmath=False,parallel=True)
def decode_coeff(coeff,amp):
    H,W=amp.shape[:2];out=np.empty(amp.shape,np.complex64)
    unit=np.complex64(1j)
    for y in prange(H):
        for x in range(W):
            for i in range(3):
                q=np.complex64(0)
                for j in range(3):
                    k=i*3+j
                    c=np.complex64(np.complex64(coeff[y,x,k])+np.complex64(unit*coeff[y,x,k+9]))
                    q=np.complex64(q+np.complex64(c*amp[y,x,j]))
                out[y,x,i]=q
    return out

@njit(cache=True,nogil=True,fastmath=False,parallel=True)
def carrier_from_channels(real,imag,phase):
    """Original pack_complex then complex64 multiply, retaining only its real output."""
    H,W,C=real.shape;out=np.empty(real.shape,np.float32)
    zero=np.float32(0);negative_zero=np.float32(-0.)
    for y in prange(H):
        for x in range(W):
            for k in range(C):
                r=real[y,x,k];i=imag[y,x,k]
                if r==zero:
                    if np.signbit(r) and np.signbit(i):r=negative_zero
                    else:r=zero
                if i==zero:i=zero
                cr=np.float32(phase[x].real);ci=np.float32(phase[x].imag)
                # NumPy complex64 carrier multiplication uses this fused rounding.
                out[y,x,k]=_fma32(r,cr,np.float32(-np.float32(i*ci)))
    return out

from numba import types
from numba.extending import intrinsic
@intrinsic
def _fma32(typingctx,a,b,c):
    if a!=types.float32 or b!=types.float32 or c!=types.float32:return None
    sig=types.float32(a,b,c)
    def codegen(context,builder,signature,args):
        from llvmlite import ir
        fn=builder.module.globals.get('llvm.fma.f32')
        if fn is None:fn=ir.Function(builder.module,ir.FunctionType(args[0].type,[args[0].type]*3),name='llvm.fma.f32')
        return builder.call(fn,args)
    return sig,codegen


@njit(cache=True,nogil=True,fastmath=False,parallel=True)
def invert3x3(m):
    H,W=m.shape[:2];out=np.empty(m.shape,np.float32);bad=np.zeros(H,np.bool_)
    for y in prange(H):
        for x in range(W):
            a=np.float64(m[y,x,0,0]);b=np.float64(m[y,x,0,1]);c=np.float64(m[y,x,0,2])
            d=np.float64(m[y,x,1,0]);e=np.float64(m[y,x,1,1]);f=np.float64(m[y,x,1,2])
            g=np.float64(m[y,x,2,0]);h=np.float64(m[y,x,2,1]);i=np.float64(m[y,x,2,2])
            c00=e*i-f*h;c01=c*h-b*i;c02=b*f-c*e
            c10=f*g-d*i;c11=a*i-c*g;c12=c*d-a*f
            c20=d*h-e*g;c21=b*g-a*h;c22=a*e-b*d
            determinant=a*c00+b*c10+c*c20
            if determinant==0. or not np.isfinite(determinant):
                bad[y]=True
                for q in range(3):
                    for r in range(3):out[y,x,q,r]=np.float32(np.nan)
            else:
                out[y,x,0,0]=np.float32(c00/determinant);out[y,x,0,1]=np.float32(c01/determinant);out[y,x,0,2]=np.float32(c02/determinant)
                out[y,x,1,0]=np.float32(c10/determinant);out[y,x,1,1]=np.float32(c11/determinant);out[y,x,1,2]=np.float32(c12/determinant)
                out[y,x,2,0]=np.float32(c20/determinant);out[y,x,2,1]=np.float32(c21/determinant);out[y,x,2,2]=np.float32(c22/determinant)
    return out,np.any(bad)
