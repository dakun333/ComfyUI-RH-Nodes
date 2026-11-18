"""Compose then accumulate in the historical FLOAT32 order; exact-state flag.
No approximate epsilon stop, no changed permit, no new model or protection.
"""
import numpy as np
from numba import njit

@njit(cache=True,nogil=True,fastmath=False)
def compose_accumulate(extra,rgb,layer,permit,alpha,compose_limit,maximum,minimum):
    H,W=alpha.shape
    zero=np.float32(0);one=np.float32(1);full=np.float32(255)
    eps=np.float32(1e-7);threshold=np.float32(.95)
    limit=np.float32(compose_limit);maximum=np.float32(maximum);minimum=np.float32(minimum)
    any_change=False
    for y in range(H):
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
    return not any_change
