"""CPU-only exact palette/whole-vector/paired directed8 dataflow."""
import numpy as np
from numba import njit

@njit(cache=True,nogil=True,fastmath=False)
def restrict_pixels(rgb,target):
    H,W=rgb.shape[:2];removed=np.empty_like(target);result=np.empty_like(target);allowed=np.empty((H,W),np.bool_);active=np.empty((H,W),np.bool_)
    eps=np.float32(1e-12);small=np.float32(1e-5);two=np.float32(2);four=np.float32(4);sixty=np.float32(60);full=np.float32(360)
    for y in range(H):
        for x in range(W):
            r=np.float32(rgb[y,x,0]-target[y,x,0]);g=np.float32(rgb[y,x,1]-target[y,x,1]);b=np.float32(rgb[y,x,2]-target[y,x,2])
            removed[y,x,0]=r;removed[y,x,1]=g;removed[y,x,2]=b
            hi=max(max(r,g),b);lo=min(min(r,g),b);span=np.float32(hi-lo);den=max(span,eps)
            if hi==r:hue=np.float32(np.float32(g-b)/den)
            elif hi==g:hue=np.float32(np.float32(np.float32(b-r)/den)+two)
            else:hue=np.float32(np.float32(np.float32(r-g)/den)+four)
            hue=np.float32(np.mod(np.float32(hue*sixty),full))
            keep=span>small and (np.float32(30)<=hue<=np.float32(150) or np.float32(210)<=hue<=np.float32(330))
            allowed[y,x]=keep;active[y,x]=r!=0 or g!=0 or b!=0
            for k in range(3):result[y,x,k]=target[y,x,k] if keep else rgb[y,x,k]
    return removed,result,allowed,active

@njit(cache=True,nogil=True,fastmath=False,inline='always')
def _clip(v):return np.float32(min(max(v,np.float32(0)),np.float32(255)))

@njit(cache=True,nogil=True,fastmath=False,inline='always')
def _rounded(row,error,enabled,phase):
    if not enabled:return np.uint8(np.rint(row)),phase
    value=np.float32(row+error);v=np.float32(np.floor(np.float32(value+np.float32(.5))));lo=np.float32(np.floor(row));hi=np.float32(np.ceil(row))
    if v<lo:v=lo
    if v>hi:v=hi
    return np.uint8(v),np.float32(value-v)

@njit(cache=True,nogil=True,fastmath=False)
def encode_pair(original,baseline,target,prior8):
    H,W=target.shape[:2];out=np.empty((H,W,3),np.uint8);phase=np.empty(W,np.float32);base_error=np.empty((W,3),np.float32);target_error=np.empty((W,3),np.float32)
    zero=np.float32(0);small=np.float32(1e-5);maximum_error=zero
    for x in range(W):
        p=zero
        for bit in range(8):
            if (x%256 >> bit)&1:p=np.float32(p+np.float32(2.**(-bit-1)))
        phase[x]=np.float32(p-np.float32(.5))
        for k in range(3):base_error[x,k]=phase[x];target_error[x,k]=phase[x]
    for y in range(H):
        for x in range(W):
            eb=zero;et=zero;same=True;base_same=True
            for k in range(3):
                eb=max(eb,abs(np.float32(original[y,x,k]-_clip(baseline[y,x,k]))));et=max(et,abs(np.float32(original[y,x,k]-_clip(target[y,x,k]))))
                same=same and target[y,x,k]==baseline[y,x,k];base_same=base_same and baseline[y,x,k]==original[y,x,k]
            for k in range(3):
                q,target_error[x,k]=_rounded(_clip(target[y,x,k]),target_error[x,k],et>small,phase[x])
                if prior8 is None:
                    prior,base_error[x,k]=_rounded(_clip(baseline[y,x,k]),base_error[x,k],eb>small,phase[x])
                    if base_same:prior=np.uint8(_clip(np.float32(np.rint(original[y,x,k]))))
                else:prior=prior8[y,x,k]
                if same:q=prior
                out[y,x,k]=q;maximum_error=max(maximum_error,abs(np.float32(np.float32(q)-target[y,x,k])))
    return out,maximum_error
