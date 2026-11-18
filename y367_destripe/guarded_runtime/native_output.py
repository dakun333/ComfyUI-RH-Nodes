"""Exact CPU output primitives. No model, thresholds, hue sectors, or decisions changed."""
import numpy as np
from numba import njit

@njit(cache=True,nogil=True,fastmath=False)
def directed_round(target, enabled, fallback):
    h,w,c=target.shape;out=fallback.copy();phase=np.empty(w,np.float32);error=np.empty((w,3),np.float32)
    half=np.float32(.5);zero=np.float32(0)
    for x in range(w):
        p=zero
        for bit in range(8):
            if (x%256 >> bit)&1:p=np.float32(p+np.float32(2.**(-bit-1)))
        phase[x]=np.float32(p-half)
        for k in range(3):error[x,k]=phase[x]
    for y in range(h):
        for x in range(w):
            for k in range(3):
                row=target[y,x,k];value=np.float32(row+error[x,k]);rounded=np.float32(np.floor(np.float32(value+half)))
                low=np.float32(np.floor(row));high=np.float32(np.ceil(row))
                if rounded<low:rounded=low
                if rounded>high:rounded=high
                if enabled[y,x]:
                    out[y,x,k]=np.uint8(rounded);error[x,k]=np.float32(value-rounded)
                else:error[x,k]=phase[x]
    return out

@njit(cache=True,nogil=True,fastmath=False,inline='always')
def _chroma(r,g,b):
    high=max(max(r,g),b);low=min(min(r,g),b);spread=np.float32(high-low)
    amplitude=max(max(abs(r),abs(g)),abs(b))
    return spread>np.float32(1e-5) and spread>=np.float32(np.float32(.25)*amplitude)

@njit(cache=True,nogil=True,fastmath=False,inline='always')
def _admission(r,g,b):
    high=max(max(r,g),b);low=min(min(r,g),b);spread=np.float32(high-low)
    if not _chroma(r,g,b):return False
    return np.float32(np.float32(2)*abs(np.float32(r-b)))<=spread or np.float32(np.float32(2)*abs(np.float32(r-g)))<=spread

@njit(cache=True,nogil=True,fastmath=False)
def guard_candidates(original,target,candidate8):
    h,w,c=original.shape;effective=np.empty_like(target);eight=np.empty_like(candidate8);keep=np.empty((h,w),np.bool_)
    counts=np.zeros(5,np.int64);error_max=np.float32(0)
    for y in range(h):
        for x in range(w):
            r=np.float32(original[y,x,0]-target[y,x,0]);g=np.float32(original[y,x,1]-target[y,x,1]);b=np.float32(original[y,x,2]-target[y,x,2])
            ar=np.float32(original[y,x,0]-np.float32(candidate8[y,x,0]));ag=np.float32(original[y,x,1]-np.float32(candidate8[y,x,1]));ab=np.float32(original[y,x,2]-np.float32(candidate8[y,x,2]))
            dr=np.float32(min(max(np.float32(np.float32(128)+np.float32(np.float32(16)*ar)),np.float32(0)),np.float32(255))-np.float32(128))
            dg=np.float32(min(max(np.float32(np.float32(128)+np.float32(np.float32(16)*ag)),np.float32(0)),np.float32(255))-np.float32(128))
            db=np.float32(min(max(np.float32(np.float32(128)+np.float32(np.float32(16)*ab)),np.float32(0)),np.float32(255))-np.float32(128))
            fk=_admission(r,g,b);ak=_admission(ar,ag,ab);dk=_admission(dr,dg,db);admit=fk and ak and dk
            keep[y,x]=admit;active=ar!=0 or ag!=0 or ab!=0
            if active:
                counts[0]+=1
                if admit:counts[1]+=1
                if not _chroma(r,g,b):counts[2]+=1
                if not ak:counts[3]+=1
                if fk and ak and not dk:counts[4]+=1
            for k in range(3):
                if admit:effective[y,x,k]=target[y,x,k];eight[y,x,k]=candidate8[y,x,k]
                else:effective[y,x,k]=original[y,x,k];eight[y,x,k]=np.uint8(original[y,x,k])
                error_max=max(error_max,abs(np.float32(np.float32(eight[y,x,k])-effective[y,x,k])))
    return effective,eight,keep,counts,error_max

@njit(cache=True,nogil=True,fastmath=False)
def finish_edges(rgb,target,candidate8,protected):
    effective=np.empty_like(target);eight=np.empty_like(candidate8);h,w,c=rgb.shape
    active_count=0;restored=0;error_max=np.float32(0)
    for y in range(h):
        for x in range(w):
            active=False
            for k in range(3):
                if np.float32(candidate8[y,x,k])!=rgb[y,x,k]:active=True
                if protected[y,x]:effective[y,x,k]=rgb[y,x,k];eight[y,x,k]=np.uint8(rgb[y,x,k])
                else:effective[y,x,k]=target[y,x,k];eight[y,x,k]=candidate8[y,x,k]
                error_max=max(error_max,abs(np.float32(np.float32(eight[y,x,k])-effective[y,x,k])))
            if active:
                active_count+=1
                if protected[y,x]:restored+=1
    return effective,eight,active_count,restored,error_max
