"""Local single-sided amplitude transition. No new restoration/detector pass."""
import numpy as np
from numba import njit,prange

@njit(cache=True,nogil=True,fastmath=False)
def boundary_seeds(rgb,raw,target,forced,seed_policy):
    h,w,_=rgb.shape;seeds=np.zeros((h,w),np.uint8);severity=np.zeros((h,w),np.float32)
    for y in range(h):
        for x in range(w):
            if not forced[y,x] or not seed_policy[y,x]:continue
            # A natural carrier zero is not a policy-created discontinuity.
            raw_mag=max(abs(raw[y,x,0]),abs(raw[y,x,1]),abs(raw[y,x,2]))
            if raw_mag<np.float32(.5):continue
            weight=np.float32(0)
            for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                yy=y+dy;xx=x+dx
                if yy<0 or yy>=h or xx<0 or xx>=w or forced[yy,xx]:continue
                mag=np.float32(0);raw_jump=np.float32(0);source_jump=np.float32(0)
                for c in range(3):
                    mag=max(mag,abs(np.float32(rgb[yy,xx,c]-target[yy,xx,c])))
                    raw_jump=max(raw_jump,abs(np.float32(raw[yy,xx,c]-raw[y,x,c])))
                    source_jump=max(source_jump,abs(np.float32(rgb[yy,xx,c]-rgb[y,x,c])))
                if mag<=np.float32(.5):continue
                if source_jump>np.float32(8):continue
                if raw_jump>max(np.float32(.15),np.float32(.35)*max(raw_mag,mag)):continue
                t=min(np.float32(1),max(np.float32(0),np.float32((mag-np.float32(.5))/np.float32(.5))))
                u=np.float32(t*t*np.float32(np.float32(3)-np.float32(2)*t))
                weight=max(weight,u)
            if weight>0:seeds[y,x]=1;severity[y,x]=weight
    return seeds,severity

@njit(cache=True,nogil=True,fastmath=False)
def seed_lookup(rgb,seeds,labels,severity,n):
    colors=np.zeros((n,3),np.float32);weights=np.zeros(n,np.float32)
    for y in range(seeds.shape[0]):
        for x in range(seeds.shape[1]):
            if seeds[y,x]:
                i=labels[y,x];weights[i]=severity[y,x]
                for k in range(3):colors[i,k]=rgb[y,x,k]
    return colors,weights

@njit(cache=True,nogil=True,fastmath=False,parallel=True)
def gain_and_target(rgb,target,forced,distance,labels,colors,weights,width):
    h,w,_=rgb.shape;gain=np.ones((h,w),np.float32);scaled=target.copy();affected=np.zeros((h,w),np.bool_)
    width=np.float32(width)
    for y in prange(h):
        for x in range(w):
            if forced[y,x] or distance[y,x]>=width+np.float32(.5):continue
            mag=max(abs(rgb[y,x,0]-target[y,x,0]),abs(rgb[y,x,1]-target[y,x,1]),abs(rgb[y,x,2]-target[y,x,2]))
            if mag==0:continue
            t=max(np.float32(0),min(np.float32(1),np.float32((distance[y,x]-np.float32(.5))/width)))
            ramp=np.float32(t*t*np.float32(np.float32(3)-np.float32(2)*t))
            i=labels[y,x];delta=np.float32(0)
            for k in range(3):delta=max(delta,abs(np.float32(rgb[y,x,k]-colors[i,k])))
            # Fade influence to exactly zero before a materially different source surface.
            c=max(np.float32(0),min(np.float32(1),np.float32(1)-np.float32(delta/np.float32(16))))
            material=np.float32(c*c*np.float32(np.float32(3)-np.float32(2)*c))
            g=np.float32(np.float32(1)-np.float32(weights[i]*material*np.float32(np.float32(1)-ramp)))
            if g>=np.float32(1):continue
            gain[y,x]=g;affected[y,x]=True
            for k in range(3):scaled[y,x,k]=np.float32(rgb[y,x,k]-np.float32(g*np.float32(rgb[y,x,k]-target[y,x,k])))
    return gain,scaled,affected
@njit(cache=True,nogil=True,fastmath=False)
def finish_retreat(rgb,old_target,old8,target,eight,affected):
    h,w,_=rgb.shape;cancelled=0;changed=0;error=np.float32(0)
    for y in range(h):
        for x in range(w):
            if not affected[y,x]:
                for k in range(3):target[y,x,k]=old_target[y,x,k];eight[y,x,k]=old8[y,x,k]
            else:
                inside=True
                for k in range(3):
                    low=min(rgb[y,x,k],np.float32(old8[y,x,k]));high=max(rgb[y,x,k],np.float32(old8[y,x,k]))
                    if np.float32(eight[y,x,k])<low or np.float32(eight[y,x,k])>high:inside=False
                if not inside:
                    cancelled+=1
                    for k in range(3):target[y,x,k]=rgb[y,x,k];eight[y,x,k]=np.uint8(rgb[y,x,k])
            active=False
            for k in range(3):
                if eight[y,x,k]!=old8[y,x,k]:active=True
                error=max(error,abs(np.float32(eight[y,x,k])-target[y,x,k]))
            if active:changed+=1
    return cancelled,changed,error
