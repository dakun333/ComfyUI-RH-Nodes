"""Sparse, local upper-quantile amplitude reference near original protection."""
import numpy as np
from numba import njit,prange

@njit(cache=True,nogil=True,fastmath=False,parallel=True)
def magnitude(rgb,target):
    h,w,_=rgb.shape;m=np.empty((h,w),np.float32)
    for y in prange(h):
        for x in range(w):
            m[y,x]=max(abs(rgb[y,x,0]-target[y,x,0]),abs(rgb[y,x,1]-target[y,x,1]),abs(rgb[y,x,2]-target[y,x,2]))
    return m

@njit(cache=True,nogil=True,fastmath=False,parallel=True)
def local_outlier_gain(rgb,target,eight,current_magnitude,reference_magnitude,distance,offsets,radius,clearance,start_ratio,full_ratio):
    h,w,_=rgb.shape;gain=np.ones((h,w),np.float32);reference=np.zeros((h,w),np.float32);affected=np.zeros((h,w),np.bool_);count=np.zeros((h,w),np.uint16);effective=target.copy()
    slots=len(offsets)**2
    for y in prange(h):
        for x in range(w):
            a=current_magnitude[y,x];d=distance[y,x]
            if a<np.float32(1.5) or d<=0 or d>=radius:continue
            if (np.float32(eight[y,x,0])==rgb[y,x,0] and np.float32(eight[y,x,1])==rgb[y,x,1] and np.float32(eight[y,x,2])==rgb[y,x,2]):continue
            # References come from the PRE-ST1 output, away from protection. Do not
            # dilute the reference distribution with cancelled/natural zero pixels.
            values=np.empty(slots,np.float32);weights=np.empty(slots,np.float32);n=0;total=np.float32(0)
            for dy in offsets:
                yy=y+dy
                if yy<0 or yy>=h:continue
                for dx in offsets:
                    xx=x+dx
                    if xx<0 or xx>=w or distance[yy,xx]<=clearance:continue
                    v=reference_magnitude[yy,xx]
                    if v<np.float32(.125):continue
                    cd=max(abs(rgb[yy,xx,0]-rgb[y,x,0]),abs(rgb[yy,xx,1]-rgb[y,x,1]),abs(rgb[yy,xx,2]-rgb[y,x,2]))
                    z=np.float32(cd/np.float32(24))
                    # Prefer similar source colors continuously. A thin outline
                    # need not have interior same-color pixels: keep weak context
                    # weights instead of a discontinuous fallback or zero cutoff.
                    weight=np.float32(np.float32(.2)+np.float32(.8)/np.float32(np.float32(1)+z*z))
                    values[n]=v;weights[n]=weight;n+=1;total+=weight
            count[y,x]=n
            if n<12:continue
            order=np.argsort(values[:n]);acc=np.float32(0);upper=values[order[n-1]]
            for j in range(n):
                i=order[j];acc+=weights[i]
                if acc>=np.float32(.9)*total:
                    upper=values[i];break
            # One native RGB level floor keeps weak correct edits out of this rule.
            ref=max(np.float32(1),upper);reference[y,x]=ref
            ratio=np.float32(a/ref)
            u=min(np.float32(1),max(np.float32(0),np.float32((ratio-start_ratio)/(full_ratio-start_ratio))))
            evidence=np.float32(u*u*np.float32(np.float32(3)-np.float32(2)*u))
            z=min(np.float32(1),max(np.float32(0),np.float32((radius-d)/(radius*np.float32(.5)))))
            proximity=np.float32(z*z*np.float32(np.float32(3)-np.float32(2)*z))
            g=np.float32(np.float32(1)-np.float32(proximity*evidence*np.float32(np.float32(1)-ref/a)))
            if g>=np.float32(1):continue
            gain[y,x]=g;affected[y,x]=True
            for k in range(3):effective[y,x,k]=np.float32(rgb[y,x,k]-np.float32(g*np.float32(rgb[y,x,k]-target[y,x,k])))
    return effective,gain,reference,affected,count
