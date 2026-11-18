"""Strict CPU kernels for existing 3-channel analysis. No fastmath or new rules."""
import numpy as np
from numba import njit

@njit(cache=True,nogil=True,fastmath=False)
def real_complex_matvec(m,a):
    H,W=a.shape[:2];out=np.empty(a.shape,np.complex64)
    for y in range(H):
        for x in range(W):
            for i in range(3):
                q=np.complex64(0)
                for j in range(3):
                    q=np.complex64(q+np.complex64(np.complex64(m[y,x,i,j])*a[y,x,j]))
                out[y,x,i]=q
    return out

@njit(cache=True,nogil=True,fastmath=False)
def tensor_moments(gx,gy):
    H,W=gx.shape[:2];jxx=np.empty((H,W),np.float32);jyy=np.empty((H,W),np.float32);jxy=np.empty((H,W),np.float32)
    three=np.float32(3)
    for y in range(H):
        for x in range(W):
            xx=np.float32(0);yy=np.float32(0);xy=np.float32(0)
            for k in range(3):
                xx=np.float32(xx+np.float32(gx[y,x,k]*gx[y,x,k]));yy=np.float32(yy+np.float32(gy[y,x,k]*gy[y,x,k]));xy=np.float32(xy+np.float32(gx[y,x,k]*gy[y,x,k]))
            jxx[y,x]=np.float32(xx/three);jyy[y,x]=np.float32(yy/three);jxy[y,x]=np.float32(xy/three)
    return jxx,jyy,jxy

@njit(cache=True,nogil=True,fastmath=False)
def tensor_finish(jxx,jyy,jxy):
    H,W=jxx.shape;nx=np.empty((H,W),np.float32);ny=np.empty_like(nx);co=np.empty_like(nx);strength=np.empty_like(nx)
    eps=np.float32(1e-12);two=np.float32(2);zero=np.float32(0)
    for y in range(H):
        for x in range(W):
            a=np.float32(jxx[y,x]-jyy[y,x]);b=np.float32(two*jxy[y,x]);mag=np.float32(np.sqrt(np.float32(np.float32(a*a)+np.float32(b*b))));trace=np.float32(jxx[y,x]+jyy[y,x])
            nx[y,x]=np.float32(a/max(mag,eps));ny[y,x]=np.float32(b/max(mag,eps));co[y,x]=np.float32(mag/max(trace,eps));strength[y,x]=np.float32(np.sqrt(max(trace,zero)))
    return nx,ny,co,strength
