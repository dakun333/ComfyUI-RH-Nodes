"""Carrier and subpixel shift utilities; no data/reference dependencies."""
import numpy as np,cv2
from . import envelope as base

def _shift(a,dx):
 h,w=a.shape[:2];xx=np.broadcast_to(np.arange(w,dtype=np.float32)[None,:]+dx,(h,w)).copy();yy=np.broadcast_to(np.arange(h,dtype=np.float32)[:,None],(h,w)).copy()
 if np.iscomplexobj(a):return (cv2.remap(a.real,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)+1j*cv2.remap(a.imag,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)).astype(np.complex64)
 return cv2.remap(a,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)

def _companion_amp(image,f,original_width,*,full_output=False):
 # Identical FFT/response arithmetic; reuse shape-only frequency and carrier arrays.
 return base.analysis_amp(image,original_width,tuple(f['grid_shape']),f['configuration'],full_output=full_output)
