"""Exact odd horizontal rank median; preserve SciPy fallback for other call shapes."""
import numpy as np
from scipy.ndimage import median_filter as _scipy_median
try:
 from . import native_kernels as _native
except ImportError:
 _native=None

def median_filter(image,size,mode='reflect'):
 if _native is not None and image.dtype==np.float32 and image.ndim in (2,3) and mode=='reflect' and isinstance(size,tuple) and len(size)==image.ndim and size[0]==1 and (image.ndim==2 or size[2]==1) and size[1]%2==1:
  return _native.horizontal_median(image,size[1])
 return _scipy_median(image,size=size,mode=mode)
