"""Input-only phase-invariant source context; no reference or filename access."""
import numpy as np,cv2
from .exact_median import median_filter
from .carrier_tools import _companion_amp
from .envelope import features as original_features

def source_context(rgb,f):
 w,h=f['analysis_shape'];p=w/f['configuration']['cycles']
 image=f['_analysis_image'] if f.get('_analysis_source') is rgb and '_analysis_image' in f else cv2.resize(np.asarray(rgb,np.float32),(w,h),interpolation=cv2.INTER_AREA)
 k=max(3,round(2*p));k+=1-k%2
 companion=median_filter(image,size=(1,k,1),mode='reflect').astype(np.float32)
 amp=_companion_amp(companion,f,rgb.shape[1])
 return amp,image,companion

def features(amp,geom,companion_amp=None):
 original=original_features(amp,geom)
 if companion_amp is None:return original
 cross=companion_amp*np.conj(amp)/4
 return np.concatenate((original,np.clip(np.abs(companion_amp)/2,0,5),np.clip(cross.real,-5,5),np.clip(cross.imag,-5,5)),axis=2).astype(np.float32)
