"""Signed RGB recurrence on original analysis pixels; no reference/filename use."""
import numpy as np,cv2
from .carrier_tools import _shift

def recurrence(rgb,f,amplitude=(.10,.35),score_range=(.25,.60),along=.40):
 w,h=f['analysis_shape'];p=w/48;image=f['_analysis_image'] if f.get('_analysis_source') is rgb and '_analysis_image' in f else cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA)
 pre=cv2.GaussianBlur(image,(0,1),max(.5,p*.10));dx=max(1.,p*.16);transfer=(2-2*np.cos(2*np.pi*dx/p))**2
 noise=(_shift(pre,-2*dx)-4*_shift(pre,-dx)+6*pre-4*_shift(pre,dx)+_shift(pre,2*dx))/transfer
 smooth=lambda q:cv2.GaussianBlur(q,(0,0),max(.5,p*.40),sigmaY=max(.5,p*along))
 energy=smooth(np.mean(noise*noise,axis=2));scores=[]
 for signed,half in ((1,False),(-1,False),(1,True),(-1,True)):
  best=np.full((h,w),-1,np.float32)
  for jitter in (-.20,0,.20):
   peer=_shift(noise,signed*p*(.5 if half else 1)*(1+jitter));cross=smooth(np.mean(noise*peer,axis=2));norm=np.sqrt(energy*smooth(np.mean(peer*peer,axis=2)))+1e-6
   best=np.maximum(best,(-cross if half else cross)/norm)
  scores.append(best)
 # Full and signed half period agreement is required on at least one side.
 # A natural contour/occlusion may invalidate the opposite side.
 score=np.maximum(np.minimum(scores[0],scores[2]),np.minimum(scores[1],scores[3]));strength=np.sqrt(energy)
 confidence=np.clip((score-score_range[0])/(score_range[1]-score_range[0]),0,1)*np.clip((strength-amplitude[0])/(amplitude[1]-amplitude[0]),0,1)
 gw,gh=f['grid_shape'];grid=cv2.resize(confidence,(gw,gh),interpolation=cv2.INTER_AREA)
 return grid,dict(score=score,strength=strength,confidence=confidence)

