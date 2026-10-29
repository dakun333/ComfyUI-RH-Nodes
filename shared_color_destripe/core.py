"""Self-contained shared-color broadband candidate (2026-10-22).

No continuous-field solver, output gates/caps, luminance lock, clean reference,
or external project dependency. Numeric order follows the source snapshot.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class Parameters:
    period_min: float = 22.0
    period_max: float = 80.0
    guide_sigma: float = 12.0
    sigma_y: float = 12.0
    pad: int = 128
    transition_low: float = .003
    transition_high: float = .005

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 2 <= self.period_min < self.period_max:
            raise ValueError("Require 2 <= period_min < period_max, in native input pixels")
        if self.guide_sigma <= 0 or self.sigma_y < 0:
            raise ValueError("guide_sigma must be positive; sigma_y nonnegative")
        if not isinstance(self.pad, int) or self.pad < 1:
            raise ValueError("pad must be a positive integer")
        if self.transition_low <= 0 or self.transition_high <= 0:
            raise ValueError("Frequency transitions must be positive")


@dataclass(frozen=True)
class Prior:
    base: Parameters = Parameters()

    def validate(self):
        self.base.validate()


@dataclass(frozen=True)
class Settings:
    min_tile_rank: float = .70
    min_color_separation: float = .50
    cluster_cosine: float = .94
    max_gradient: float = 1.8
    min_support: int = 6
    denominator_floor: float = .15
    along_ratio: float = .14
    gains: tuple = (.8, .9)
    # Low-level callers retain the historical policy; the node opts in.
    include_near_white: bool = False

    def validate(self):
        if not isinstance(self.include_near_white, bool): raise ValueError("include_near_white must be boolean")
        for value in [self.min_tile_rank, self.min_color_separation, self.cluster_cosine, self.denominator_floor]:
            if not math.isfinite(value) or not 0 < value < 1: raise ValueError('Invalid fractional setting')
        if not isinstance(self.min_support,int) or isinstance(self.min_support,bool) or self.min_support < 3: raise ValueError('Invalid support')
        if not math.isfinite(self.max_gradient) or self.max_gradient<=0 or not math.isfinite(self.along_ratio) or self.along_ratio<=0: raise ValueError('Invalid scale')
        if not self.gains or any(not math.isfinite(g) or not 0<=g<=1 for g in self.gains): raise ValueError('Invalid gains')


def guided_detail(image: np.ndarray, params: Parameters) -> np.ndarray:
    """Horizontal self-guided residual with a shared RGB edge coefficient."""
    radius = max(2, int(round(0.3 * params.period_max)))

    def average(values):
        return cv2.boxFilter(values, -1, (2 * radius + 1, 1),
                             borderType=cv2.BORDER_REFLECT)

    centered = image - 128.0
    mean = average(centered)
    variance = np.maximum(average(centered * centered) - mean * mean, 0)
    edge_variance = np.max(variance, axis=2)
    coefficient = edge_variance / (edge_variance + params.guide_sigma ** 2)
    base = (average(coefficient)[:, :, None] * centered
            + average((1 - coefficient[:, :, None]) * mean))
    return (centered - base).astype(np.float32)


def quantize(image: np.ndarray) -> np.ndarray:
    return np.uint8(np.clip(np.rint(image), 0, 255))


def _short_vertical_band(residual: np.ndarray, params: Parameters,
                         max_slope: float) -> np.ndarray:
    """Smooth x-frequency band with a near-vertical angular taper."""
    pad = params.pad
    z = np.pad(residual, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    h, w = z.shape[:2]
    fx, fy = np.fft.rfftfreq(w), np.fft.fftfreq(h)
    across = 0.5 * (np.tanh((fx - 1 / params.period_max) / params.transition_low)
                   - np.tanh((fx - 1 / params.period_min) / params.transition_high))
    across[0] = 0
    along = np.exp(-2 * np.pi**2 * params.sigma_y**2 * fy**2)
    ratio = np.abs(fy[:, None]) / np.maximum(fx[None, :] * max_slope, 1e-12)
    angular = np.exp(-np.minimum(ratio, 10.)**8)
    spectrum = np.fft.rfft2(z, axes=(0, 1))
    field = np.fft.irfft2(spectrum * (along[:, None] * across[None, :] * angular)[:, :, None],
                         s=(h, w), axes=(0, 1))
    return field[pad:-pad, pad:-pad].astype(np.float32)


def directed_round(target: np.ndarray, enabled: np.ndarray,
                   fallback: np.ndarray) -> np.ndarray:
    """Deterministic first-order error feedback along y inside enabled runs."""
    h, w, _ = target.shape
    out = fallback.copy()
    indices = np.arange(w, dtype=np.uint32) % 256
    phase = np.zeros(w, np.float32)
    for bit in range(8):
        phase += ((indices >> bit) & 1) * 2. ** (-bit - 1)
    phase = phase[:, None] - 0.5
    error = np.broadcast_to(phase, (w, 3)).copy()
    for y in range(h):
        row = target[y]
        value = row + error
        rounded = np.clip(np.floor(value + 0.5), np.floor(row), np.ceil(row))
        out[y, enabled[y]] = rounded[enabled[y]].astype(np.uint8)
        error = np.where(enabled[y, :, None], value - rounded, phase)
    return out


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def shade_tangent(rgb):
    # Linear sRGB illumination tangent expressed in encoded RGB.
    a=np.asarray(rgb, dtype=np.float64)
    return normalize(np.where(a > 10.31475, (a+14.025)/2.4, a))


def canonical(v):
    return v*np.sign(v[np.argmax(abs(v))])


def scalar_filter(a, prior, settings):
    scale=math.sqrt(prior.base.period_min*prior.base.period_max)
    p=replace(prior.base,sigma_y=settings.along_ratio*scale)
    return _short_vertical_band(a[:,:,None],p,.5)[:,:,0]


def estimate_color(src, prior, settings):
    scale=math.sqrt(prior.base.period_min*prior.base.period_max)
    residual=guided_detail(src,prior.base)
    f=np.stack([scalar_filter(residual[:,:,c],prior,settings) for c in range(3)],axis=2)
    size=max(40,round(2*scale));h,w=src.shape[:2];tiles=[]
    low=cv2.GaussianBlur(src,(0,0),.12*scale)
    gy,gx=np.gradient(low,axis=(0,1))
    for y in range(0,h-size+1,size):
        for x in range(0,w-size+1,size):
            region=np.s_[y:y+size,x:x+size]
            mean=np.mean(low[region],axis=(0,1)).astype(float)
            gradient=float(np.sqrt(np.mean(gx[region]**2+gy[region]**2)))
            if mean.min()<5 or (not settings.include_near_white and mean.max()>249) or gradient>settings.max_gradient:continue
            z=f[region].reshape(-1,3).astype(float);z-=np.median(z,axis=0)
            amplitude=np.linalg.norm(z,axis=1)
            weights=np.minimum(1,2*np.median(amplitude)/np.maximum(amplitude,1e-9))
            cov=(z*weights[:,None]).T@z/max(weights.sum(),1e-9)
            vals,vecs=np.linalg.eigh(cov);direction=canonical(vecs[:,-1])
            rank=float(vals[-1]/max(vals.sum(),1e-12))
            separation=float(1-(direction@shade_tangent(mean))**2)
            if vals[-1]<.025 or rank<settings.min_tile_rank or separation<settings.min_color_separation:continue
            tiles.append(dict(x=x,y=y,rank=rank,separation=separation,direction=direction.tolist(),rms=float(np.sqrt(np.trace(cov)/3))))
    info=dict(tile_size=size,candidate_tile_count=len(tiles),tiles=tiles)
    if len(tiles)<settings.min_support:return None,{**info,'reason':'insufficient_color_tiles'}
    directions=np.array([t['direction'] for t in tiles]);dots=abs(directions@directions.T)
    inlier=dots[np.argmax((dots>=settings.cluster_cosine).sum(axis=1))]>=settings.cluster_cosine
    for _ in range(2):
        _,v=np.linalg.eigh(directions[inlier].T@directions[inlier]);c=canonical(v[:,-1])
        inlier=abs(directions@c)>=settings.cluster_cosine
    count=int(inlier.sum())
    info.update(support_tile_count=count,support_fraction=count/len(tiles),color_direction=c.tolist(),supporting_tiles=[t for t,k in zip(tiles,inlier) if k])
    if count<settings.min_support:return None,{**info,'reason':'insufficient_consensus'}
    split=[]
    for parity in [0,1]:
        mask=inlier & np.array([(t['x']//size+t['y']//size)%2==parity for t in tiles])
        if mask.sum()>=3:
            _,v=np.linalg.eigh(directions[mask].T@directions[mask]);split.append(canonical(v[:,-1]))
    info['split_directions']=[v.tolist() for v in split]
    info['split_angle_degrees']=float(np.degrees(np.arccos(np.clip(abs(split[0]@split[1]),0,1)))) if len(split)==2 else None
    info['median_support_angle_degrees']=float(np.median(np.degrees(np.arccos(np.clip(abs(directions[inlier]@c),0,1)))))
    return c.astype(np.float32),info


def separate_scalar(residual, tangent, color, floor):
    """Oblique projection separates shading from the full RGB pollution."""
    dot=tangent@color
    detector=color-tangent*dot[:,:,None]
    denominator=np.maximum(1-dot*dot,floor)
    return (np.sum(residual*detector,axis=2)/denominator).astype(np.float32)


def restore(rgb,prior=Prior(),settings=Settings(),across_sigma=.65):
    prior.validate();settings.validate();a=np.asarray(rgb)
    if isinstance(across_sigma,bool) or not math.isfinite(across_sigma) or across_sigma<=0:
        raise ValueError("across_sigma must be finite and positive")
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<2:raise ValueError('Expected uint8 HxWx3 RGB')
    src=a.astype(np.float32);color,evidence=estimate_color(src,prior,settings)
    info=dict(method='shared_color_broadband',settings=asdict(settings),evidence=evidence,luminance_locked=False,uses_simulator=False,uses_pilot=False,application_gates=False,correction_caps=False,application_x_sigma=across_sigma,application_frequency_band=False,color_fit_and_guide_still_use_scale_prior=True)
    if color is None:return src,{**info,'status':'abstained'},dict(field=np.zeros_like(src))
    scale=math.sqrt(prior.base.period_min*prior.base.period_max)
    tangent=shade_tangent(cv2.GaussianBlur(src,(0,0),.3*scale)).astype(np.float32)
    output=src.copy()
    for gain in settings.gains:
        residual=guided_detail(output,prior.base)
        scalar=separate_scalar(residual,tangent,color,settings.denominator_floor)
        scalar=cv2.GaussianBlur(scalar,(0,0),across_sigma,sigmaY=settings.along_ratio*scale,borderType=cv2.BORDER_REFLECT)
        output-=gain*scalar[:,:,None]*color
    if not np.isfinite(output).all():raise FloatingPointError('Invalid output')
    field=src-output
    info.update(status='processed_not_quality_approved',correction_rms=float(np.sqrt(np.mean(field**2))))
    return output,info,dict(field=field,color=color)


def encode_directed8(original, target):
    """Historical directed8 encoding."""
    target = np.clip(target, 0, 255)
    nearest = quantize(target)
    enabled = np.max(np.abs(original.astype(np.float32) - target), axis=2) > 1e-5
    return directed_round(target, enabled, nearest)
