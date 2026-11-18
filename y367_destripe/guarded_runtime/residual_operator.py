"""BR10 experiment: remove its geometric edge halo, filter final FLOAT RGB direction.
No refit, no external reference, no additional residual pass or replacement encoder.
"""
import os
from pathlib import Path
import time, hashlib
import numpy as np
from . import br10_core as core
try:
    from . import native_pipeline as _pipeline_native
except ImportError:
    _pipeline_native=None

ROOT=Path(__file__).absolute().parent
MODEL_SHA256='32c4375a718762bec6352c12f320d3f760a04cc03a8b39ba46f2d09967870268'
COLOR_RANGES=((30.,90.),(90.,150.),(210.,270.),(270.,330.))


def color_mask(removed):
    """HSV hue of the raw signed FLOAT RGB vector, before 8-bit/display clipping.
    Yellow 30..90, green 90..150, blue 210..270, purple 270..330.
    Preserve an admitted vector as is. Never rotate/project/recolor its channels.
    """
    r,g,b=np.moveaxis(removed,-1,0)
    hi=np.maximum(np.maximum(r,g),b);lo=np.minimum(np.minimum(r,g),b)
    span=hi-lo;den=np.maximum(span,np.float32(1e-12))
    hue=np.where(hi==r,(g-b)/den,np.where(hi==g,(b-r)/den+2.,(r-g)/den+4.))
    hue=np.mod(hue*60.,360.)
    return (span>1e-5)&(((hue>=30.)&(hue<=150.))|((hue>=210.)&(hue<=330.)))


def restrict_target(rgb,target):
    start=time.perf_counter()
    # Dispatch only the original FLOAT32 input. Preserve the historical fallback
    # for public callers using other dtypes rather than changing it via coercion.
    native=_pipeline_native is not None and rgb.dtype==target.dtype==np.float32 and rgb.shape==target.shape and rgb.ndim==3 and rgb.shape[2]==3
    if native:
        removed,result,allowed,active=_pipeline_native.restrict_pixels(rgb,target)
    else:
        removed=np.asarray(rgb,np.float32)-target
        allowed=color_mask(removed)
        active=np.any(removed!=0,axis=2)
        result=target.copy();result[~allowed]=rgb[~allowed]
    total=float(np.sum(removed.astype(np.float64)**2))
    rejected=float(np.sum(removed[~allowed].astype(np.float64)**2))
    report=dict(policy='PG_BY_FLOAT_hue_30deg',hue_degrees=[list(r) for r in COLOR_RANGES],
                scope='final total real FLOAT RGB correction, after both axes; before directed8 encoding',
                achromatic_FLOAT_rejected=True,channels_rotated_or_projected=False,
                raw_active_pixel_percent=float(active.mean()*100),
                rejected_raw_active_pixel_percent=float((active&~allowed).sum()/max(int(active.sum()),1)*100),
                rejected_raw_RGB_energy_percent=100*rejected/max(total,1e-30),
                filter_seconds=time.perf_counter()-start)
    if not native:
        assert np.array_equal(result[~allowed],rgb[~allowed])
        assert np.array_equal(result[allowed],target[allowed])
    # Native path performs these same whole-RGB copies by construction; tested
    # independently, avoiding two advanced-index copies used only for assertions.
    return result,report,allowed


def encode(rgb,baseline,target,prior8=None):
    # Keep the historical BR10 encoder and its baseline fallback.
    eight=core.encode(rgb,baseline,target,prior8)
    same=np.all(target==rgb,axis=2)
    eight[same]=np.uint8(np.clip(np.rint(rgb[same]),0,255))
    assert np.max(np.abs(eight.astype(np.float32)-target))<1.00001
    return eight


def apply(rgb,alpha,*,orientation='both',maps=False,weights_path=None,
          edge_protection=False,color_policy='pgby'):
    if color_policy not in ('pgby','none'):raise ValueError('Unknown color policy')
    if hashlib.sha256((ROOT/'model.npz').read_bytes()).hexdigest()!=MODEL_SHA256:
        raise ValueError('Frozen original Y367 model hash changed')
    if weights_path is not None and hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()!=MODEL_SHA256:
        raise ValueError('This experiment does not allow refitted weights')
    start=time.perf_counter()
    raw,report,cache=core.apply(rgb,alpha,orientation=orientation,maps=True,
                              weights_path=weights_path,edge_protection=edge_protection)
    if color_policy=='pgby':
        target,creport,allowed=restrict_target(rgb,raw)
    else:
        target=raw;allowed=np.ones(raw.shape[:2],bool)
        creport=dict(policy='none',filter_seconds=0.)
    report.update(implementation='BR10_no_edge_PGBY_FLOAT',
                  method='BR10_original_model_no_edge_PGBY' if not edge_protection and color_policy=='pgby' else 'BR10_controlled_ablation',
                  geometric_edge_halo_enabled=bool(edge_protection),
                  only_boundary_halo_removed=not edge_protection,
                  alpha_protection_retained=True,source_component_evidence_retained=True,
                  recurrence_and_phase_controls_retained=True,
                  residual_branch_already_without_geometric_halo_in_original_BR10=True,
                  final_color_gate=creport,model_unchanged=True,
                  additional_CNN_calls=0,additional_FFT_calls=0,
                  original_fixed_five_refreshes_each_axis=True,
                  references_read=False,ROI_read=False,training_this_round=False,
                  boundary_release_not_guaranteed_to_release_source_evidence_rejections=True,
                  seconds=time.perf_counter()-start,
                  limitation='Color direction is a constraint, not proof of artifact. A wrong PG/BY correction can still survive.')
    cache.update(raw_target=raw,color_allowed=allowed)
    return (target,report,cache) if maps else (target,report)
