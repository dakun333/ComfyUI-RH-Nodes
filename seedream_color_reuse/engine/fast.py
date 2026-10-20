"""Exact-arithmetic CPU acceleration of the complete standard pipeline.

A private module namespace keeps optimized globals out of the standard engine.
All protection policies and thresholds are retained. Parallelism is frozen
per-layer work plus per-channel SciPy filtering, using one bounded thread pool.
Channel order, float32 intermediates and boundary modes are unchanged. No
lower-resolution processing, quality preset, affinity or new dependency.
"""
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import importlib
from pathlib import Path
import sys
import time
from threading import Semaphore
from types import ModuleType

import numpy as np
from scipy import ndimage as ndi
from .core import composite as _standard_composite

_POOL = ContextVar("seedream_fast_pool", default=None)
_LAYER_POOL = ContextVar("seedream_fast_layer_pool", default=None)
_LAYER_WORKERS = ContextVar("seedream_fast_layer_workers", default=3)
_STAGE_TIMES = ContextVar("seedream_fast_stage_times", default=None)
_REUSE_COMPOSITES = ContextVar("seedream_fast_reuse_composites", default=True)


def _map_layer_stage(operation, layers):
    """Parallel local work; return in stack order; never share canvas writes.

    Pool tasks intentionally do not inherit _POOL. Their SciPy filters run
    serially, avoiding nested channel jobs / same-pool deadlocks. Between layer
    stages the calling thread can still use three-channel filtering.
    """
    started = time.perf_counter()
    pool = _LAYER_POOL.get()
    try:
        if pool is None or len(layers) < 2 or sum(l.alpha.size for l in layers) < 65536:
            for i, layer in enumerate(layers):
                yield operation(i, layer)
            return
        # Large boxes first for load balancing, independent of result order.
        order = sorted(range(len(layers)), key=lambda i: layers[i].alpha.size, reverse=True)
        limit = Semaphore(_LAYER_WORKERS.get())
        def run_local(i):
            with limit:
                return operation(i, layers[i])
        futures = {i: pool.submit(run_local, i) for i in order}
        for i in range(len(layers)):
            yield futures[i].result()
    finally:
        timings = _STAGE_TIMES.get()
        if timings is not None:
            timings[operation.__name__] = time.perf_counter() - started


def _unchanged_composite(layers, shape, cached):
    # Pipeline hands us a snapshot only where no RGB/alpha write intervenes.
    # Do not cache by array identity: multiple solvers mutate RGB in place.
    return cached if _REUSE_COMPOSITES.get() else _runtime.composite(layers, shape)


class _ExactPixelReductions:
    """Accelerate fixed three-channel reductions, not spatial summations.

    Preserve left-to-right channel order, dtype, NaN propagation and signed
    zero behavior. No monkeypatch of the shared numpy module. All other API
    forms delegate unchanged (out/where/initial/keepdims/tuple axes included).
    """
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def _three_channels(a, args, kwargs):
        axis = kwargs.get("axis", args[0] if args else None)
        return (type(a) is np.ndarray and a.ndim >= 2 and a.shape[-1] == 3
                and a.size >= 12288 and a.dtype in (np.dtype("float32"), np.dtype("bool"))
                and len(args) <= 1 and not (args and "axis" in kwargs)
                and set(kwargs) <= {"axis"}
                and isinstance(axis, int) and not isinstance(axis, bool)
                and axis in (-1, a.ndim - 1))

    def max(self, a, *args, **kwargs):
        if self._three_channels(a, args, kwargs):
            result = np.maximum(a[..., 0], a[..., 1])
            return np.maximum(result, a[..., 2], out=result)
        return np.max(a, *args, **kwargs)

    def min(self, a, *args, **kwargs):
        if self._three_channels(a, args, kwargs):
            result = np.minimum(a[..., 0], a[..., 1])
            return np.minimum(result, a[..., 2], out=result)
        return np.min(a, *args, **kwargs)

    def all(self, a, *args, **kwargs):
        if self._three_channels(a, args, kwargs) and a.dtype == np.bool_:
            result = np.logical_and(a[..., 0], a[..., 1])
            return np.logical_and(result, a[..., 2], out=result)
        return np.all(a, *args, **kwargs)

    def any(self, a, *args, **kwargs):
        if self._three_channels(a, args, kwargs) and a.dtype == np.bool_:
            result = np.logical_or(a[..., 0], a[..., 1])
            return np.logical_or(result, a[..., 2], out=result)
        return np.any(a, *args, **kwargs)


class _ParallelFilters:
    def __getattr__(self, name):
        return getattr(ndi, name)

    def gaussian_filter(self, image, sigma, order=0, output=None, mode="reflect",
                        cval=0.0, truncate=4.0, *, radius=None, axes=None):
        pool = _POOL.get()
        sigma_tuple = tuple(sigma) if not np.isscalar(sigma) else None
        if (pool is not None and image.ndim == 3 and image.shape[-1] == 3
                and image.dtype == np.float32 and image.shape[0]*image.shape[1] >= 65536
                and sigma_tuple is not None and len(sigma_tuple) == 3
                and sigma_tuple[2] == 0 and np.isscalar(order) and order == 0
                and output is None and axes is None and np.isscalar(mode)
                and radius is None):
            jobs = [pool.submit(ndi.gaussian_filter, image[..., c], sigma_tuple[:2],
                                mode=mode, cval=cval, truncate=truncate) for c in range(3)]
            return np.stack([job.result() for job in jobs], axis=-1)
        return ndi.gaussian_filter(image, sigma, order=order, output=output, mode=mode,
                                   cval=cval, truncate=truncate, radius=radius, axes=axes)


def _composite(layers, shape):
    """Source-over with reusable scratch; preserve original NumPy operation order."""
    # API users may supply float64 RGB. Keep the original mixed-precision
    # expression instead of rounding its product early into float32 scratch.
    if any(layer.rgb.dtype != np.float32 for layer in layers):
        return _standard_composite(layers, shape)
    rgb = np.zeros((*shape, 3), np.float32)
    alpha = np.zeros(shape, np.float32)
    scratch = np.empty_like(rgb)
    for layer in layers:
        s = layer.region
        a = layer.alpha.astype(np.float32) / 255
        inverse = 1 - a
        local = scratch[s]
        np.multiply(layer.rgb, a[..., None], out=local)
        np.multiply(rgb[s], inverse[..., None], out=rgb[s])
        np.add(local, rgb[s], out=rgb[s])
        np.multiply(alpha[s], inverse, out=alpha[s])
        np.add(a, alpha[s], out=alpha[s])
    return rgb, alpha


def _confidence_maps(predicted, reference, eligible, cfg):
    # Reuse the two identical high-pass computations instead of four filters.
    difference = np.max(np.abs(predicted-reference), axis=-1)
    source_high = reference - _runtime.smooth(reference, 1.1)
    model_high = predicted - _runtime.smooth(predicted, 1.1)
    structural = np.max(np.abs(model_high-source_high), axis=-1)
    source_detail = np.max(np.abs(source_high), axis=-1)
    model_detail = np.max(np.abs(model_high), axis=-1)
    unmatched = (source_detail > 3) & (model_detail < source_detail*.3) & eligible
    unmatched = ndi.binary_closing(unmatched, iterations=1)
    holes = ndi.binary_fill_holes(unmatched) & ~unmatched
    labels, _ = ndi.label(holes)
    sizes = np.bincount(labels.ravel())
    small = sizes <= max(512, int(eligible.size*.0005))
    small[0] = False
    unmatched |= small[labels]
    unmatched &= eligible
    bad = ((difference > cfg.mismatch_threshold) |
           (structural > cfg.structure_threshold) | unmatched | ~eligible)
    distance = _runtime.interior_distance(~bad)
    trust = np.clip((distance-1)/(cfg.guard_radius+1), 0, 1).astype(np.float32)
    return trust, bad, structural


def _make_runtime():
    # Load the *same* pipeline modules into a private package, not monkeypatch
    # scipy or standard engine globals. Safe when both nodes run concurrently.
    name = __package__ + "._fast_runtime"
    package = ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parent)]
    package.__package__ = name
    sys.modules[name] = package
    runtime = importlib.import_module(name + ".core")
    facade = _ParallelFilters()
    numpy_facade = _ExactPixelReductions()
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith(name + ".") and getattr(module, "ndi", None) is ndi:
            module.ndi = facade
        if module_name.startswith(name + ".") and getattr(module, "np", None) is np:
            module.np = numpy_facade
    class CachedColorField(runtime.ColorField):
        def apply(self, image):
            flat = image.reshape(-1, 3)
            if self.tree is None or len(flat) < 65536 or image.dtype != np.float32:
                return super().apply(image)
            # Exact 8-bit RGB only. Fractional post-correction colors take the
            # original path; NEVER quantize colors to make the cache applicable.
            sample = flat[::max(1, len(flat)//256)]
            if not np.array_equal(sample, np.rint(sample)):
                return super().apply(image)
            integer = flat.astype(np.uint32)
            if not np.array_equal(flat, integer):
                return super().apply(image)
            codes = integer[:, 0]*65536 + integer[:, 1]*256 + integer[:, 2]
            _, indices, inverse = np.unique(codes, return_index=True, return_inverse=True)
            if len(indices) > len(flat)//2:
                return super().apply(image)
            mapped, support = super().apply(flat[indices, None, :])
            return mapped[:, 0][inverse].reshape(image.shape), support[:, 0][inverse].reshape(image.shape[:2])

    runtime.ColorField = CachedColorField
    runtime.composite = _composite
    runtime.confidence_maps = _confidence_maps
    runtime._map_layer_stage = _map_layer_stage
    runtime._unchanged_composite = _unchanged_composite
    standard_repair = runtime.repair_interior_coverage
    def repair_interior_coverage(layers, weights, reference, eligible, protected,
                                 exact_candidates, exact_requested, stats, cfg):
        if cfg.repair_opaque_interiors:
            return standard_repair(layers, weights, reference, eligible, protected,
                                   exact_candidates, exact_requested, stats, cfg)
        # Disabled legacy repair originally recomposited the entire stack then
        # only wrote zero counters. Keep those counters, not the unused image.
        for layer_stats in stats:
            layer_stats["interior_alpha_repaired_pixels"] = 0
        return np.zeros(eligible.shape, bool)
    runtime.repair_interior_coverage = repair_interior_coverage
    return runtime


_runtime = _make_runtime()


def calibrate_fast(layers, source_rgba, cfg=None, *, protected_mask=None, model_guide=None,
                   layer_workers=3, reuse_composites=True):
    if isinstance(layer_workers, bool) or layer_workers not in (0, 1, 2, 3):
        raise ValueError("layer_workers must be 0, 1, 2 or 3")
    timings = {}
    # Optional arguments are engine experiment controls, not new node widgets.
    # Total concurrency is bounded: layer work and channel work share one pool.
    with ThreadPoolExecutor(max_workers=max(3, layer_workers), thread_name_prefix="seedream-fast") as pool:
        token = _POOL.set(pool)
        layer_token = _LAYER_POOL.set(pool if layer_workers > 1 else None)
        workers_token = _LAYER_WORKERS.set(layer_workers)
        timing_token = _STAGE_TIMES.set(timings)
        reuse_token = _REUSE_COMPOSITES.set(reuse_composites)
        try:
            result = _runtime.calibrate(layers, source_rgba, cfg,
                                        protected_mask=protected_mask, model_guide=model_guide)
        finally:
            _REUSE_COMPOSITES.reset(reuse_token)
            _STAGE_TIMES.reset(timing_token)
            _LAYER_WORKERS.reset(workers_token)
            _LAYER_POOL.reset(layer_token)
            _POOL.reset(token)
    # Preserve standard diagnostics and version; label the execution backend.
    result[1]["execution_backend"] = "fast-exact-cpu"
    result[1]["execution_stage_seconds"] = timings
    result[1]["execution_layer_workers"] = layer_workers
    result[1]["execution_optimizations"] = [
        "parallel RGB Gaussian channels; identical SciPy axis order and boundary modes",
        "deduplicated confidence high-pass filters", "scratch-buffer source-over",
        "exact unique 8-bit RGB map evaluation; fractional colors use the original path",
        "frozen-evidence layer stages with ordered canvas aggregation; bounded pool",
        "reuse explicitly unchanged composite snapshots",
        "skip unused composite in disabled legacy alpha repair",
        "fixed-order exact three-channel pixel reductions; general reductions unchanged",
    ]
    return result
