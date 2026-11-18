"""CG1: reject near-neutral real RGB corrections, including actual 8-bit output.

The detector, reconstruction, original hue gate and directed8 codec are frozen.
An admitted output pixel is byte-identical to NOEDGE_PGBY; a rejected pixel is
byte-identical to the input. No channel projection, neutral subtraction, or
recoloring of the diagnostic PNG is performed.
"""
import os
import time
import numpy as np
from . import residual_operator as previous
try:
    from .native_output import guard_candidates as _native_guard
except ImportError:
    _native_guard=None

MIN_CHROMA_RATIO = np.float32(0.25)
METHOD = 'CG1_chroma_and_actual8_guard'


def chroma_mask(removed, threshold=MIN_CHROMA_RATIO):
    """Scale-invariant RGB channel spread / largest absolute component >= 1/4.

    Small colorful corrections remain possible; nearly equal-channel corrections
    are rejected regardless of their nominal hue. This is an empirical operating
    choice, not a learned artifact/correctness classifier.
    """
    r, g, b = np.moveaxis(np.asarray(removed, np.float32), -1, 0)
    high = np.maximum(np.maximum(r, g), b)
    low = np.minimum(np.minimum(r, g), b)
    spread = high - low
    amplitude = np.maximum(np.maximum(np.abs(r), np.abs(g)), np.abs(b))
    return (spread > np.float32(1e-5)) & (spread >= np.float32(threshold) * amplitude)


def palette_mask(removed):
    """Exact 30..150 or 210..330 degree hue sectors without division.

    R/B similarity is the green/purple cone; R/G similarity is blue/yellow.
    This condition is translation-invariant, but does not itself reject a large
    achromatic common component. Pair it with chroma_mask.
    """
    r, g, b = np.moveaxis(np.asarray(removed, np.float32), -1, 0)
    spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    return (spread > np.float32(1e-5)) & ((2 * np.abs(r - b) <= spread) | (2 * np.abs(r - g) <= spread))


def admission_mask(removed):
    return chroma_mask(removed) & palette_mask(removed)


def guard_output(original, target, candidate8):
    """Gate the proposed whole RGB correction, never individual channels.

    Also checks the actual clipped x16 map, so the saved view cannot silently
    drift out of the permitted hue/chroma sectors at an 8-level clipping edge.
    Returns the effective FLOAT target and actual native8 output together.
    """
    started = time.perf_counter()
    original = np.asarray(original, np.float32)
    target = np.asarray(target, np.float32)
    candidate8 = np.asarray(candidate8)
    if original.shape != target.shape or original.shape != candidate8.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError('Expected matching HxWx3 arrays')
    if candidate8.dtype != np.uint8:
        raise ValueError('candidate8 must be native uint8')
    original8 = np.uint8(original)
    if not np.array_equal(original, original8.astype(np.float32)):
        raise ValueError('Native integer-valued RGB input is required')
    if _native_guard is not None and target.dtype==np.float32:
        effective,eight,keep,counts,error_max=_native_guard(original,target,candidate8)
        assert error_max < 1.00001
    else:
        removed = original - target
        actual = original - candidate8.astype(np.float32)
        display_vector = np.clip(128 + 16 * actual, 0, 255) - 128
        float_keep = admission_mask(removed)
        actual_keep = admission_mask(actual)
        display_keep = admission_mask(display_vector)
        keep = float_keep & actual_keep & display_keep
        active = np.any(actual != 0, axis=2)
        reject = ~keep
        eight = candidate8.copy();eight[reject] = original8[reject]
        effective = target.copy();effective[reject] = original[reject]
        assert np.array_equal(eight[keep], candidate8[keep])
        assert np.array_equal(eight[reject], original8[reject])
        assert np.max(np.abs(eight.astype(np.float32) - effective)) < 1.00001
        counts=[active.sum(),(active & keep).sum(),(active & ~chroma_mask(removed)).sum(),
            (active & ~actual_keep).sum(),(active & float_keep & actual_keep & ~display_keep).sum()]
    report = dict(
        method=METHOD, minimum_RGB_spread_over_peak_absolute_component=float(MIN_CHROMA_RATIO),
        candidate_changed8_pixels=int(counts[0]), final_changed8_pixels=int(counts[1]),
        near_neutral_FLOAT_candidate_pixels_reverted=int(counts[2]),
        final_changed8_pixels_outside_palette_or_chroma_reverted=int(counts[3]),
        display_clipping_color_pixels_reverted=int(counts[4]),
        original_model_and_codec_unchanged=True, only_keep_whole_RGB_vector_or_restore_input=True,
        channels_projected_rotated_or_individually_clamped=False,
        actual8_and_actual_x16_palette_and_chroma_checked=True,
        diagnostic_recolored=False, detector_rules_or_iterations_added=False,
        guard_seconds=time.perf_counter() - started,
        limitation='An allowed colorful correction is not proof of an artifact; rejecting pixels can also retain real stripes.'
    )
    return effective, eight, report, keep


def process(rgb, alpha, *, orientation='both', keep_edge_protection=False, maps=False):
    """Standalone inference. No reference image, filename, region, or cached output read."""
    started = time.perf_counter()
    target, report, cache = previous.apply(rgb, alpha, orientation=orientation, maps=True,
        edge_protection=keep_edge_protection, color_policy='pgby')
    proposed = previous.encode(rgb, cache['baseline'], target)
    final_target, eight, guard_report, keep = guard_output(rgb, target, proposed)
    report.update(method=METHOD, final_native8_chroma_guard=guard_report,
        compute_and_encode_seconds=time.perf_counter() - started,
        additional_CNN_calls=0, additional_FFT_calls=0, references_read=False, ROI_read=False,
        one_global_chroma_ratio_for_all_inputs=True, trained_this_round=False,
        limitation=guard_report['limitation'])
    if maps:
        cache.update(previous_target=target, previous_actual8=proposed, output_admitted=keep)
        return final_target, eight, report, cache
    return final_target, eight, report
