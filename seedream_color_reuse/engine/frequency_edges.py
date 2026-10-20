"""Geometry-first frequency separation for source/model contour conflicts.

The reference and the separated stack do not necessarily share the same
sub-pixel geometry.  Reusing reference RGB at identical canvas coordinates can
therefore inject the reference's high-frequency contour into the model's fixed
alpha geometry.  This module identifies those contour bands before borrowing,
then restores the model layer's high frequencies over the corrected layer's
low-frequency colour field.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def edge_ridge(rgb: np.ndarray, threshold: float) -> np.ndarray:
    """Return a one-pixel-ish colour-gradient ridge without an OpenCV dependency."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3")
    low = ndi.gaussian_filter(rgb.astype(np.float32), (0.65, 0.65, 0), mode="nearest")
    gx = np.stack([ndi.sobel(low[..., c], axis=1, mode="nearest") / 8.0 for c in range(3)], axis=-1)
    gy = np.stack([ndi.sobel(low[..., c], axis=0, mode="nearest") / 8.0 for c in range(3)], axis=-1)
    energy = gx * gx + gy * gy
    channel = np.argmax(energy, axis=-1)
    gradient = np.sqrt(np.take_along_axis(energy, channel[..., None], axis=-1)[..., 0])
    denominator = np.maximum(gradient, 1e-6)
    nx = np.take_along_axis(gx, channel[..., None], axis=-1)[..., 0] / denominator
    ny = np.take_along_axis(gy, channel[..., None], axis=-1)[..., 0] / denominator
    yy, xx = np.indices(gradient.shape, dtype=np.float32)
    plus = ndi.map_coordinates(gradient, [yy + ny, xx + nx], order=1, mode="nearest")
    minus = ndi.map_coordinates(gradient, [yy - ny, xx - nx], order=1, mode="nearest")
    return (gradient > threshold) & (gradient >= plus) & (gradient >= minus)


def build_frequency_edge_gate(mapped: np.ndarray, reference: np.ndarray, layers, weights,
                              eligible: np.ndarray, background_index: int | None, cfg):
    """Build an independent, feathered gate around model/reference/alpha edges.

    The gate is intentionally restricted to pixels influenced by a foreground
    layer.  A fully visible background pixel is therefore never changed and
    the existing background fit/reuse policy remains byte-for-byte compatible.
    """
    if mapped.shape != reference.shape or mapped.shape[:2] != eligible.shape:
        raise ValueError("mapped/reference/eligible shapes do not agree")
    shape = eligible.shape
    model_edge = edge_ridge(mapped, cfg.frequency_edge_threshold)
    reference_edge = edge_ridge(reference, cfg.frequency_edge_threshold)
    alpha_edge = np.zeros(shape, bool)
    editable = np.zeros(shape, bool)
    for i, (layer, weight) in enumerate(zip(layers, weights)):
        # Layer zero is the receiver/background under both supported stack
        # conventions. Frequency repair must have a foreground owner.
        if i == background_index or (background_index is None and i == 0):
            continue
        local_alpha = layer.alpha
        local_boundary = (
            ndi.maximum_filter(local_alpha, size=3, mode="nearest").astype(np.int16)
            - ndi.minimum_filter(local_alpha, size=3, mode="nearest").astype(np.int16)
        ) > 8
        visible_nearby = ndi.maximum_filter(weight, size=3, mode="nearest") > 0.005
        alpha_edge[layer.region] |= local_boundary & visible_nearby
        editable[layer.region] |= visible_nearby

    # Gate only *corresponding but displaced* contours. Exact-coincident
    # texture is safe to reuse, while an unpaired model-only seam should remain
    # available to the existing seam/matte repair. This keeps the mechanism a
    # geometry-conflict gate instead of a blanket edge suppressor.
    model_distance = ndi.distance_transform_edt(~model_edge)
    reference_distance = ndi.distance_transform_edt(~reference_edge)
    correspondence_radius = max(2.0, float(cfg.frequency_edge_radius + 2))
    displaced_pair = (
        model_edge & (reference_distance > 0.75) & (reference_distance <= correspondence_radius)
    ) | (
        reference_edge & (model_distance > 0.75) & (model_distance <= correspondence_radius)
    )
    displaced_alpha = alpha_edge & (reference_distance > 0.75) & (reference_distance <= correspondence_radius)
    seed = displaced_pair & ndi.binary_dilation(editable, iterations=1)
    if cfg.frequency_edge_radius:
        core = ndi.binary_dilation(seed, iterations=cfg.frequency_edge_radius)
    else:
        core = seed
    if core.any():
        distance = ndi.distance_transform_edt(~core)
        # Keep the core and its first exterior pixel at full strength, then
        # use a C1-continuous falloff to avoid creating a new band boundary.
        amount = _smoothstep01(1.0 - (distance - 1.0) / cfg.frequency_edge_feather)
    else:
        amount = np.zeros(shape, np.float32)
    amount = (amount * editable * eligible).astype(np.float32)
    diagnostics = {
        "model_edge": model_edge & editable & eligible,
        "reference_edge": reference_edge & editable & eligible,
        "alpha_edge": alpha_edge & eligible,
        "displaced_pair": displaced_pair & editable & eligible,
        "core": core & editable & eligible,
    }
    return amount, diagnostics


def _normalized_lowpass(rgb: np.ndarray, support: np.ndarray, sigma: float) -> np.ndarray:
    denominator = ndi.gaussian_filter(support.astype(np.float32), sigma, mode="nearest")
    numerator = ndi.gaussian_filter(
        rgb.astype(np.float32) * support[..., None], (sigma, sigma, 0), mode="nearest"
    )
    return np.divide(
        numerator,
        denominator[..., None],
        out=rgb.astype(np.float32).copy(),
        where=denominator[..., None] > 1e-5,
    )


def frequency_split_edges(guide: np.ndarray, corrected: np.ndarray, alpha: np.ndarray,
                          visibility: np.ndarray, locked: np.ndarray,
                          gate_amount: np.ndarray, sigma: float):
    """Use corrected low frequencies and model-guide high frequencies in a gate.

    Alpha-normalized convolution prevents invisible RGB outside the cutout from
    leaking into its antialiased edge. Hidden RGB, alpha and locked regions are
    never modified.
    """
    shape = alpha.shape
    if guide.shape != (*shape, 3) or corrected.shape != guide.shape:
        raise ValueError("guide/corrected/alpha shapes do not agree")
    if visibility.shape != shape or locked.shape != shape or gate_amount.shape != shape:
        raise ValueError("visibility/locked/gate shapes do not agree")
    support = alpha.astype(np.float32) / 255.0
    low_corrected = _normalized_lowpass(corrected, support, sigma)
    low_guide = _normalized_lowpass(guide, support, sigma)
    target = low_corrected + (guide - low_guide)
    allowed = (alpha > 8) & (visibility > 0.05) & ~locked
    amount = np.clip(gate_amount, 0.0, 1.0).astype(np.float32) * allowed
    result = corrected + amount[..., None] * (target - corrected)
    result = np.clip(result, 0.0, 255.0)
    changed = np.any(np.rint(result) != np.rint(corrected), axis=-1)
    return result, changed, amount




def frequency_projection_owner(layers, weights, gate_amount: np.ndarray,
                               locked: np.ndarray, layer_locks):
    """Choose one visible foreground projection owner with layer-local guards.

    ``locked`` contains canvas-wide invariants (user protection, ineligible
    pixels and earlier repairs).  Shadow/material guards are deliberately
    supplied per layer: a protected lower receiver must not block an unrelated
    upper contour at the same canvas coordinate.
    """
    shape = gate_amount.shape
    if locked.shape != shape or len(layers) != len(weights) or len(layers) != len(layer_locks):
        raise ValueError("projection owner inputs do not agree")
    best_weight = np.zeros(shape, np.float32)
    owner = np.full(shape, -1, np.int16)
    for i, layer in enumerate(layers):
        if i == 0:
            continue
        s = layer.region
        if weights[i].shape != layer.alpha.shape or layer_locks[i].shape != layer.alpha.shape:
            raise ValueError("layer projection masks do not agree")
        allowed = ((layer.alpha > 8) & (weights[i] > .05) &
                   ~locked[s] & ~layer_locks[i])
        allowed &= gate_amount[s] > 1e-3
        take = allowed & (weights[i] > best_weight[s])
        best_weight[s][take] = weights[i][take]
        owner[s][take] = i
    return owner, best_weight



def guided_low_frequency_residual(model_guide: np.ndarray, corrected: np.ndarray,
                                  radius: int, epsilon: float) -> np.ndarray:
    """Smooth colour correction without spreading it across model luminance edges.

    The old Gaussian split is algebraically ``model + Gaussian(corrected-model)``.
    A displaced corrected edge therefore survives as a broad, soft ghost.  A
    guided filter instead makes the slowly varying residual follow MODEL_RAW's
    edge geometry, while retaining constant per-material colour corrections.
    """
    if model_guide.shape != corrected.shape or model_guide.ndim != 3 or model_guide.shape[2] != 3:
        raise ValueError("model guide and corrected image must be matching HxWx3 arrays")
    if not isinstance(radius, (int, np.integer)) or radius < 1 or epsilon <= 0:
        raise ValueError("guided residual radius and epsilon must be positive")
    model = model_guide.astype(np.float32)
    residual = corrected.astype(np.float32) - model
    guide = (0.2126 * model[..., 0] + 0.7152 * model[..., 1] +
             0.0722 * model[..., 2]) / 255.0
    size = radius * 2 + 1
    mean_guide = ndi.uniform_filter(guide, size=size, mode="nearest")
    variance = ndi.uniform_filter(guide * guide, size=size, mode="nearest") - mean_guide * mean_guide
    result = np.empty_like(residual)
    for channel in range(3):
        value = residual[..., channel]
        mean_value = ndi.uniform_filter(value, size=size, mode="nearest")
        covariance = ndi.uniform_filter(guide * value, size=size, mode="nearest") - mean_guide * mean_value
        slope = covariance / (variance + epsilon)
        intercept = mean_value - slope * mean_guide
        mean_slope = ndi.uniform_filter(slope, size=size, mode="nearest")
        mean_intercept = ndi.uniform_filter(intercept, size=size, mode="nearest")
        result[..., channel] = mean_slope * guide + mean_intercept
    return result

def frequency_split_target(model_guide: np.ndarray, corrected: np.ndarray,
                           gate_amount: np.ndarray, sigma: float,
                           edge_aware_residual: bool = False, *,
                           calibrated_baseline: np.ndarray | None = None) -> np.ndarray:
    """Construct one composite-domain target for a layered projection.

    Computing the target once after compositing avoids independent per-layer
    low-pass fields disagreeing where translucent layers overlap.  V6.2+ use
    an edge-aware residual so displaced corrected contours cannot become soft
    Gaussian ghosts; the default ``False`` keeps this helper backward-compatible.

    With ``calibrated_baseline``, infer the residual BEFORE pixel borrowing.
    Filtering a post-reuse image is unsafe: mismatched reference contours have
    already entered it, and their low frequencies can survive as stepped or
    doubled bevel bands. The source-free calibrated stack supplies colour,
    MODEL_RAW supplies detail, and corrected remains the ungated output.
    """
    if model_guide.shape != corrected.shape or model_guide.ndim != 3 or model_guide.shape[2] != 3:
        raise ValueError("model guide and corrected composite must be matching HxWx3 arrays")
    if gate_amount.shape != corrected.shape[:2]:
        raise ValueError("gate amount must match the composite canvas")
    colour_field = corrected if calibrated_baseline is None else calibrated_baseline
    if colour_field.shape != model_guide.shape or not np.isfinite(colour_field).all():
        raise ValueError("calibrated baseline must be a finite matching HxWx3 image")
    if edge_aware_residual:
        radius = max(1, int(np.ceil(4.0 * sigma)))
        epsilon = (8.0 / 255.0) ** 2
        residual = guided_low_frequency_residual(model_guide, colour_field, radius, epsilon)
        separated = model_guide.astype(np.float32) + residual
    else:
        low_corrected = ndi.gaussian_filter(colour_field.astype(np.float32), (sigma, sigma, 0), mode="nearest")
        low_model = ndi.gaussian_filter(model_guide.astype(np.float32), (sigma, sigma, 0), mode="nearest")
        separated = low_corrected + (model_guide - low_model)
    amount = np.clip(gate_amount, 0.0, 1.0).astype(np.float32)
    return np.clip(corrected + amount[..., None] * (separated - corrected), 0.0, 255.0)

def frequency_residual_halo(model_guide: np.ndarray, corrected: np.ndarray,
                            edge_core: np.ndarray, sigma: float):
    """Find and replace broad correction halos next to displaced contours.

    The ordinary frequency gate is deliberately narrow.  A source/model
    mismatch can, however, leave a several-pixel low-contrast band just inside
    an otherwise flat material; those pixels sit outside the feather and never
    receive the model-guided target.  This detector starts from the existing
    displaced-edge core, requires a flat MODEL_RAW neighbourhood and a
    non-constant medium-frequency correction residual, then grows a smooth
    material-interior gate.  Constant colour calibration therefore remains a
    no-op and strong model edges are never selected.
    """
    if model_guide.shape != corrected.shape or model_guide.ndim != 3 or model_guide.shape[2] != 3:
        raise ValueError("model guide and corrected image must be matching HxWx3 arrays")
    if edge_core.shape != corrected.shape[:2]:
        raise ValueError("edge core must match the composite canvas")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    shape = edge_core.shape
    if not np.any(edge_core):
        return corrected.astype(np.float32).copy(), np.zeros(shape, np.float32), np.zeros(shape, bool)

    model = model_guide.astype(np.float32)
    corrected_f = corrected.astype(np.float32)
    radius = max(1, int(np.ceil(16.0 * sigma)))
    epsilon = (8.0 / 255.0) ** 2
    halo_target = np.clip(
        model + guided_low_frequency_residual(model, corrected_f, radius, epsilon),
        0.0, 255.0)

    smooth = ndi.gaussian_filter(model, (0.65, 0.65, 0), mode="nearest")
    gx = np.stack([ndi.sobel(smooth[..., c], axis=1, mode="nearest") / 8.0 for c in range(3)], axis=-1)
    gy = np.stack([ndi.sobel(smooth[..., c], axis=0, mode="nearest") / 8.0 for c in range(3)], axis=-1)
    gradient = np.sqrt(np.max(gx * gx + gy * gy, axis=-1))

    residual = corrected_f - model
    outer_sigma = max(2.0, 8.0 * float(sigma))
    medium = np.max(np.abs(
        ndi.gaussian_filter(residual, (1.0, 1.0, 0), mode="nearest")
        - ndi.gaussian_filter(residual, (outer_sigma, outer_sigma, 0), mode="nearest")
    ), axis=-1)
    # A piecewise-constant colour calibration also has a large DoG response
    # near its material boundary, but it is not a halo. A real soft residual
    # band continues to vary inside the flat model material. Requiring that
    # local residual slope prevents a distant edge from selecting an otherwise
    # constant opaque interior merely because the large filter sees outside it.
    smooth_residual = ndi.gaussian_filter(residual, (0.65, 0.65, 0), mode="nearest")
    residual_gx = np.stack([
        ndi.sobel(smooth_residual[..., c], axis=1, mode="nearest") / 8.0
        for c in range(3)
    ], axis=-1)
    residual_gy = np.stack([
        ndi.sobel(smooth_residual[..., c], axis=0, mode="nearest") / 8.0
        for c in range(3)
    ], axis=-1)
    residual_gradient = np.sqrt(np.max(
        residual_gx * residual_gx + residual_gy * residual_gy, axis=-1))
    target_delta = np.max(np.abs(halo_target - corrected_f), axis=-1)
    core_distance = ndi.distance_transform_edt(~edge_core.astype(bool))

    # Seed only broad, unsupported correction structure inside a flat model
    # material and within a bounded neighbourhood of a known displaced edge.
    seed = ((gradient < 8.0) & (target_delta > 1.5) & (medium > 2.0)
            & (residual_gradient > 0.55) & (core_distance <= 20.0))
    seed = ndi.binary_closing(seed, structure=np.ones((3, 3), bool), iterations=2)
    seed = ndi.binary_opening(seed, structure=np.ones((3, 3), bool), iterations=1)
    labels, _ = ndi.label(seed)
    counts = np.bincount(labels.ravel())
    seed &= counts[labels] >= 24
    if not np.any(seed):
        return halo_target, np.zeros(shape, np.float32), seed

    core = ndi.binary_dilation(seed, iterations=1)
    distance = ndi.distance_transform_edt(~core)
    feather = _smoothstep01(1.0 - (distance - 1.0) / 3.0)
    flatness = _smoothstep01((12.0 - gradient) / 6.0)
    proximity = _smoothstep01(1.0 - (core_distance - 12.0) / 8.0)
    amount = (feather * flatness * proximity).astype(np.float32)
    return halo_target, amount, seed
