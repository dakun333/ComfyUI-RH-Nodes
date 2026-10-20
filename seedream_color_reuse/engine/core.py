from __future__ import annotations

from dataclasses import dataclass, field, fields
from numbers import Integral, Real
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .alpha import normalize_alpha
from .receiver_policy import flat_receiver_guard
from .shadows import removed_shadow_guard
from .surfaces import surface_consistency
from .borrowing_policy import contact_shadow_tail_guard, protected_model_target
from .contours import regularize_contours, introduced_contour_detail
from .model_contours import preserve_model_contours
from .frequency_edges import (build_frequency_edge_gate, frequency_projection_owner,
                              frequency_split_target, frequency_residual_halo)
from .reuse_seams import harmonize_reuse_boundaries
from .layer_ownership import (ownership_reuse_anchors, harmonize_nonreuse,
                              receiver_texture_score, harmonize_occlusion_seams)


@dataclass
class Config:
    min_samples: int = 32
    max_samples: int = 40000
    max_color_shift: float = 48.0
    mismatch_threshold: float = 28.0
    structure_threshold: float = 20.0
    guard_radius: int = 3
    edge_radius: int = 5
    edge_max_shift: float = 80.0
    foreground_difference: float = 48.0
    foreground_structure: float = 35.0
    background_guard: int = 8
    background_feather: int = 12
    background_smooth: float = 16.0
    edge_exact_min_weight: float = 0.15
    edge_anchor_tolerance: float = 32.0
    edge_refine_radius: int = 5
    repair_opaque_interiors: int = 0
    alpha_levels: int = 1
    alpha_black: int = 3
    alpha_white: int = 252
    interior_alpha_guard: int = 3
    spatial_continuity: int = 1
    protect_removed_shadows: int = 1
    protect_changed_surfaces: int = 1
    protect_contact_shadow_tails: int = 1
    frequency_soft_protection: int = 1
    regularize_contour_colors: int = 1
    preserve_model_edges: int = 1
    chromatic_contour_ink: int = 1
    frequency_split_edges: int = 1
    frequency_edge_aware_residual: int = 0
    frequency_calibrated_baseline: int = 1
    frequency_residual_halo: int = 0
    protect_layer_ownership: int = 1
    ownership_local_seams: int = 1
    ownership_flat_receivers: int = 1
    reuse_boundary_continuity: int = 1
    ownership_local_max_texture: float = 8.0
    ownership_min_occlusion: float = .10
    ownership_max_texture: float = 2.0
    frequency_edge_sigma: float = 1.5
    frequency_edge_threshold: float = 5.0
    frequency_edge_radius: int = 2
    frequency_edge_feather: int = 4
    color_only: int = 0

    def __post_init__(self):
        integers = {"chromatic_contour_ink", "reuse_boundary_continuity", "color_only", "min_samples", "max_samples", "guard_radius", "edge_radius", "background_guard", "background_feather", "edge_refine_radius", "repair_opaque_interiors", "interior_alpha_guard", "alpha_levels", "alpha_black", "alpha_white", "spatial_continuity", "protect_removed_shadows", "protect_changed_surfaces", "protect_contact_shadow_tails", "frequency_soft_protection", "regularize_contour_colors", "preserve_model_edges", "frequency_split_edges", "frequency_edge_aware_residual", "frequency_calibrated_baseline", "frequency_residual_halo", "protect_layer_ownership", "ownership_local_seams", "ownership_flat_receivers", "frequency_edge_radius", "frequency_edge_feather"}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{item.name} must be a finite number")
            if item.name in integers and not isinstance(value, Integral):
                raise ValueError(f"{item.name} must be an integer")
        if self.chromatic_contour_ink not in (0, 1):
            raise ValueError("chromatic_contour_ink must be 0 or 1")
        if self.reuse_boundary_continuity not in (0, 1):
            raise ValueError("reuse_boundary_continuity must be 0 or 1")
        if self.color_only not in (0, 1):
            raise ValueError("color_only must be 0 or 1")
        if self.color_only and self.repair_opaque_interiors:
            raise ValueError("color_only cannot use legacy alpha repair")
        if self.min_samples < 3 or self.max_samples < self.min_samples:
            raise ValueError("Require 3 <= min_samples <= max_samples")
        if min(self.max_color_shift, self.mismatch_threshold, self.structure_threshold,
               self.edge_max_shift, self.foreground_difference, self.foreground_structure,
               self.background_smooth, self.edge_anchor_tolerance,
               self.ownership_max_texture, self.ownership_local_max_texture) <= 0 or min(self.guard_radius, self.edge_radius,
               self.background_guard, self.background_feather, self.edge_refine_radius, self.interior_alpha_guard) < 1:
            raise ValueError("Thresholds must be positive; radii must be >= 1")
        if self.protect_contact_shadow_tails not in (0, 1):
            raise ValueError("protect_contact_shadow_tails must be 0 or 1")
        if self.frequency_soft_protection not in (0, 1):
            raise ValueError("frequency_soft_protection must be 0 or 1")
        if self.preserve_model_edges not in (0, 1):
            raise ValueError("preserve_model_edges must be 0 or 1")
        if self.frequency_split_edges not in (0, 1):
            raise ValueError("frequency_split_edges must be 0 or 1")
        if self.frequency_edge_aware_residual not in (0, 1):
            raise ValueError("frequency_edge_aware_residual must be 0 or 1")
        if self.ownership_flat_receivers not in (0, 1):
            raise ValueError("ownership_flat_receivers must be 0 or 1")
        if self.ownership_local_seams not in (0, 1):
            raise ValueError("ownership_local_seams must be 0 or 1")
        if self.frequency_calibrated_baseline not in (0, 1):
            raise ValueError("frequency_calibrated_baseline must be 0 or 1")
        if self.frequency_residual_halo not in (0, 1):
            raise ValueError("frequency_residual_halo must be 0 or 1")
        if self.protect_layer_ownership not in (0, 1):
            raise ValueError("protect_layer_ownership must be 0 or 1")
        if not 0 <= self.ownership_min_occlusion < 1:
            raise ValueError("ownership_min_occlusion must be in [0,1)")
        if self.frequency_edge_sigma <= 0 or self.frequency_edge_threshold <= 0:
            raise ValueError("frequency edge sigma and threshold must be positive")
        if self.frequency_edge_radius < 0 or self.frequency_edge_feather < 1:
            raise ValueError("frequency edge radius must be >= 0 and feather >= 1")
        if self.regularize_contour_colors not in (0, 1):
            raise ValueError("regularize_contour_colors must be 0 or 1")
        if self.protect_changed_surfaces not in (0, 1):
            raise ValueError("protect_changed_surfaces must be 0 or 1")
        if self.protect_removed_shadows not in (0, 1):
            raise ValueError("protect_removed_shadows must be 0 or 1")
        if self.spatial_continuity not in (0, 1):
            raise ValueError("spatial_continuity must be 0 or 1")
        if self.alpha_levels not in (0, 1) or not 0 <= self.alpha_black < self.alpha_white <= 255:
            raise ValueError("Require alpha_levels=0/1 and 0 <= alpha_black < alpha_white <=255")
        if self.alpha_levels and self.repair_opaque_interiors:
            raise ValueError("Use alpha levels OR legacy late repair, not both")
        if self.repair_opaque_interiors not in (0, 1):
            raise ValueError("repair_opaque_interiors must be 0 or 1")
        if not 0 < self.edge_exact_min_weight <= 1:
            raise ValueError("edge_exact_min_weight must be in (0, 1]")


@dataclass
class Layer:
    name: str
    rgb: np.ndarray  # straight/unassociated sRGB, float32, 0..255
    alpha: np.ndarray  # immutable uint8 coverage on the working grid
    box: tuple[int, int, int, int]
    metadata: dict = field(default_factory=dict)

    @property
    def region(self):
        x0, y0, x1, y1 = self.box
        return np.s_[y0:y1, x0:x1]

    def copy(self):
        return Layer(self.name, self.rgb.copy(), self.alpha.copy(), self.box,
                     self.metadata.copy())


def composite(layers: list[Layer], shape: tuple[int, int]):
    """Porter-Duff source-over in encoded sRGB, matching common PNG workflows.

    Returns premultiplied RGB in 0..255 and alpha in 0..1. No matte is added.
    """
    rgb = np.zeros((*shape, 3), np.float32)
    alpha = np.zeros(shape, np.float32)
    for layer in layers:
        s = layer.region
        a = layer.alpha.astype(np.float32) / 255
        rgb[s] = layer.rgb * a[..., None] + rgb[s] * (1 - a[..., None])
        alpha[s] = a + alpha[s] * (1 - a)
    return rgb, alpha


def visibility(layers, shape):
    """w_i = alpha_i * product(1-alpha_j), j above i; NOT raw alpha area."""
    t = np.ones(shape, np.float32)
    weights = [None] * len(layers)
    for i in range(len(layers) - 1, -1, -1):
        layer = layers[i]
        s = layer.region
        a = layer.alpha.astype(np.float32) / 255
        weights[i] = t[s] * a
        t[s] *= 1 - a
    return weights


def smooth(rgb, sigma):
    return ndi.gaussian_filter(rgb, (sigma, sigma, 0), mode="nearest")


def structure_error(a, b):
    # Color offsets vanish in high-pass space; missing lines/details do not.
    ah = a - smooth(a, 1.1)
    bh = b - smooth(b, 1.1)
    return np.max(np.abs(ah - bh), axis=-1)


def interior_distance(mask):
    # Explicit zero pad also handles an entirely opaque rectangular layer.
    return ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]


class ColorField:
    """Robust diagonal affine map + supported, color-conditioned residuals.

    Fit ONLY to visible reliable samples, apply to the entire layer (including
    occluded pixels). Unsupported colors tend to identity, not foreground color.
    No spatial copying or histogram matching across unrelated layers.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.gain = np.ones(3, np.float32)
        self.bias = np.zeros(3, np.float32)
        self.tree = None
        self.centers = np.empty((0, 3), np.float32)
        self.residuals = self.centers.copy()
        self.count = 0
        self.fit_error = None

    def fit(self, x, y):
        if len(x) < self.cfg.min_samples:
            return self
        # Deterministic uniform subsampling keeps the result reproducible.
        idx = np.linspace(0, len(x) - 1, min(len(x), self.cfg.max_samples)).astype(int)
        x, y = x[idx].astype(np.float64), y[idx].astype(np.float64)
        keep = np.max(np.abs(y - x), axis=1) < 110
        x, y = x[keep], y[keep]
        if len(x) < self.cfg.min_samples:
            return self
        for c in range(3):
            xc, delta = x[:, c], y[:, c] - x[:, c]
            mu = np.median(xc)
            z = xc - mu
            beta = np.array([0., np.median(delta)])
            for _ in range(7):
                residual = delta - (z * beta[0] + beta[1])
                scale = max(2., 1.4826 * np.median(np.abs(residual - np.median(residual))))
                w = np.minimum(1., 1.5 * scale / np.maximum(np.abs(residual), 1e-6))
                # Ridge prior: strong slope shrinkage in low-variance colors.
                matrix = np.array([[np.sum(w*z*z) + len(x)*100., np.sum(w*z)],
                                   [np.sum(w*z), np.sum(w) + len(x)*.005]])
                beta = np.linalg.solve(matrix, [np.sum(w*z*delta), np.sum(w*delta)])
            self.gain[c] = 1 + np.clip(beta[0], -.3, .3)
            self.bias[c] = np.clip(beta[1] - (self.gain[c]-1)*mu, -60, 60)
        residual = y - (x * self.gain + self.bias)
        error = np.max(np.abs(residual), axis=1)
        cutoff = max(10., min(30., float(np.median(error) + 2.5 * np.median(np.abs(error-np.median(error))))))
        good = error <= cutoff
        x, residual = x[good], residual[good]
        if len(x) < self.cfg.min_samples:
            self.gain[:] = 1; self.bias[:] = 0
            return self
        bins = np.clip((x / 32).astype(int), 0, 7)
        codes = bins[:, 0]*64 + bins[:, 1]*8 + bins[:, 2]
        centers, corrections = [], []
        for code in np.unique(codes):
            selected = codes == code
            if selected.sum() >= 4:
                centers.append(np.mean(x[selected], axis=0))
                corrections.append(np.median(residual[selected], axis=0))
        if not centers:
            centers = [np.mean(x, axis=0)]; corrections = [np.median(residual, axis=0)]
        self.centers = np.array(centers, np.float32)
        self.residuals = np.array(corrections, np.float32)
        self.tree = cKDTree(self.centers)
        self.count = len(x)
        self.fit_error = float(np.median(error[good]))
        return self

    def apply(self, image):
        if self.tree is None:
            return image.copy(), np.zeros(image.shape[:2], np.float32)
        flat = image.reshape(-1, 3)
        result = np.empty_like(flat)
        support = np.empty(len(flat), np.float32)
        for start in range(0, len(flat), 150000):
            x = flat[start:start+150000]
            d, ix = self.tree.query(x, k=min(4, len(self.centers)))
            if d.ndim == 1:
                d, ix = d[:, None], ix[:, None]
            w = np.exp(-d*d / (2*35**2)) / (d + 4)
            local = np.sum(self.residuals[ix] * w[..., None], axis=1) / np.maximum(w.sum(axis=1)[:, None], 1e-20)
            # 0..32 RGB distance: supported; >100: strongly attenuated.
            conf = np.exp(-np.maximum(d[:, 0]-32, 0)**2/(2*35**2))
            shift = x*(self.gain-1) + self.bias + local
            shift = np.clip(shift, -self.cfg.max_color_shift, self.cfg.max_color_shift)
            result[start:start+len(x)] = np.clip(x + shift*conf[:, None], 0, 255)
            support[start:start+len(x)] = conf
        return result.reshape(image.shape), support.reshape(image.shape[:2])


def confidence_maps(predicted, reference, eligible, cfg):
    difference = np.max(np.abs(predicted - reference), axis=-1)
    structural = structure_error(predicted, reference)
    source_detail = np.max(np.abs(reference-smooth(reference, 1.1)), axis=-1)
    model_detail = np.max(np.abs(predicted-smooth(predicted, 1.1)), axis=-1)
    # Low-contrast deleted marks can fall below an absolute color threshold.
    # Detect source-only detail too, and protect the enclosed flat interior.
    unmatched = (source_detail > 3) & (model_detail < source_detail*.3) & eligible
    unmatched = ndi.binary_closing(unmatched, iterations=1)
    holes = ndi.binary_fill_holes(unmatched) & ~unmatched
    labels, _ = ndi.label(holes)
    sizes = np.bincount(labels.ravel())
    # Never let an exterior outline flood-protect an entire otherwise valid
    # object. Large ambiguous regions require the explicit protected mask.
    small = sizes <= max(512, int(eligible.size*.0005))
    small[0] = False
    unmatched |= small[labels]
    unmatched &= eligible
    # A source-only detail is a conflict, not an opportunity to paste it back.
    bad = ((difference > cfg.mismatch_threshold) |
           (structural > cfg.structure_threshold) | unmatched | ~eligible)
    distance = interior_distance(~bad)
    trust = np.clip((distance - 1) / (cfg.guard_radius + 1), 0, 1).astype(np.float32)
    return trust, bad, structural



def background_policy(layer, reference, eligible, protected, foreground, cfg):
    """Never paste a foreground fringe into the editable background.

    Learn only a low-frequency residual from distant background anchors; extend
    it smoothly beneath foreground holes. Detail borrowing is allowed only away
    from BOTH model foreground and source/model disagreement footprints.
    """
    distance = ndi.distance_transform_edt(~foreground) if foreground.any() else np.full(eligible.shape, 1e6)
    delta = reference-layer.rgb
    difference = np.max(np.abs(delta), axis=-1)
    detail = structure_error(layer.rgb, reference)
    # Residuals near an extracted object are treated as its uncertain footprint,
    # even if the per-pixel difference is below the normal color threshold.
    suspicious = (difference > cfg.mismatch_threshold) | (detail > cfg.structure_threshold)
    suspicious &= eligible
    labels, _ = ndi.label(suspicious)
    sizes = np.bincount(labels.ravel())
    substantial = sizes >= max(32, int(eligible.size*.00008))
    substantial[0] = False
    # Small isolated stars/particles far from an extracted object are valid
    # background texture, not missing foreground. Avoid dilating each speck
    # into a large un-reusable island.
    suspicious &= substantial[labels] | (distance < cfg.background_guard*3)
    excluded = foreground | suspicious | protected | ~eligible
    distance_all = ndi.distance_transform_edt(~excluded) if excluded.any() else np.full(eligible.shape, 1e6)
    anchors = eligible & ~protected & (distance_all > cfg.background_guard)
    spatial = np.zeros_like(layer.rgb)
    if anchors.sum() >= cfg.min_samples:
        # Work on a coarser grid for a broad illumination correction. No source
        # texture is transported: only bounded, low-pass RGB residuals.
        step = max(1, int(round(max(eligible.shape)/500)))
        den = ndi.gaussian_filter(anchors.astype(np.float32), max(1., step))[::step, ::step]
        numerator = smooth(np.clip(delta, -32, 32)*anchors[..., None], max(1., step))[::step, ::step]
        supported = den > .25
        if supported.any():
            coarse = numerator/np.maximum(den[..., None], 1e-6)
            nearest = ndi.distance_transform_edt(~supported, return_distances=False, return_indices=True)
            coarse = coarse[nearest[0], nearest[1]]
            coarse = smooth(coarse, cfg.background_smooth/step)
            for c in range(3):
                spatial[..., c] = ndi.zoom(coarse[..., c],
                    (eligible.shape[0]/coarse.shape[0], eligible.shape[1]/coarse.shape[1]), order=1)[:eligible.shape[0], :eligible.shape[1]]
            layer.rgb = np.clip(layer.rgb+spatial, 0, 255)
    # Keep small source-only background particles: they belong to this layer,
    # unlike object fringes. Large disagreements stay protected, not pasted.
    distance_all = ndi.distance_transform_edt(~excluded) if excluded.any() else np.full(eligible.shape, 1e6)
    trust = np.clip((distance_all-cfg.background_guard)/cfg.background_feather, 0, 1).astype(np.float32)
    trust *= eligible & ~protected
    return trust, excluded, {"background_anchor_pixels": int(anchors.sum()),
                             "background_guard_pixels": int((distance <= cfg.background_guard).sum()),
                             "background_spatial_shift_mean": float(np.abs(spatial).mean())}



def layer_boundary(layer, shape, radius=2):
    """Alpha boundary including opaque crop rims, but not canvas borders.

    A rectangular crop can end at alpha=254/255 without transparent pixels in
    its file. Nearest-mode filtering alone misses that seam. Outside a crop
    INSIDE the canvas has zero coverage; outside the canvas has no neighbor.
    """
    a = layer.alpha
    x0, y0, x1, y1 = layer.box
    padded = np.pad(a, radius, mode="edge")
    if x0 > 0: padded[:, :radius] = 0
    if x1 < shape[1]: padded[:, -radius:] = 0
    if y0 > 0: padded[:radius, :] = 0
    if y1 < shape[0]: padded[-radius:, :] = 0
    minimum = ndi.minimum_filter(padded, size=radius*2+1, mode="nearest")[radius:-radius, radius:-radius]
    return (a > 0) & (minimum < 250)

def recover_supported_holes(amount, residual, safe, *, radius=5, max_area=64,
                            endpoint_tolerance=16., prediction_tolerance=12.):
    """Fill only enclosed, small confidence holes with two-sided evidence.

    Every pixel must have a bounded pair of original trusted anchors whose
    correction residuals agree, and must itself agree with their interpolation.
    A missing source-only dot/line fails this test: its residual has no support
    on both sides. No iterative promotion, background propagation or RGB blur.
    """
    anchors = (amount >= 1-1e-7) & safe
    holes = ndi.binary_fill_holes(anchors) & ~anchors
    labels, _ = ndi.label(holes)
    recovered = np.zeros(amount.shape, bool)
    h, w = amount.shape
    for number, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        local = labels[box] == number
        if local.sum() > max_area:
            continue
        points = np.argwhere(local) + [box[0].start, box[1].start]
        if not np.all(safe[points[:, 0], points[:, 1]]):
            continue
        supported = True
        for y, x in points:
            found = False
            for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                ends = []
                for sign in (-1, 1):
                    endpoint = None
                    for step in range(1, radius+1):
                        yy, xx = y+sign*dy*step, x+sign*dx*step
                        if not (0 <= yy < h and 0 <= xx < w) or not safe[yy, xx]:
                            break
                        if anchors[yy, xx]:
                            endpoint = (step, residual[yy, xx])
                            break
                    ends.append(endpoint)
                if any(e is None for e in ends):
                    continue
                (d0, r0), (d1, r1) = ends
                estimate = (r0*d1 + r1*d0)/(d0+d1)
                if (np.max(np.abs(r0-r1)) <= endpoint_tolerance and
                        np.max(np.abs(residual[y, x]-estimate)) <= prediction_tolerance):
                    found = True
                    break
            if not found:
                supported = False
                break
        if supported:
            recovered[points[:, 0], points[:, 1]] = True
    return recovered


def local_anchor_support(rgb, desired, interior, queries, radius, tolerance):
    """Use the whole bounded local anchor set, not a discontinuous nearest one.

    Each accepted query has an actual visible, already matched same-layer
    anchor within the original radius AND original RGB tolerance. Nearby
    protected neighborhoods are excluded by the caller.
    No relaxed color threshold and no new source colors in unsupported regions.
    """
    supported = np.zeros(queries.shape, bool)
    yy, xx = np.nonzero(queries)
    h, w = queries.shape
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dx*dx+dy*dy > radius*radius or not len(yy):
                continue
            ay, ax = yy+dy, xx+dx
            valid = (ay >= 0) & (ay < h) & (ax >= 0) & (ax < w)
            indices = np.flatnonzero(valid)
            if not len(indices):
                continue
            indices = indices[interior[ay[indices], ax[indices]]]
            good = np.max(np.abs(rgb[ay[indices], ax[indices]] -
                                 desired[yy[indices], xx[indices]]), axis=-1) <= tolerance
            hit = indices[good]
            supported[yy[hit], xx[hit]] = True
            remaining = np.ones(len(yy), bool)
            remaining[hit] = False
            yy, xx = yy[remaining], xx[remaining]
    return supported


def coherent_crop_rim(layer, reference, selected, eligible, shape, *, color_step=8.):
    """Do not pepper a smooth crop rim with isolated exact solves.

    On a crop rim with continuous color and coverage,
    either every pixel has foreground anchor support, or retain the existing
    color correction for the whole run. Never broaden source borrowing to
    unsupported pixels just to make the binary decision look continuous.
    """
    result = selected.copy()
    h, w = layer.alpha.shape
    x0, y0, x1, y1 = layer.box
    rims = []
    if y0 > 0: rims.append((np.zeros(w, int), np.arange(w)))
    if y1 < shape[0]: rims.append((np.full(w, h-1), np.arange(w)))
    if x0 > 0: rims.append((np.arange(h), np.zeros(h, int)))
    if x1 < shape[1]: rims.append((np.arange(h), np.full(h, w-1)))
    for yy, xx in rims:
        valid = (layer.alpha[yy, xx] > 0) & eligible[yy, xx]
        # A source-only new mark breaks a run; never bridge it.
        rgb = layer.rgb[yy, xx]
        target = reference[yy, xx]
        breaks = (np.max(np.abs(np.diff(rgb, axis=0)), axis=-1) > color_step)
        breaks |= np.max(np.abs(np.diff(target, axis=0)), axis=-1) > color_step
        breaks |= np.abs(np.diff(layer.alpha[yy, xx].astype(int))) > 16
        starts = np.r_[True, breaks | ~valid[:-1] | ~valid[1:]]
        indices = np.flatnonzero(starts)
        for lo, hi in zip(indices, np.r_[indices[1:], len(yy)]):
            if hi-lo < 3 or not valid[lo:hi].all():
                continue
            choices = result[yy[lo:hi], xx[lo:hi]]
            if choices.any() and not choices.all():
                result[yy[lo:hi], xx[lo:hi]] = False
    return result


def refine_edges(layers, weights, reference, eligible, protected, candidates, stats, cfg,
                 layer_protection=None):
    """Local, fixed-other-layers gamut check plus foreground-supported solve.

    Pixels outside the RGB interval attainable by THIS layer are diagnostics,
    not an excuse to change alpha or paint another layer. Color inversion is
    accepted only close to a currently visible, matched foreground anchor.
    """
    shape = eligible.shape
    current, _ = composite(layers, shape)
    owner = np.full(shape, -1, np.int16)
    largest = np.zeros(shape, np.float32)
    for i, layer in enumerate(layers):
        a = layer.alpha
        edge = layer_boundary(layer, shape)
        selected = edge & (weights[i] > largest[layer.region])
        largest[layer.region][selected] = weights[i][selected]
        owner[layer.region][selected] = i
    gamut_conflicts = np.zeros(shape, bool)
    edge_refined = np.zeros(shape, bool)
    for i, layer in enumerate(layers):
        s = layer.region
        weight = weights[i]
        local_protected = protected[s].copy()
        if layer_protection is not None:
            local_protected |= layer_protection[i] > 0
        region_eligible = eligible[s] & ~local_protected
        edge = (owner[s] == i) & region_eligible & (weight >= .05)
        other = current[s] - layer.rgb*weight[..., None]
        low = other
        high = other + 255*weight[..., None]
        # Test actual encoded targets: tolerance follows 8-bit round-to-nearest.
        in_gamut = np.all((reference[s] >= np.rint(low)) &
                          (reference[s] <= np.rint(high)), axis=-1)
        gamut_conflicts[s] |= edge & ~in_gamut
        stats[i]["edge_fixed_stack_unreachable_pixels"] = int((edge & ~in_gamut).sum())
        stats[i]["edge_exact_candidate_pixels"] = 0
        # Require reliable interior evidence in THIS layer, not just an opaque
        # alpha somewhere underneath an occluder. No target borrowing at anchors.
        interior = (layer.alpha >= 250) & (weight >= .90) & region_eligible
        interior &= np.max(np.abs(current[s]-reference[s]), axis=-1) <= 3
        selected = edge & in_gamut & (weight >= cfg.edge_exact_min_weight)
        if not interior.any() or not selected.any():
            continue
        distance, nearest = ndi.distance_transform_edt(~interior, return_indices=True)
        anchor = layer.rgb[nearest[0], nearest[1]]
        desired = (reference[s]-other)/np.maximum(weight[..., None], .001)
        color_distance = np.max(np.abs(desired-anchor), axis=-1)
        # Nearby supported foreground color prevents background-colored pixels
        # from being pasted into a displaced/overlarge object outline.
        selected &= (distance <= cfg.edge_refine_radius)
        selected &= np.max(np.abs(desired-layer.rgb), axis=-1) <= cfg.edge_max_shift
        nearest_supported = (color_distance <= cfg.edge_anchor_tolerance)
        nearest_supported &= ~local_protected[nearest[0], nearest[1]]
        extra = np.zeros(selected.shape, bool)
        # Layer zero retains its conservative background policy byte-for-byte.
        if cfg.spatial_continuity and i > 0:
            # Do not use a neighborhood that crosses explicit user protection.
            guard = ndi.maximum_filter(local_protected, size=2*cfg.edge_refine_radius+1)
            queries = selected & ~nearest_supported & ~guard
            extra = local_anchor_support(layer.rgb, desired, interior, queries,
                                         cfg.edge_refine_radius, cfg.edge_anchor_tolerance)
        selected &= nearest_supported | extra
        suppressed = 0
        if cfg.spatial_continuity and i > 0:
            coherent = coherent_crop_rim(layer, reference[s], selected,
                                         region_eligible & (weight >= cfg.edge_exact_min_weight), shape)
            suppressed = int((selected & ~coherent).sum())
            selected = coherent
        stats[i]["edge_rim_fragment_suppressed_pixels"] = suppressed
        stats[i]["edge_alternative_anchor_pixels"] = int((extra & selected).sum())
        proposed = np.clip(desired, 0, 255)
        change = (proposed-layer.rgb)*selected[..., None]
        layer.rgb += change
        current[s] += change*weight[..., None]
        candidates[i] |= selected
        edge_refined[s] |= selected
        stats[i]["edge_exact_candidate_pixels"] = int(selected.sum())
    return gamut_conflicts, edge_refined


def repair_interior_coverage(layers, weights, reference, eligible, protected,
                             exact_candidates, exact_requested, stats, cfg):
    """Repair only a 254 -> 255 interior quantization deficit, never geometry.

    Saturated source colors may be outside the fixed-alpha gamut by ONE code
    value. Never change the background/occluded layers to compensate. A narrow,
    explicitly reported alpha repair is the only exception to alpha identity.
    """
    shape = eligible.shape
    repair_map = np.zeros(shape, bool)
    current, _ = composite(layers, shape)
    for i, layer in enumerate(layers):
        stats[i]["interior_alpha_repaired_pixels"] = 0
        if i == 0 or not cfg.repair_opaque_interiors:
            continue
        s = layer.region
        weight = weights[i]
        ref = reference[s]
        old = current[s]
        other = old-layer.rgb*weight[..., None]
        unreachable = np.any((ref < np.rint(other)) | (ref > np.rint(other+255*weight[..., None])), axis=-1)
        # Alpha=254 with transmittance of exactly one: no contribution above.
        owner = (layer.alpha == 254) & (np.abs(weight-254/255) < 1e-7)
        # Include crop edges in the geometry guard. A transparent hole cannot
        # be promoted, and a true soft fringe never reaches this plateau.
        plateau = interior_distance(layer.alpha >= 253) > cfg.interior_alpha_guard
        selected = owner & plateau & eligible[s] & ~protected[s]
        selected &= exact_candidates[i] & unreachable & ~exact_requested[s]
        selected &= np.max(np.abs(old-ref), axis=-1) <= 1.01
        # The numerical deficit must involve a saturated source channel.
        selected &= np.any((ref == 0) | (ref == 255), axis=-1)
        layer.alpha[selected] = 255
        layer.rgb[selected] = ref[selected]
        current[s][selected] = ref[selected]
        repair_map[s] |= selected
        exact_requested[s] |= selected
        stats[i]["interior_alpha_repaired_pixels"] = int(selected.sum())
        stats[i]["exact_reuse_pixels"] += int(selected.sum())
    return repair_map

def _unchanged_composite(layers, shape, cached):
    """Standard recomputes; Fast reuses only an explicitly unchanged snapshot."""
    return composite(layers, shape)


def _map_layer_stage(operation, layers):
    """Run one frozen-evidence stage in layer order.

    The standard backend remains sequential. Fast may replace this dispatcher
    in its private runtime: workers own a layer and never write canvas maps.
    """
    return (operation(i, layer) for i, layer in enumerate(layers))


def calibrate(layers: list[Layer], source_rgba: np.ndarray, cfg: Config | None = None,
              *, protected_mask: np.ndarray | None = None,
              model_guide: np.ndarray | None = None):
    cfg = cfg or Config()
    if not layers or source_rgba.ndim != 3 or source_rgba.shape[2] != 4:
        raise ValueError("Need at least one layer and an HxWx4 reference")
    h, w = source_rgba.shape[:2]
    if source_rgba.dtype != np.uint8 or not h or not w:
        raise ValueError("Reference must be a nonempty uint8 RGBA image")
    if protected_mask is None:
        protected_mask = np.zeros((h, w), bool)
    else:
        if protected_mask.shape != (h, w):
            raise ValueError("Protected mask must have the exact reference size")
        protected_mask = protected_mask.astype(bool)
    for layer in layers:
        x0, y0, x1, y1 = layer.box
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"Invalid canvas box: {layer.name}: {layer.box}")
        if layer.rgb.shape != (y1-y0, x1-x0, 3) or layer.alpha.shape != (y1-y0, x1-x0):
            raise ValueError(f"Array/box mismatch: {layer.name}")
        if layer.alpha.dtype != np.uint8 or not np.isfinite(layer.rgb).all():
            raise ValueError("Alpha must be uint8; RGB must be finite")
        if np.any(layer.rgb < 0) or np.any(layer.rgb > 255):
            raise ValueError("Layer RGB must be in [0, 255]")
    # CLI stamps native-grid normalization in the loader. For direct API
    # arrays, this is the earliest available stage and is reported explicitly.
    prepared = []
    for layer in layers:
        copy = layer.copy()
        copy.alpha, copy.metadata = normalize_alpha(copy.alpha, copy.metadata, cfg, "api_input_grid")
        prepared.append(copy)
    original = prepared
    layers = [l.copy() for l in original]
    shape = (h, w)
    reference = source_rgba[..., :3].astype(np.float32)
    if model_guide is not None:
        if model_guide.shape != (h, w, 3) or not np.isfinite(model_guide).all():
            raise ValueError("model_guide must be a finite HxWx3 image on the reference grid")
        model_guide = np.clip(model_guide.astype(np.float32), 0, 255)
    # Transparent source RGB is undefined, never use it to train or restore.
    valid_source = source_rgba[..., 3] == 255
    before, output_alpha = composite(original, shape)
    frequency_model_guide = before if model_guide is None else model_guide
    weights = visibility(layers, shape)
    eligible = valid_source & (output_alpha >= 1 - 1e-6)
    background_index = 0 if len(layers) > 1 and layers[0].box == (0, 0, w, h) and np.all(layers[0].alpha == 255) else None
    foreground_footprint = np.zeros(shape, bool)
    if background_index is not None:
        for foreground_layer in layers[1:]:
            foreground_footprint[foreground_layer.region] |= foreground_layer.alpha > 2
    # Higher-layer footprints are geometric evidence, never training colors.
    higher_footprints = [None]*len(layers)
    footprint = np.zeros(shape, bool)
    for i in range(len(layers)-1, -1, -1):
        higher_footprints[i] = footprint[layers[i].region].copy()
        footprint[layers[i].region] |= layers[i].alpha > 32
    shadow_guards = []
    shadow_baselines = []
    shadow_core_map = np.zeros(shape, bool)
    shadow_guard_map = np.zeros(shape, np.float32)
    stats = []
    support_maps = []
    def map_initial_layer(i, layer):
        s = layer.region
        target = reference[s]
        core = interior_distance(layer.alpha >= 250) > 1
        fit_mask = (weights[i] >= .97) & core & eligible[s] & ~protected_mask[s]
        fit_mask &= structure_error(layer.rgb, target) < cfg.structure_threshold
        fit_mask &= np.max(np.abs(layer.rgb - target), axis=-1) < 100
        field_model = ColorField(cfg).fit(layer.rgb[fit_mask], target[fit_mask])
        layer.rgb, support = field_model.apply(layer.rgb)
        shadow_guard = np.zeros(layer.alpha.shape, np.float32)
        shadow_core = np.zeros(layer.alpha.shape, bool)
        if cfg.protect_removed_shadows and i > 0:
            visible = (weights[i] >= .97) & core & eligible[s] & ~protected_mask[s]
            shadow_guard, shadow_core = removed_shadow_guard(
                layer.rgb, target, visible, higher_footprints[i],
                scale=max(.5, min(h,w)/2048.))
            shadow_guard *= layer.alpha > 0
            if shadow_core.any():
                # Provisional fitting is detection-only. Learn material color
                # from unshadowed evidence and apply that map to the receiver.
                fit_mask &= shadow_guard == 0
                field_model = ColorField(cfg).fit(original[i].rgb[fit_mask], target[fit_mask])
                layer.rgb, support = field_model.apply(original[i].rgb)
        visible_guard = shadow_guard * (weights[i] > 1e-7)
        area = int(np.count_nonzero(layer.alpha))
        visible_mass = float(weights[i].sum())
        layer_stats = {"name": layer.name,
                      "removed_shadow_core_pixels": int(shadow_core.sum()),
                      "removed_shadow_guard_pixels": int((visible_guard > 0).sum()),
                      "alpha_normalization": layer.metadata.get("alpha_normalization"), "alpha_area": area,
                      "visible_alpha_mass": visible_mass,
                      "visible_fraction": visible_mass / max(1, float((layer.alpha / 255).sum())),
                      "candidate_samples": int(fit_mask.sum()), "fit_samples": field_model.count,
                      "fit_median_error": field_model.fit_error,
                      "gain": field_model.gain.tolist(), "bias": field_model.bias.tolist(),
                      "status": "calibrated" if field_model.count else "insufficient_evidence_identity",
                      "unsupported_color_pixels": int(((support < .2) & (layer.alpha > 0)).sum())}
        return support, shadow_guard, shadow_core, layer_stats

    for i, (support, shadow_guard, shadow_core, layer_stats) in enumerate(
            _map_layer_stage(map_initial_layer, layers)):
        layer = layers[i]
        s = layer.region
        shadow_guards.append(shadow_guard)
        shadow_baselines.append(np.rint(layer.rgb[shadow_guard >= 1-1e-7]).copy())
        visible_guard = shadow_guard * (weights[i] > 1e-7)
        shadow_guard_map[s] = np.maximum(shadow_guard_map[s], visible_guard)
        shadow_core_map[s] |= shadow_core
        support_maps.append(support)
        stats.append(layer_stats)
    if cfg.color_only:
        # Stop before ALL spatial residuals, reference borrowing, inverse-alpha
        # solves and post-reuse ownership/contour operations. The same learned
        # RGB map applies to visible and occluded colors within each layer.
        from .color_only import finish_color_only
        return finish_color_only(layers, original, source_rgba, cfg, before,
                                 output_alpha, eligible, protected_mask, weights,
                                 stats, support_maps, shadow_guards,
                                 shadow_core_map, shadow_guard_map)
    surface_guards = []
    surface_baselines = []
    surface_guard_map = np.zeros(shape, np.float32)
    def map_surface_layer(i, layer):
        s = layer.region
        guard = np.zeros(layer.alpha.shape, np.float32)
        if cfg.protect_changed_surfaces and i > 0:
            visible = (weights[i] >= .97) & (layer.alpha >= 250) & eligible[s]
            forbidden = protected_mask[s] | (shadow_guards[i] > 0)
            guard, correction = surface_consistency(layer.rgb, reference[s], visible, forbidden,
                                                   scale=max(.5, min(h,w)/2048.),
                                                   coverage=(layer.alpha > 0) & (weights[i] > 1e-7) & eligible[s])
            layer.rgb = np.clip(layer.rgb + correction, 0, 255)
        return guard

    for i, guard in enumerate(_map_layer_stage(map_surface_layer, layers)):
        layer = layers[i]
        s = layer.region
        surface_guards.append(guard)
        surface_baselines.append(np.rint(layer.rgb[guard >= 1-1e-7]).copy())
        surface_guard_map[s] = np.maximum(surface_guard_map[s], guard)
        stats[i]["changed_surface_guard_pixels"] = int((guard > 0).sum())
    # A reference-only object can be absent from upper layer ownership. Use
    # the already established material-conflict core as contact evidence, not
    # a blanket detector of every dark difference on the receiver.
    contact_guards = []
    contact_baselines = []
    contact_guard_map = np.zeros(shape, np.float32)
    def map_contact_layer(i, layer):
        s = layer.region
        guard = np.zeros(layer.alpha.shape, np.float32)
        if cfg.protect_contact_shadow_tails and i > 0:
            solid = ndi.minimum_filter(layer.alpha, size=3, mode="constant", cval=0) == 255
            visible = solid & (weights[i] >= .97) & eligible[s] & ~protected_mask[s]
            guard = contact_shadow_tail_guard(
                layer.rgb, reference[s], visible,
                surface_guards[i] >= 1-1e-7, scale=max(.5, min(h,w)/2048.))
        return guard

    for i, guard in enumerate(_map_layer_stage(map_contact_layer, layers)):
        layer = layers[i]
        s = layer.region
        contact_guards.append(guard)
        contact_baselines.append(np.rint(layer.rgb[guard >= 1-1e-7]).copy())
        contact_guard_map[s] = np.maximum(contact_guard_map[s], guard)
        stats[i]["contact_shadow_guard_pixels"] = int((guard > 0).sum())
    # Receiver-only ownership exclusion. Do not merge it into a canvas guard:
    # a protected shirt must not prevent its independent upper lettering from
    # receiving an edge correction at the same coordinate.
    receiver_guards = []
    receiver_texture_scores = {}
    receiver_baselines = []
    receiver_guard_map = np.zeros(shape, np.float32)
    def map_receiver_layer(i, layer):
        texture_score = None
        s = layer.region
        guard = np.zeros(layer.alpha.shape, np.float32)
        if cfg.protect_layer_ownership and cfg.ownership_flat_receivers and i > 0:
            locked = (protected_mask[s] | (shadow_guards[i] > 0) |
                      (surface_guards[i] > 0) | (contact_guards[i] > 0))
            # Existing smooth whole-layer/local seam policies already handle
            # low-texture receivers. This is only their missing local-material
            # fallback, not a competing second calibration of those receivers.
            occluded_solid = (layer.alpha == 255) & (weights[i] <= .5) & eligible[s] & ~locked
            if int(occluded_solid.sum()) >= cfg.min_samples:
                texture_score = receiver_texture_score(layer.rgb, layer.alpha)
                if texture_score > cfg.ownership_local_max_texture:
                    guard = flat_receiver_guard(
                        layer.rgb, layer.alpha, weights[i], eligible[s], locked,
                        radius=max(8., 2. * cfg.edge_refine_radius), min_samples=cfg.min_samples)
        return guard, texture_score

    for i, (guard, texture_score) in enumerate(_map_layer_stage(map_receiver_layer, layers)):
        layer = layers[i]
        s = layer.region
        if texture_score is not None:
            receiver_texture_scores[i] = texture_score
        receiver_guards.append(guard)
        receiver_baselines.append(np.rint(layer.rgb[guard >= 1-1e-7]).copy())
        receiver_guard_map[s] = np.maximum(receiver_guard_map[s], guard)
        stats[i]["flat_receiver_guard_pixels"] = int((guard > 0).sum())
    bg_trust = None
    bg_excluded = np.zeros(shape, bool)
    if background_index is not None:
        bg_trust, bg_excluded, bg_stats = background_policy(layers[0], reference, eligible,
                                                          protected_mask, foreground_footprint, cfg)
        stats[0].update(bg_stats)
    # Per-layer guards only exclude source borrowing at visible receivers;
    # background fitting/correction remains exactly the previous policy.
    borrowing_protected = protected_mask | (shadow_guard_map > 0) | (surface_guard_map > 0) | (contact_guard_map > 0)
    contour_guides = [layer.rgb.copy() if (cfg.regularize_contour_colors or cfg.preserve_model_edges or cfg.frequency_split_edges or cfg.protect_layer_ownership) and i > 0 else None
                      for i, layer in enumerate(layers)]
    mapped, _ = composite(layers, shape)
    frequency_edge_gate = np.zeros(shape, np.float32)
    frequency_edge_parts = {
        "model_edge": np.zeros(shape, bool),
        "reference_edge": np.zeros(shape, bool),
        "alpha_edge": np.zeros(shape, bool),
        "displaced_pair": np.zeros(shape, bool),
        "core": np.zeros(shape, bool),
    }
    if cfg.frequency_split_edges:
        frequency_edge_gate, frequency_edge_parts = build_frequency_edge_gate(
            mapped, reference, layers, weights, eligible, background_index, cfg)
    # This gate is deliberately independent from protected_mask: it does not
    # alter model fitting, matte repair or the established background policy.
    conservative_trust, conservative_bad, structural = confidence_maps(mapped, reference, eligible & ~protected_mask, cfg)
    # Foreground interiors do not need the background's wide safety band. A
    # small antialiasing/textural disagreement should not veto a whole object.
    difference = np.max(np.abs(mapped-reference), axis=-1)
    fg_bad = (difference > cfg.foreground_difference) | (structural > cfg.foreground_structure) | ~eligible | protected_mask
    fg_trust = np.clip(interior_distance(~fg_bad)-1, 0, 1).astype(np.float32)
    trust = conservative_trust.copy()
    bad = conservative_bad.copy()
    if background_index is not None:
        trust = bg_trust.copy()
        bad = bg_excluded.copy()
    for i, layer in enumerate(layers):
        if i == background_index or (background_index is None and i == 0):
            continue
        selected = weights[i] >= .5
        trust[layer.region][selected] = fg_trust[layer.region][selected]
        bad[layer.region][selected] = fg_bad[layer.region][selected]
    exact_requested = np.zeros(shape, bool)
    # Near-opaque model interiors are frequently alpha=254, not 255. Direct
    # copying would retain background leakage. Solve the compositing equation
    # instead; only w=1 is literal RGB copying. There is at most one w>=.97 owner.
    exact_candidates = []
    supported_hole_map = np.zeros(shape, bool)
    reuse_amounts = []
    for i, layer in enumerate(layers):
        s = layer.region
        is_foreground = i != background_index and not (background_index is None and i == 0)
        dominant = (weights[i] >= (.90 if is_foreground else .97)) & (layer.alpha >= (230 if is_foreground else 250))
        core_distance = interior_distance(dominant)
        layer_trust = fg_trust[s] if is_foreground else (bg_trust[s] if i == background_index else conservative_trust[s])
        amount = layer_trust * (np.clip(core_distance-1, 0, 1) if is_foreground else np.clip((core_distance - 1)/2, 0, 1))
        amount *= dominant
        delta = (reference[s] - mapped[s]) / np.maximum(weights[i][..., None], .01)
        recovered = np.zeros(amount.shape, bool)
        if cfg.spatial_continuity and is_foreground:
            safe = (weights[i] >= 1-1e-7) & eligible[s] & ~protected_mask[s]
            # Two-sided colour agreement does not establish geometric agreement.
            # A tiny recovered island inside a displaced contour used to be
            # globally locked against the later frequency repair, leaving an
            # exact-reference tooth surrounded by the restored model profile.
            if cfg.frequency_split_edges and cfg.frequency_calibrated_baseline:
                safe &= frequency_edge_gate[s] <= 1e-3
            recovered = recover_supported_holes(amount, delta, safe)
            amount[recovered] = 1
            trust[s][recovered] = 1
            bad[s][recovered] = False
        amount *= (1-shadow_guards[i]) * (1-surface_guards[i]) * (1-contact_guards[i]) * (1-receiver_guards[i])
        supported_hole_map[s] |= recovered & (amount >= 1-1e-7)
        trust[s] *= (1-shadow_guards[i]*(weights[i] > 1e-7)) * (1-surface_guards[i]) * (1-contact_guards[i])
        trust[s][frequency_edge_gate[s] > 1e-3] = 0
        bad[s] |= contact_guards[i] >= 1-1e-7
        bad[s] |= surface_guards[i] >= 1-1e-7
        bad[s] |= (shadow_guards[i] >= 1-1e-7) & (weights[i] > 1e-7)
        stats[i]["supported_confidence_hole_pixels"] = int((recovered & (shadow_guards[i] == 0) & (surface_guards[i] == 0) & (contact_guards[i] == 0) & (receiver_guards[i] == 0)).sum())
        reuse_amounts.append(amount.astype(np.float32).copy())
        layer.rgb = np.clip(layer.rgb + amount[..., None] * delta, 0, 255)
        exact_here = (amount >= 1-1e-7) & eligible[s]
        exact_candidates.append(exact_here)
        stats[i]["exact_reuse_pixels"] = 0
    # Edge solve observes the actual underlying stack, not source RGB as a
    # putative foreground. Update just one owner per pixel to avoid double edits.
    current, _ = composite(layers, shape)
    best_weight = np.zeros(shape, np.float32)
    edge_owner = np.full(shape, -1, np.int16)
    for i, layer in enumerate(layers):
        a = layer.alpha
        edge = (a > 0) & (a < 255) & (ndi.minimum_filter(a, size=5, mode="nearest") < 250)
        eligible_edge = edge & (weights[i] > best_weight[layer.region])
        best_weight[layer.region][eligible_edge] = weights[i][eligible_edge]
        edge_owner[layer.region][eligible_edge] = i
    # Invalid/missing content is never used for an inverse-alpha solve.
    edge_trust = np.clip(1 - np.maximum(structural-cfg.structure_threshold*.5, 0) /
                         (cfg.structure_threshold*.5), 0, 1)
    edge_trust *= eligible & ~ndi.binary_dilation(bad, iterations=1)
    edge_decontaminated_map = np.zeros(shape, bool)
    for i, layer in enumerate(layers):
        s = layer.region
        a = layer.alpha.astype(np.float32) / 255
        weight = weights[i]
        edge_eligible = ((edge_owner[s] == i) & eligible[s] & ~borrowing_protected[s] &
                         (receiver_guards[i] == 0))
        selected = edge_eligible & (edge_trust[s] > 0)
        solve_trust = edge_trust[s].copy()
        # Extend a nearby opaque color only when this improves the observed
        # composite. This preserves intentional black outlines and shadows.
        opaque = layer.alpha >= 245
        decontaminated = np.zeros(a.shape, bool)
        if opaque.any() and edge_eligible.any():
            distance, nearest = ndi.distance_transform_edt(~opaque, return_indices=True)
            near_color = layer.rgb[nearest[0], nearest[1]]
            delta = near_color - layer.rgb
            candidate = current[s] + delta * weight[..., None]
            old_err = np.mean((current[s]-reference[s])**2, axis=-1)
            new_err = np.mean((candidate-reference[s])**2, axis=-1)
            # A severe matte fringe itself raises the conflict detector. It
            # can be rescued only by an independently validated nearby visible
            # foreground anchor and a candidate almost matching the reference.
            anchor_visible = weights[i][nearest[0], nearest[1]] >= .97
            anchor_error = np.max(np.abs(layer.rgb-reference[s]), axis=-1)[nearest[0], nearest[1]]
            rescue = edge_eligible & anchor_visible & (anchor_error < 12)
            rescue &= (new_err < 36) & (new_err < old_err*.4)
            take = (selected | rescue) & (a < .6) & (distance <= cfg.edge_radius)
            take &= (new_err < old_err*.8) & (old_err > .05)
            solve_trust[take & rescue] = 1
            selected |= take
            # Full extension can legitimately exceed the inverse-solve cap.
            change = delta * (take * solve_trust)[..., None]
            layer.rgb += change
            current[s] += change * weight[..., None]
            decontaminated = take
            edge_decontaminated_map[s] |= take
            selected |= take
        # Tikhonov alpha-aware residual solve. Never divide by tiny alpha.
        residual = reference[s] - current[s]
        factor = weight / (weight*weight + .08**2)
        correction = np.clip(residual * factor[..., None], -cfg.edge_max_shift, cfg.edge_max_shift)
        correction *= (selected * solve_trust)[..., None]
        updated = np.clip(layer.rgb + correction, 0, 255)
        current[s] += (updated-layer.rgb) * weight[..., None]
        layer.rgb = updated
        stats[i]["edge_solved_pixels"] = int(selected.sum())
        stats[i]["edge_decontaminated_pixels"] = int(decontaminated.sum())
    gamut_conflicts, edge_refined = refine_edges(layers, weights, reference, eligible,
                                                borrowing_protected, exact_candidates, stats, cfg, receiver_guards)
    # Quantize every asset BEFORE measurement. This is what users actually get.
    for layer in layers:
        layer.rgb = np.rint(np.clip(layer.rgb, 0, 255)).astype(np.float32)
    # Integer-aware final projection: solve against the other *quantized*
    # layers. Some colors are outside the fixed-alpha gamut; do not claim them.
    exact_owner = np.full(shape, -1, np.int16)
    quantized, _ = composite(layers, shape)
    for i, layer in enumerate(layers):
        s = layer.region
        selected = exact_candidates[i]
        weight = weights[i]
        delta = (reference[s]-quantized[s]) / np.maximum(weight[..., None], .01)
        proposed = np.rint(np.clip(layer.rgb + delta, 0, 255))
        predicted = quantized[s] + (proposed-layer.rgb)*weight[..., None]
        feasible = selected & np.all(np.rint(predicted) == reference[s], axis=-1)
        change = (proposed-layer.rgb)*feasible[..., None]
        layer.rgb += change
        quantized[s] += change*weight[..., None]
        exact_requested[s] |= feasible
        exact_owner[s][feasible] = i
        stats[i]["exact_reuse_pixels"] = int(feasible.sum())
    alpha_repaired = repair_interior_coverage(layers, weights, reference, eligible, borrowing_protected,
                                              exact_candidates, exact_requested, stats, cfg)
    # Operate AFTER integer projection, so no solver can undo the repair or
    # compensate by painting the underlying background. Only color residuals
    # change; alpha, hidden pixels and all prior protected cores are locked.
    contour_changed = np.zeros(shape, bool)
    artifact_strength = np.zeros(shape, np.float32)
    if cfg.regularize_contour_colors:
        legacy_after, _ = composite(layers, shape)
        artifact_strength = introduced_contour_detail(legacy_after, mapped, reference)
    def map_contour_layer(i, layer):
        changed = np.zeros(layer.alpha.shape, bool)
        if cfg.regularize_contour_colors and i > 0:
            locked = (borrowing_protected[layer.region] | alpha_repaired[layer.region] |
                      ~eligible[layer.region] | (receiver_guards[i] > 0))
            adjusted, changed, _ = regularize_contours(contour_guides[i], layer.rgb,
                                                       layer.alpha, weights[i], locked,
                                                       reuse_mask=exact_requested[layer.region],
                                                       artifact_strength=artifact_strength[layer.region])
            layer.rgb = np.rint(adjusted).astype(np.float32)
        return changed

    for i, changed in enumerate(_map_layer_stage(map_contour_layer, layers)):
        layer = layers[i]
        stats[i]["contour_regularized_pixels"] = int(changed.sum())
        contour_changed[layer.region] |= changed
    recolored, _ = composite(layers, shape)
    lost_exact = exact_requested & contour_changed & ~np.all(np.rint(recolored) == reference, axis=-1)
    for i, layer in enumerate(layers):
        stats[i]["exact_reuse_pixels"] -= int((lost_exact & (exact_owner == i)).sum())
    exact_requested[lost_exact] = False
    model_contour_changed = np.zeros(shape, bool)
    model_contour_weights = []
    # Freeze the V5.4 reuse decisions for all layers; do not update evidence
    # while visiting the stack. Full edge segments replace isolated decisions.
    model_contour_reuse = exact_requested.copy()
    chromatic_contour_candidates = [{} for _ in layers]
    chromatic_contour_enabled = bool(cfg.chromatic_contour_ink and
                                    cfg.preserve_model_edges and cfg.spatial_continuity)
    def map_model_contour_layer(i, layer):
        changed = np.zeros(layer.alpha.shape, bool)
        amount = np.zeros(layer.alpha.shape, np.float32)
        if cfg.preserve_model_edges and i > 0:
            locked = (borrowing_protected[layer.region] | alpha_repaired[layer.region] |
                      ~eligible[layer.region] | (receiver_guards[i] > 0))
            adjusted, changed, amount = preserve_model_contours(
                contour_guides[i], layer.rgb, layer.alpha, weights[i], locked,
                model_contour_reuse[layer.region],
                continuation=chromatic_contour_candidates[i] if chromatic_contour_enabled else None)
            layer.rgb = np.rint(adjusted).astype(np.float32)
        return changed, amount

    for i, (changed, amount) in enumerate(_map_layer_stage(map_model_contour_layer, layers)):
        layer = layers[i]
        model_contour_weights.append(amount)
        stats[i]["model_contour_changed_pixels"] = int(changed.sum())
        model_contour_changed[layer.region] |= changed
    model_contour_snapshot = composite(layers, shape)
    recolored, _ = model_contour_snapshot
    lost_exact = exact_requested & model_contour_changed & ~np.all(np.rint(recolored) == reference, axis=-1)
    for i, layer in enumerate(layers):
        stats[i]["exact_reuse_pixels"] -= int((lost_exact & (exact_owner == i)).sum())
    exact_requested[lost_exact] = False
    frequency_edge_changed = np.zeros(shape, bool)
    frequency_halo_amount = np.zeros(shape, np.float32)
    frequency_halo_seed = np.zeros(shape, bool)
    frequency_edge_layer_weights = [np.zeros(layer.alpha.shape, np.float32) for layer in layers]
    if cfg.frequency_split_edges:
        # Build one desired composite so translucent/overlapping layers do not
        # each invent a different low-frequency field. Then change exactly one
        # highest-visibility foreground owner at every editable canvas pixel.
        frequency_before, _ = _unchanged_composite(layers, shape, model_contour_snapshot)
        frequency_target = frequency_split_target(
            frequency_model_guide, frequency_before, frequency_edge_gate,
            cfg.frequency_edge_sigma, bool(cfg.frequency_edge_aware_residual),
            calibrated_baseline=mapped if cfg.frequency_calibrated_baseline else None)
        if cfg.frequency_calibrated_baseline:
            # Source-free colour is reliable for solid material, not for a
            # soft matte that earlier inverse-alpha solves have decontaminated.
            # Preserve the established soft-edge target in that latter case.
            solid_foreground = np.zeros(shape, bool)
            for i, (layer, weight) in enumerate(zip(layers, weights)):
                if i == 0:
                    continue
                solid = ndi.minimum_filter(layer.alpha, size=3, mode="constant", cval=0) == 255
                solid_foreground[layer.region] |= solid & (weight >= .97)
            legacy_target = frequency_split_target(
                frequency_model_guide, frequency_before, frequency_edge_gate,
                cfg.frequency_edge_sigma, bool(cfg.frequency_edge_aware_residual))
            frequency_target = np.where(solid_foreground[..., None], frequency_target, legacy_target)
        frequency_projection_gate = frequency_edge_gate.copy()
        if cfg.frequency_residual_halo and cfg.frequency_edge_aware_residual:
            halo_target, frequency_halo_amount, frequency_halo_seed = frequency_residual_halo(
                frequency_model_guide, frequency_before, frequency_edge_parts["core"],
                cfg.frequency_edge_sigma)
            # A broad residual halo is a material-interior repair, not an
            # alternative treatment for soft alpha. Restrict it to canvas
            # pixels owned by a foreground layer whose complete 3x3 alpha
            # neighbourhood is fully opaque. This still reaches several
            # pixels beyond the narrow displaced-contour gate, while keeping
            # feathered/near-opaque edges on the existing exact-reuse path.
            opaque_foreground_interior = np.zeros(shape, bool)
            for i, (layer, weight) in enumerate(zip(layers, weights)):
                if i == background_index or (background_index is None and i == 0):
                    continue
                fully_opaque = ndi.minimum_filter(
                    layer.alpha, size=3, mode="constant", cval=0) == 255
                opaque_foreground_interior[layer.region] |= fully_opaque & (weight > .005)
            frequency_halo_amount *= eligible & opaque_foreground_interior
            frequency_halo_seed &= opaque_foreground_interior
            frequency_target += frequency_halo_amount[..., None] * (halo_target - frequency_target)
            frequency_projection_gate = np.maximum(frequency_projection_gate, frequency_halo_amount)
        # Protection is layer-local here.  A removed-shadow/material guard on
        # an underlying receiver must freeze that receiver's RGB, but it must
        # not punch a checkerboard hole through an unrelated foreground
        # contour crossing the same canvas coordinates.  The frequency target
        # is model-derived (it never borrows reference RGB), so projecting it
        # into an unguarded upper owner preserves the protected lower layer and
        # keeps the visible contour coherent.
        frequency_locked = (protected_mask | alpha_repaired | edge_decontaminated_map |
                            supported_hole_map | ~eligible)
        frequency_layer_locks = []
        frequency_surface_amounts = []
        for layer, weight, shadow_guard, surface_guard, contact_guard, receiver_guard in zip(
                layers, weights, shadow_guards, surface_guards, contact_guards, receiver_guards):
            # Only source-free, solid material may use continuous surface
            # feather repair. Soft mattes stay on their established path.
            soft_repair = np.zeros(layer.alpha.shape, bool)
            if cfg.frequency_soft_protection and cfg.frequency_calibrated_baseline:
                soft_repair = (ndi.minimum_filter(layer.alpha, size=3,
                    mode="constant", cval=0) == 255) & (weight >= .97)
            surface_lock = (surface_guard > 0) & (~soft_repair | (surface_guard >= 1-1e-7))
            frequency_layer_locks.append((shadow_guard > 0) | surface_lock | (contact_guard > 0) | (receiver_guard > 0))
            frequency_surface_amounts.append(np.where(soft_repair, surface_guard, 0))
        frequency_owner, _ = frequency_projection_owner(
            layers, weights, frequency_projection_gate, frequency_locked, frequency_layer_locks)
        projected = frequency_before.copy()
        for i, layer in enumerate(layers):
            changed = np.zeros(layer.alpha.shape, bool)
            if i > 0:
                s = layer.region
                selected = frequency_owner[s] == i
                # frequency_target already includes the frequency gate. Apply
                # the surface weight once; do not multiply the gate twice.
                local_target = frequency_target[s]
                if np.any(frequency_surface_amounts[i] > 0):
                    local_target = protected_model_target(
                        frequency_before[s], local_target,
                        np.ones(layer.alpha.shape, np.float32), frequency_surface_amounts[i])
                delta = (local_target - projected[s]) / np.maximum(weights[i][..., None], .01)
                proposed = np.rint(np.clip(layer.rgb + delta, 0, 255))
                changed = selected & np.any(proposed != layer.rgb, axis=-1)
                change = (proposed - layer.rgb) * selected[..., None]
                layer.rgb += change
                projected[s] += change * weights[i][..., None]
                frequency_edge_layer_weights[i][selected] = (frequency_projection_gate[s] * (1-frequency_surface_amounts[i]))[selected]
            stats[i]["frequency_edge_changed_pixels"] = int(changed.sum())
            frequency_edge_changed[layer.region] |= changed
    else:
        for stats_item in stats:
            stats_item["frequency_edge_changed_pixels"] = 0
    recolored, _ = composite(layers, shape)
    lost_frequency_exact = exact_requested & frequency_edge_changed & ~np.all(np.rint(recolored) == reference, axis=-1)
    for i, layer in enumerate(layers):
        stats[i]["exact_reuse_pixels"] -= int((lost_frequency_exact & (exact_owner == i)).sum())
    exact_requested[lost_frequency_exact] = False
    # A flattened reference has no layer ownership labels. Per-pixel exact or
    # frequency projection can therefore leave the silhouette of an upper
    # layer in an underlying asset: visible receiver pixels are corrected,
    # while hidden receiver pixels keep the colour-fit baseline. Replace that
    # discontinuity with a low-frequency residual inferred from reliable,
    # structurally quiet visible samples of the receiver itself.
    ownership_changed = np.zeros(shape, bool)
    ownership_amount = np.zeros(shape, np.float32)
    ownership_invalidated_exact = np.zeros(shape, bool)
    ownership_layer_amounts = [np.zeros(layer.alpha.shape, np.float32) for layer in layers]
    ownership_radius = max(cfg.edge_refine_radius, 2 * cfg.guard_radius)
    for i, layer in enumerate(layers):
        changed = np.zeros(layer.alpha.shape, bool)
        amount = np.zeros(layer.alpha.shape, np.float32)
        anchors = 0
        occlusion_fraction = 0.0
        texture_score = None
        texture_gate_passed = False
        local_seam_repair = False
        if cfg.protect_layer_ownership and i > 0:
            s = layer.region
            locked = (protected_mask[s] | alpha_repaired[s] | ~eligible[s] |
                      (shadow_guards[i] > 0) | (surface_guards[i] > 0) | (contact_guards[i] > 0) |
                      (receiver_guards[i] > 0))
            safe_reuse, covered, occlusion_fraction = ownership_reuse_anchors(
                layer.alpha, weights[i],
                exact_requested[s] & (exact_owner[s] == i), eligible[s],
                structural[s], locked, guard_radius=ownership_radius,
                structure_threshold=cfg.structure_threshold * .6,
                min_occlusion_fraction=cfg.ownership_min_occlusion)
            texture_score = receiver_texture_scores.get(i)
            if texture_score is None:
                texture_score = receiver_texture_score(contour_guides[i], layer.alpha)
            texture_gate_passed = texture_score <= cfg.ownership_max_texture
            anchors = int(safe_reuse.sum())
            if texture_gate_passed and anchors >= cfg.min_samples:
                # Pass 2 is learned from the pixels that survived exact reuse.
                # The pass-1 guide remains the sole source of material detail.
                second_field = ColorField(cfg).fit(
                    contour_guides[i][safe_reuse], layer.rgb[safe_reuse])
                calibrated_base, _ = second_field.apply(contour_guides[i])
                adjusted, changed, amount, anchors = harmonize_nonreuse(
                    contour_guides[i], calibrated_base, layer.rgb, layer.alpha,
                    safe_reuse, covered, eligible[s], locked,
                    max_color_shift=cfg.max_color_shift,
                    seam_sigma=max(2., ownership_radius / 2),
                    min_anchor_pixels=cfg.min_samples)
                layer.rgb = adjusted
                ownership_changed[s] |= changed
                ownership_amount[s] = np.maximum(ownership_amount[s], amount)
                ownership_layer_amounts[i] = amount
            elif cfg.ownership_local_seams and texture_score <= cfg.ownership_local_max_texture:
                # The whole-layer fit is intentionally gated by texture and
                # occlusion fraction. Neither gate justifies leaving a visible
                # upper silhouette in a textured or lightly occluded asset.
                local_safe, local_covered, _ = ownership_reuse_anchors(
                    layer.alpha, weights[i],
                    exact_requested[s] & (exact_owner[s] == i), eligible[s],
                    structural[s], locked, guard_radius=ownership_radius,
                    structure_threshold=cfg.structure_threshold * .6,
                    min_occlusion_fraction=0.)
                adjusted, changed, amount, local_anchors = harmonize_occlusion_seams(
                    contour_guides[i], layer.rgb, layer.alpha,
                    local_safe, local_covered, eligible[s], locked,
                    max_color_shift=cfg.max_color_shift,
                    seam_sigma=max(12., 3. * ownership_radius),
                    min_anchor_pixels=cfg.min_samples)
                if local_anchors:
                    anchors = local_anchors
                    local_seam_repair = True
                    layer.rgb = adjusted
                    ownership_changed[s] |= changed
                    ownership_amount[s] = np.maximum(ownership_amount[s], amount)
                    ownership_layer_amounts[i] = amount
        stats[i]["ownership_local_seam_repair"] = local_seam_repair
        stats[i]["ownership_regularized_pixels"] = int(changed.sum())
        stats[i]["ownership_guard_pixels"] = int((amount > 0).sum())
        stats[i]["ownership_anchor_pixels"] = anchors
        stats[i]["ownership_occlusion_fraction"] = float(occlusion_fraction)
        stats[i]["ownership_texture_score_p75"] = texture_score
        stats[i]["ownership_texture_gate_passed"] = bool(texture_gate_passed)
    ownership_snapshot = composite(layers, shape)
    recolored, _ = ownership_snapshot
    lost_ownership_exact = exact_requested & ownership_changed & ~np.all(
        np.rint(recolored) == reference, axis=-1)
    ownership_invalidated_exact |= lost_ownership_exact
    for i, layer in enumerate(layers):
        stats[i]["exact_reuse_pixels"] -= int((lost_ownership_exact & (exact_owner == i)).sum())
        stats[i]["ownership_invalidated_exact_pixels"] = int(
            (lost_ownership_exact & (exact_owner == i)).sum())
    exact_requested[lost_ownership_exact] = False
    # Final, layer-local residual continuity. Frozen borrowing amounts explain
    # this seam; no flattened-reference pixels or mutable neighbour evidence.
    reuse_seam_changed = np.zeros(shape, bool)
    reuse_seam_amount = np.zeros(shape, np.float32)
    reuse_seam_layer_amounts = []
    reuse_seam_enabled = bool(cfg.reuse_boundary_continuity and
                              cfg.spatial_continuity and cfg.protect_layer_ownership)
    def map_reuse_seam_layer(i, layer):
        changed = np.zeros(layer.alpha.shape, bool)
        amount = np.zeros(layer.alpha.shape, np.float32)
        seed = np.zeros(layer.alpha.shape, bool)
        if reuse_seam_enabled and i > 0:
            locked = (protected_mask[layer.region] | alpha_repaired[layer.region] |
                      ~eligible[layer.region] | (shadow_guards[i] > 0) |
                      (surface_guards[i] > 0) | (contact_guards[i] > 0) |
                      (receiver_guards[i] > 0) | (ownership_layer_amounts[i] > 0))
            adjusted, changed, amount, seed = harmonize_reuse_boundaries(
                contour_guides[i], layer.rgb, layer.alpha, weights[i],
                reuse_amounts[i], eligible[layer.region], locked)
            layer.rgb = adjusted
        return changed, amount, seed

    for i, (changed, amount, seed) in enumerate(_map_layer_stage(map_reuse_seam_layer, layers)):
        layer = layers[i]
        reuse_seam_changed[layer.region] |= changed
        reuse_seam_amount[layer.region] = np.maximum(reuse_seam_amount[layer.region], amount)
        reuse_seam_layer_amounts.append(amount)
        stats[i]["reuse_boundary_changed_pixels"] = int(changed.sum())
        stats[i]["reuse_boundary_seed_pixels"] = int(seed.sum())
    final_snapshot = (composite(layers, shape) if reuse_seam_changed.any()
                      else ownership_snapshot)
    recolored, _ = final_snapshot
    lost_reuse_seam_exact = exact_requested & reuse_seam_changed & ~np.all(
        np.rint(recolored) == reference, axis=-1)
    for i in range(len(layers)):
        lost_count = int((lost_reuse_seam_exact & (exact_owner == i)).sum())
        stats[i]["exact_reuse_pixels"] -= lost_count
        stats[i]["reuse_boundary_invalidated_exact_pixels"] = lost_count
    exact_requested[lost_reuse_seam_exact] = False
    # Repair only legacy ink-clipped extensions of ALREADY accepted contours.
    # Run after all existing stages: no candidate RGB may feed their fitting,
    # filters, ownership decisions, matte solves, or reuse-boundary repair.
    chromatic_contour_changed = np.zeros(shape, bool)
    chromatic_contour_amount = np.zeros(shape, np.float32)
    chromatic_contour_layer_amounts = []
    def map_chromatic_contour_layer(i, layer):
        amount = np.zeros(layer.alpha.shape, np.float32)
        changed = np.zeros(layer.alpha.shape, bool)
        candidate = chromatic_contour_candidates[i]
        if chromatic_contour_enabled and i > 0 and candidate:
            s = layer.region
            solid = ndi.minimum_filter(layer.alpha, size=3, mode="constant", cval=0) == 255
            locked = (protected_mask[s] | alpha_repaired[s] | ~eligible[s] |
                      (shadow_guards[i] > 0) | (surface_guards[i] > 0) |
                      (contact_guards[i] > 0) | (receiver_guards[i] > 0) |
                      (ownership_layer_amounts[i] > 0) | (reuse_seam_layer_amounts[i] > 0) |
                      edge_decontaminated_map[s] | supported_hole_map[s])
            allowed = solid & (weights[i] >= 1-1e-7) & ~locked
            amount = candidate["amount"] * allowed * (1-frequency_edge_layer_weights[i])
            proposed = np.rint(np.clip(layer.rgb + amount[..., None] *
                               (candidate["target"]-layer.rgb), 0, 255)).astype(np.float32)
            changed = (amount > 0) & np.any(proposed != layer.rgb, axis=-1)
            layer.rgb[changed] = proposed[changed]
        return changed, amount
    for i, (changed, amount) in enumerate(_map_layer_stage(map_chromatic_contour_layer, layers)):
        layer = layers[i]
        chromatic_contour_layer_amounts.append(amount)
        chromatic_contour_changed[layer.region] |= changed
        chromatic_contour_amount[layer.region] = np.maximum(chromatic_contour_amount[layer.region], amount)
        stats[i]["chromatic_contour_changed_pixels"] = int(changed.sum())
    if chromatic_contour_changed.any():
        final_snapshot = composite(layers, shape)
    recolored, _ = final_snapshot
    lost_chromatic_exact = exact_requested & chromatic_contour_changed & ~np.all(
        np.rint(recolored) == reference, axis=-1)
    for i in range(len(layers)):
        lost_count = int((lost_chromatic_exact & (exact_owner == i)).sum())
        stats[i]["exact_reuse_pixels"] -= lost_count
        stats[i]["chromatic_contour_invalidated_exact_pixels"] = lost_count
    exact_requested[lost_chromatic_exact] = False
    for layer, guard, baseline in zip(layers, receiver_guards, receiver_baselines):
        assert np.array_equal(layer.rgb[guard >= 1-1e-7], baseline), "Receiver guard bypassed"
    for layer, guard, baseline in zip(layers, contact_guards, contact_baselines):
        assert np.array_equal(layer.rgb[guard >= 1-1e-7], baseline), "Contact guard bypassed"
    # Source-only shadow cores must still be the calibrated model surface,
    # never partly replaced by an interior solve, edge solve or projection.
    for layer, guard, baseline in zip(layers, shadow_guards, shadow_baselines):
        assert np.array_equal(layer.rgb[guard >= 1-1e-7], baseline), "Shadow guard bypassed"
    for layer, guard, baseline in zip(layers, surface_guards, surface_baselines):
        assert np.array_equal(layer.rgb[guard >= 1-1e-7], baseline), "Surface guard bypassed"
    assert not np.any(exact_requested & shadow_core_map), "Shadow core claimed exact"
    assert not np.any(exact_requested & (contact_guard_map >= 1-1e-7)), "Contact core claimed exact"
    after, final_alpha = _unchanged_composite(layers, shape, final_snapshot)
    before_u8 = np.rint(np.clip(before, 0, 255)).astype(np.uint8)
    after_u8 = np.rint(np.clip(after, 0, 255)).astype(np.uint8)
    equal = np.all(after_u8 == source_rgba[..., :3], axis=-1) & eligible
    verified = exact_requested & equal
    alpha_preserved = all(np.array_equal(a.alpha, b.alpha) for a, b in zip(original, layers))
    for i, (a, b) in enumerate(zip(original, layers)):
        changed = a.alpha != b.alpha
        assert int(changed.sum()) == stats[i]["interior_alpha_repaired_pixels"]
        assert np.all(a.alpha[changed] == 254) and np.all(b.alpha[changed] == 255)
        assert np.all(verified[a.region][changed])
    assert np.array_equal(output_alpha, final_alpha)
    assert np.all(equal[exact_requested]), "Export-grid exact-reuse contract failed"
    def metrics(rgb):
        diff = np.abs(rgb.astype(np.float32)-reference)[eligible]
        return {"mae": float(diff.mean()) if diff.size else None,
                "p95_max_channel_error": float(np.percentile(diff.max(axis=-1), 95)) if diff.size else None,
                "exact_rgb_pixels": int((np.all(rgb == source_rgba[..., :3], axis=-1) & eligible).sum())}
    regions = {}
    if background_index is not None:
        for name, mask in (("background", weights[0] >= .5), ("foreground", weights[0] < .5)):
            mask = mask & eligible
            regions[name] = {"evaluated_pixels": int(mask.sum()),
                             "exact_pixels": int((verified & mask).sum()),
                             "exact_fraction": float((verified & mask).sum()/max(1, mask.sum())),
                             "mae": float(np.abs(after_u8.astype(np.float32)-reference)[mask].mean()) if mask.any() else None}
    report = {"regions": regions, "algorithm_version": "6.4.5-chromatic-contour-continuity" if chromatic_contour_enabled else "6.4.4-reuse-boundary-residual" if reuse_seam_enabled else "6.4.3-flat-receiver-ownership" if (cfg.protect_contact_shadow_tails or cfg.frequency_soft_protection) else "6.4.1-occlusion-seam-residual" if cfg.frequency_calibrated_baseline and cfg.ownership_local_seams else "6.4.1-calibrated-contour-residual" if cfg.frequency_calibrated_baseline else "6.4-double-colour-calibration", "canvas": [w, h], "reference_opaque_pixels": int(valid_source.sum()),
              "evaluated_pixels": int(eligible.sum()),
              "before": metrics(before_u8), "after": metrics(after_u8),
              "verified_exact_reuse_pixels": int(verified.sum()),
              "reuse_boundary_continuity": {
                  "enabled": reuse_seam_enabled,
                  "changed_pixels": int(reuse_seam_changed.sum()),
                  "invalidated_exact_pixels": int(lost_reuse_seam_exact.sum()),
                  "method": "frozen borrowing transitions with unsupported residual jumps in quiet opaque same-layer components; residual-only normalized filtering with bounded feather; no alpha or protected/hidden/background edits"},
              "contact_shadow_protection": {"enabled": bool(cfg.protect_contact_shadow_tails),
                  "guard_pixels": int((contact_guard_map > 0).sum()),
                  "core_pixels": int((contact_guard_map >= 1-1e-7).sum()),
                  "method": "thick neutral dark tails connected to existing material conflicts on flat visible model surfaces"},
              "chromatic_contour_continuity": {"enabled": chromatic_contour_enabled, "changed_pixels": int(chromatic_contour_changed.sum()), "invalidated_exact_pixels": int(lost_chromatic_exact.sum()), "method": "final opaque continuation of legacy accepted contours; bright chromatic ink witnesses; no upstream or soft-alpha edits"},
              "model_contour_preservation": {"enabled": bool(cfg.preserve_model_edges), "changed_pixels": int(model_contour_changed.sum()), "method": "connected model edge profiles plus slow residual field; no alpha change"},
              "frequency_split_edge_gate": {
                  "enabled": bool(cfg.frequency_split_edges),
                  "gate_pixels": int((frequency_edge_gate > 1e-3).sum()),
                  "core_pixels": int(frequency_edge_parts["core"].sum()),
                  "changed_pixels": int(frequency_edge_changed.sum()),
                  "invalidated_exact_pixels": int(lost_frequency_exact.sum()),
                  "sigma": cfg.frequency_edge_sigma,
                  "edge_aware_residual": bool(cfg.frequency_edge_aware_residual),
                  "calibrated_baseline": bool(cfg.frequency_calibrated_baseline),
                  "soft_surface_protection": bool(cfg.frequency_soft_protection),
                  "residual_source": "first calibrated composite before pixel borrowing in solid foreground; legacy residual at soft alpha" if cfg.frequency_calibrated_baseline else "post-reuse composite (legacy)",
                  "residual_filter_radius": max(1, int(np.ceil(4.0 * cfg.frequency_edge_sigma))),
                  "residual_filter_epsilon": (8.0 / 255.0) ** 2,
                  "residual_halo_enabled": bool(cfg.frequency_residual_halo),
                  "residual_halo_filter_radius": max(1, int(np.ceil(16.0 * cfg.frequency_edge_sigma))),
                  "residual_halo_seed_pixels": int(frequency_halo_seed.sum()),
                  "residual_halo_pixels": int((frequency_halo_amount > 1e-3).sum()),
                  "model_guide": "MODEL_RAW" if model_guide is not None else "normalized model input",
                  "protection_scope": "canvas invariants global; shadow/material and flat-receiver ownership guards layer-local",
                  "method": "MODEL_RAW plus a low-frequency colour residual from the first calibrated composite (or legacy post-reuse residual when disabled); a model-flat residual-gradient detector expands repair beyond the narrow edge feather only inside fully opaque foreground material; displaced corrected contours are smoothed away without crossing model edges; one visible foreground owner; lower-layer guards do not punch holes through unrelated upper contours; invalid exact claims withdrawn; no alpha/background-policy change"},
              "flat_receiver_ownership": {"enabled": bool(cfg.protect_layer_ownership and cfg.ownership_flat_receivers),
                  "guard_pixels": int((receiver_guard_map > 0).sum()),
                  "method": "pre-borrow local quiet opaque receiver exclusion; receiver-only contour locks; no alpha change"},
              "layer_ownership_continuity": {
                  "enabled": bool(cfg.protect_layer_ownership),
                  "local_seams_enabled": bool(cfg.ownership_local_seams),
                  "local_max_receiver_texture_p75": cfg.ownership_local_max_texture,
                  "local_seam_layers": sum(x.get("ownership_local_seam_repair", False) for x in stats),
                  "guard_pixels": int((ownership_amount > 0).sum()),
                  "changed_pixels": int(ownership_changed.sum()),
                  "invalidated_exact_pixels": int(ownership_invalidated_exact.sum()),
                  "radius": ownership_radius,
                  "min_occlusion_fraction": cfg.ownership_min_occlusion,
                  "max_receiver_texture_p75": cfg.ownership_max_texture,
                  "alpha_modified": False,
                  "method": "low-texture receiver whole-layer second fit; local opaque occlusion-seam residual fallback for textured/small occluders; model detail preserved; protected and remote pixels immutable"},
              "contour_continuity": {"enabled": bool(cfg.regularize_contour_colors), "changed_pixels": int(contour_changed.sum()), "alpha_modified": False, "method": "model-guided stable-neighbor correction residual regularization after exact projection"},
              "changed_surface_protection": {"enabled": bool(cfg.protect_changed_surfaces), "guard_pixels": int((surface_guard_map > 0).sum()), "method": "broad chromatic conflict hysteresis; local same-material residual interpolation; feathered borrowing exclusion", "background_policy_changed": False},
              "removed_shadow_protection": {
                  "enabled": bool(cfg.protect_removed_shadows),
                  "core_pixels": int(shadow_core_map.sum()),
                  "calibrated_model_core_preserved": True,
                  "guard_pixels": int((shadow_guard_map > 0).sum()),
                  "method": "receiver dark-residual hysteresis with higher-layer contact; refit without shadows; feathered exclusion",
                  "background_policy_changed": False},
              "spatial_continuity": {
                  "enabled": bool(cfg.spatial_continuity),
                  "supported_confidence_hole_pixels": sum(x.get("supported_confidence_hole_pixels", 0) for x in stats),
                  "edge_alternative_anchor_pixels": sum(x.get("edge_alternative_anchor_pixels", 0) for x in stats),
                  "edge_rim_fragment_suppressed_pixels": sum(x.get("edge_rim_fragment_suppressed_pixels", 0) for x in stats),
                  "background_policy_changed": False},
              "conflict_pixels": int((bad & eligible).sum()),
              "user_protected_pixels": int(protected_mask.sum()),
              "edge_fixed_stack_unreachable_pixels": int(gamut_conflicts.sum()),
              "edge_exact_verified_pixels": int((edge_refined & verified).sum()),
              "alpha_preserved": alpha_preserved,
              "alpha_preserved_relative_to": "normalized_aligned_input",
              "alpha_levels": {"enabled": bool(cfg.alpha_levels), "black": cfg.alpha_black, "white": cfg.alpha_white,
                               "method": "linear_levels", "late_repair_enabled": bool(cfg.repair_opaque_interiors),
                               "changed_native_pixels": sum(l.metadata.get("alpha_normalization", {}).get("changed_pixels", 0) for l in layers)},
              "interior_alpha_repaired_pixels": int(alpha_repaired.sum()),
              "composite_alpha_preserved": True,
              "contour_alpha_preserved": True,
              "source_alpha_mismatch_pixels": int((np.rint(final_alpha*255).astype(np.uint8) != source_rgba[..., 3]).sum()),
              "layers": stats,
              "limitations": ["Unchanged-region classification is heuristic, not a semantic guarantee.",
                              "Changed-material protection needs broad chromatic evidence; neutral conflicts and narrow edits can be missed. Hidden surface color is inferred, not ground truth.",
                              "Exact RGB is guaranteed only inside exact_reuse_mask.png at this canvas grid and compositor.",
                              "Occluded/unseen colors are inferred or left unchanged, not recovered ground truth.",
                              "Flat receiver ownership protection is local model-gradient/coverage evidence, not semantic text recognition; strongly textured or weak/unsupported occlusions can remain on the legacy path.",
                              "Removed-shadow detection is heuristic; a higher-layer caster and broad residual evidence are required. Material-conflict contact tails also have a separate conservative guard. Unanchored single-layer/self shadows or weak shadows can still be missed; protect_mask remains available.",
                              "Order and placement are preserved. Alpha levels modifies original coverage; alpha preservation refers to normalized/aligned input, not raw assets.",
                              "Edge gamut conflicts hold other layers fixed; they do not alone prove wrong geometry.",
                              "Frequency separation preserves model-grid high frequencies; it does not geometrically register the reference or repair errors already present in MODEL_RAW."]}
    diagnostics = {"before": before_u8, "after": after_u8, "alpha": final_alpha,
                   "before_premultiplied": before, "after_premultiplied": after,
                   "exact": verified, "conflicts": bad & eligible, "trust": trust,
                   "supports": support_maps, "background_protection": bg_excluded,
                   "edge_gamut_conflicts": gamut_conflicts, "edge_exact": edge_refined & verified,
                   "interior_alpha_repaired": alpha_repaired,
                   "contour_regularized": contour_changed,
                   "model_contour_changed": model_contour_changed,
                   "model_contour_layer_weights": model_contour_weights,
                   "frequency_edge_gate": frequency_edge_gate,
                   "frequency_edge_changed": frequency_edge_changed,
                   "frequency_edge_model": frequency_edge_parts["model_edge"],
                   "frequency_edge_reference": frequency_edge_parts["reference_edge"],
                   "frequency_edge_alpha": frequency_edge_parts["alpha_edge"],
                   "frequency_edge_displaced_pair": frequency_edge_parts["displaced_pair"],
                   "frequency_edge_core": frequency_edge_parts["core"],
                   "frequency_halo_amount": frequency_halo_amount,
                   "frequency_halo_seed": frequency_halo_seed,
                   "frequency_edge_layer_weights": frequency_edge_layer_weights,
                   "chromatic_contour_changed": chromatic_contour_changed,
                   "chromatic_contour_amount": chromatic_contour_amount,
                   "chromatic_contour_layer_amounts": chromatic_contour_layer_amounts,
                   "reuse_boundary_changed": reuse_seam_changed,
                   "reuse_boundary_amount": reuse_seam_amount,
                   "reuse_boundary_layer_amounts": reuse_seam_layer_amounts,
                   "layer_ownership_changed": ownership_changed,
                   "layer_ownership_amount": ownership_amount,
                   "layer_ownership_layer_amounts": ownership_layer_amounts,
                   "changed_surface_guard": surface_guard_map,
                   "changed_surface_layer_guards": surface_guards,
                   "removed_shadow_core": shadow_core_map,
                   "removed_shadow_guard": shadow_guard_map,
                   "removed_shadow_layer_guards": shadow_guards,
                   "flat_receiver_guard": receiver_guard_map,
                   "flat_receiver_layer_guards": receiver_guards,
                   "contact_shadow_guard": contact_guard_map,
                   "contact_shadow_layer_guards": contact_guards}
    return layers, report, diagnostics
