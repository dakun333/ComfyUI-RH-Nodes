"""Original-grid edit localization and restricted harmonization, CPU only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import cv2
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

from .precision import quantize8


@dataclass
class Config:
    high: float = 2.7
    low: float = 1.55
    min_area: int = 5
    padding: int = 1
    feather: int = 2
    graph_smooth: float = 0.65
    screen: float = 0.08
    blend: str = "feather"
    flow_limit: float = 5.0
    seed: int = 20260923
    solver_precision: str = "float32"


def gray(array):
    return cv2.cvtColor(array.astype(np.float32), cv2.COLOR_RGB2GRAY)


def blur(array, sigma):
    return cv2.GaussianBlur(
        array, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101
    )


def rms(array):
    return np.sqrt(np.mean(array * array, axis=2))


def resize_like(array, reference):
    if array.shape[:2] == reference.shape[:2]:
        return array.copy()
    interpolation = (
        cv2.INTER_AREA
        if array.shape[0] > reference.shape[0]
        else cv2.INTER_LANCZOS4
    )
    return cv2.resize(
        array,
        (reference.shape[1], reference.shape[0]),
        interpolation=interpolation,
    )


def _registration_loss(first, second):
    first = gray(first)
    second = gray(second)
    first_residual = first - blur(first, 3)
    second_residual = second - blur(second, 3)
    error = np.abs(first_residual - second_residual)
    limit = np.quantile(error, 0.8)
    return float(np.mean(error[error <= limit]))


def global_align(original, edited):
    height, width = original.shape[:2]
    source = resize_like(edited.astype(np.float32), original)
    scale = min(1.0, 850 / max(height, width))
    size = (round(width * scale), round(height * scale))
    original_gray = cv2.resize(gray(original), size).astype(np.uint8)
    source_gray = cv2.resize(gray(source), size).astype(np.uint8)
    initial = np.eye(2, 3, dtype=np.float32)
    candidates = [("resize", initial.copy())]
    match_count = 0
    try:
        sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.025)
        original_keys, original_desc = sift.detectAndCompute(original_gray, None)
        source_keys, source_desc = sift.detectAndCompute(source_gray, None)
        if (
            original_desc is not None
            and source_desc is not None
            and len(source_desc) > 1
        ):
            pairs = cv2.BFMatcher().knnMatch(original_desc, source_desc, k=2)
            good = [match for match, other in pairs if match.distance < 0.72 * other.distance]
            match_count = len(good)
            if len(good) >= 12:
                original_points = np.float32(
                    [original_keys[match.queryIdx].pt for match in good]
                )
                source_points = np.float32(
                    [source_keys[match.trainIdx].pt for match in good]
                )
                matrix, inliers = cv2.estimateAffine2D(
                    original_points,
                    source_points,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=1.5,
                    maxIters=2000,
                )
                if matrix is not None and int(inliers.sum()) >= 10:
                    initial = matrix.astype(np.float32)
                    candidates.append(("sift", initial.copy()))
        _, ecc = cv2.findTransformECC(
            blur(original_gray.astype(np.float32), 1),
            blur(source_gray.astype(np.float32), 1),
            initial.copy(),
            cv2.MOTION_AFFINE,
            (
                cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                60,
                1e-5,
            ),
            None,
            5,
        )
        candidates.append(("ecc", ecc))
    except cv2.error:
        pass

    best = source
    best_loss = _registration_loss(original, source)
    chosen = np.eye(2, 3, dtype=np.float32)
    method = "resize"
    for name, matrix in candidates[1:]:
        matrix = matrix.copy()
        matrix[:, 2] /= scale
        singular_values = np.linalg.svd(matrix[:, :2], compute_uv=False)
        if (
            np.max(np.abs(singular_values - 1)) > 0.035
            or np.max(np.abs(matrix[:, 2])) > max(height, width) * 0.04
        ):
            continue
        warped = cv2.warpAffine(
            source,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT101,
        )
        loss = _registration_loss(original, warped)
        if loss < best_loss * 0.998:
            best, best_loss, chosen, method = warped, loss, matrix, name
    valid = (
        cv2.warpAffine(
            np.ones((height, width), np.uint8),
            chosen,
            (width, height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
        > 0
    )
    return best, valid, {
        "method": method,
        "matrix": chosen.tolist(),
        "feature_matches": match_count,
        "trimmed_detail_error": best_loss,
    }


def robust_color(original, generated, trusted=None, seed=20260923):
    """Fit per-channel affine color and a broad robust illumination field."""
    original = original.astype(np.float32)
    generated = generated.astype(np.float32)
    height, width = original.shape[:2]
    rng = np.random.default_rng(seed)
    valid = (
        np.ones((height, width), bool)
        if trusted is None
        else trusted.astype(bool)
    )
    indices = np.flatnonzero(valid)
    if len(indices) < 32:
        return generated.copy(), {
            "gain": [1.0] * 3,
            "bias": [0.0] * 3,
            "fit_pixels": int(len(indices)),
        }
    if len(indices) > 65000:
        indices = rng.choice(indices, 65000, replace=False)
    source = generated.reshape(-1, 3)[indices].astype(np.float64)
    target = original.reshape(-1, 3)[indices].astype(np.float64)
    gain = np.ones(3)
    bias = np.median(target - source, axis=0)
    bins = np.clip((source / 64).astype(np.int32), 0, 3)
    bins = bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2]
    counts = np.bincount(bins, minlength=64)
    balance = 1 / np.sqrt(np.maximum(counts[bins], 1))
    balance /= balance.mean()
    unsaturated = (source > 3) & (source < 252) & (target > 3) & (target < 252)
    weights = np.ones(len(source))
    for _ in range(9):
        for channel in range(3):
            design = np.column_stack(
                [source[:, channel] - 128, np.ones(len(source))]
            )
            penalty = np.diag([25000.0, 0.1])
            channel_weights = weights * balance * unsaturated[:, channel]
            if np.count_nonzero(channel_weights) < 32:
                gain[channel] = 1
                bias[channel] = np.median(
                    target[:, channel] - source[:, channel]
                )
                continue
            right = (
                design.T @ (channel_weights * target[:, channel])
                + penalty @ np.array([1.0, 128.0])
            )
            coefficients = np.linalg.solve(
                design.T @ (channel_weights[:, None] * design) + penalty,
                right,
            )
            gain[channel] = np.clip(coefficients[0], 0.65, 1.5)
            bias[channel] = coefficients[1] - gain[channel] * 128
        difference = (
            target - np.clip(source * gain + bias, 0, 255)
        ) ** 2
        error = np.sqrt(
            np.sum(difference * unsaturated, axis=1)
            / np.maximum(unsaturated.sum(axis=1), 1)
        )
        eligible = unsaturated.any(axis=1)
        scale = (
            max(1.0, float(np.quantile(error[eligible], 0.45)) * 1.8)
            if eligible.any()
            else 1.0
        )
        weights = (1 - np.minimum(error / (scale * 2.3), 1) ** 2) ** 2 + 0.002
    base = np.clip(
        generated * gain.astype(np.float32) + bias.astype(np.float32), 0, 255
    )
    endpoints = []
    for channel in range(3):
        high_samples = source[:, channel] >= 253
        low_samples = source[:, channel] <= 2
        high = (
            float(np.median(target[high_samples, channel]))
            if high_samples.sum() >= 32
            else 255.0
        )
        low = (
            float(np.median(target[low_samples, channel]))
            if low_samples.sum() >= 32
            else 0.0
        )
        high_weight = (
            np.clip((generated[:, :, channel] - 250) / 5, 0, 1)
            if high_samples.sum() >= 32
            else np.zeros((height, width), np.float32)
        )
        low_weight = (
            np.clip((4 - generated[:, :, channel]) / 4, 0, 1)
            if low_samples.sum() >= 32
            else np.zeros((height, width), np.float32)
        )
        base[:, :, channel] = (
            base[:, :, channel] * (1 - high_weight - low_weight)
            + high * high_weight
            + low * low_weight
        )
        endpoints.append([low, high])
    residual = original - base
    field = np.zeros_like(base)
    sigma = max(20.0, min(height, width) / 17.0)
    for _ in range(4):
        error = rms(residual - field)
        values = error[valid]
        cutoff = max(2.0, float(np.quantile(values, 0.50)) * 2.6)
        weight = np.maximum(0, 1 - (error / cutoff) ** 2) ** 2 * valid
        denominator = blur(weight.astype(np.float32), sigma)
        field = blur(residual * weight[:, :, None], sigma) / np.maximum(
            denominator[:, :, None], 1e-5
        )
        field = np.clip(field, -28, 28)
    result = np.clip(base + field, 0, 255)
    return result.astype(np.float32), {
        "gain": gain.tolist(),
        "bias": bias.tolist(),
        "fit_pixels": int(len(indices)),
        "field_abs_mean": float(np.mean(np.abs(field))),
        "saturation_endpoints": endpoints,
    }


def evidence(original, corrected):
    original = original.astype(np.float32)
    corrected = corrected.astype(np.float32)
    difference = original - corrected
    color = rms(blur(difference, 0.65))
    high_frequency = rms(blur(difference, 0.6) - blur(difference, 3.0))
    texture = blur(rms(original - blur(original, 1.5)), 1.3)
    color_noise = max(0.85, float(np.quantile(color, 0.35)) * 1.65)
    detail_noise = max(
        0.6, float(np.quantile(high_frequency, 0.35)) * 1.7
    )
    color_score = color / (color_noise + 0.24 * texture)
    detail_score = high_frequency / (detail_noise + 0.22 * texture)
    score = np.maximum(color_score, detail_score * 0.85)
    return score.astype(np.float32), {
        "noise_colour": color_noise,
        "noise_detail": detail_noise,
        "residual_median": float(np.median(color)),
        "residual_p90": float(np.quantile(color, 0.9)),
    }


def keep_components(mask, min_area, seeds=None):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    if seeds is not None:
        touched = np.zeros(count, bool)
        touched[np.unique(labels[seeds])] = True
        touched[0] = False
        keep &= touched
    return keep[labels]


def mask_a(score, config):
    seeds = score >= config.high
    possible = score >= config.low
    mask = keep_components(possible, config.min_area, seeds)
    return cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)


def mask_b(score, config):
    try:
        import maxflow
    except ImportError as exc:
        raise RuntimeError(
            "Method B requires PyMaxflow 1.3.2; install this repository's requirements."
        ) from exc
    logit = np.clip((score - (config.low + 0.3)) * 2.2, -10, 12)
    probability = 1 / (1 + np.exp(-logit))
    unchanged_cost = -np.log(np.maximum(1 - probability, 1e-6)).astype(np.float32)
    edited_cost = -np.log(np.maximum(probability, 1e-6)).astype(np.float32)
    graph = maxflow.Graph[float]()
    nodes = graph.add_grid_nodes(score.shape)
    graph.add_grid_tedges(nodes, edited_cost, unchanged_cost)
    horizontal = config.graph_smooth * np.exp(
        -np.abs(score[:, 1:] - score[:, :-1]) / 1.5
    )
    vertical = config.graph_smooth * np.exp(
        -np.abs(score[1:, :] - score[:-1, :]) / 1.5
    )
    weights = np.zeros_like(score)
    weights[:, :-1] = horizontal
    graph.add_grid_edges(
        nodes,
        weights,
        structure=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
        symmetric=True,
    )
    weights = np.zeros_like(score)
    weights[:-1, :] = vertical
    graph.add_grid_edges(
        nodes,
        weights,
        structure=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        symmetric=True,
    )
    graph.maxflow()
    mask = graph.get_grid_segments(nodes)
    mask = keep_components(mask, config.min_area, score >= config.high)
    mask = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)
    return mask, probability.astype(np.float32)


def local_align(original, globally_aligned, config):
    height, width = original.shape[:2]
    scale = min(1.0, 640 / max(height, width))
    size = (round(width * scale), round(height * scale))
    original_gray = cv2.resize(gray(original), size)
    generated_gray = cv2.resize(gray(globally_aligned), size)

    def normalized(array):
        high_frequency = array - blur(array, 9)
        return np.clip(high_frequency * 2 + 128, 0, 255).astype(np.uint8)

    original_gray = normalized(original_gray)
    generated_gray = normalized(generated_gray)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    forward = dis.calc(original_gray, generated_gray, None)
    backward = dis.calc(generated_gray, original_gray, None)
    y, x = np.mgrid[: size[1], : size[0]].astype(np.float32)
    reverse = cv2.remap(
        backward,
        x + forward[:, :, 0],
        y + forward[:, :, 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    cycle = np.linalg.norm(forward + reverse, axis=2)
    trust = np.exp(-((cycle / 0.65) ** 2))
    smooth = blur(forward * trust[:, :, None], 5) / np.maximum(
        blur(trust, 5)[:, :, None], 0.01
    )
    flow = cv2.resize(smooth, (width, height)) / scale
    magnitude = np.linalg.norm(flow, axis=2)
    flow *= np.minimum(
        1, config.flow_limit / np.maximum(magnitude, 1e-6)
    )[:, :, None]
    y, x = np.mgrid[:height, :width].astype(np.float32)
    warped = cv2.remap(
        globally_aligned,
        x + flow[:, :, 0],
        y + flow[:, :, 1],
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )
    before = _registration_loss(original, globally_aligned)
    after = _registration_loss(original, warped)
    accepted = after < before * 0.99
    if not accepted:
        warped = globally_aligned.copy()
        flow[:] = 0
    valid = (
        (x + flow[:, :, 0] >= 0)
        & (x + flow[:, :, 0] <= width - 1)
        & (y + flow[:, :, 1] >= 0)
        & (y + flow[:, :, 1] <= height - 1)
    )
    return warped, valid, {
        "accepted": accepted,
        "before": before,
        "after_candidate": after,
        "flow_p95": float(np.quantile(np.linalg.norm(flow, axis=2), 0.95)),
    }, flow


def support_alpha(core, config):
    if not np.any(core):
        return np.zeros(core.shape, np.float32), core.copy()
    expanded = (
        cv2.dilate(
            core.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (config.padding * 2 + 1,) * 2
            ),
        )
        > 0
        if config.padding
        else core.copy()
    )
    distance = cv2.distanceTransform(
        (~expanded).astype(np.uint8), cv2.DIST_L2, 5
    )
    alpha = np.clip(1 - distance / max(1, config.feather), 0, 1)
    alpha[expanded] = 1
    return alpha.astype(np.float32), alpha > 0


def feather_compose(original, generated, alpha, float_output=False):
    output = (
        original.astype(np.float32).copy() if float_output else original.copy()
    )
    active = alpha > 0
    mixed = (
        original.astype(np.float32) * (1 - alpha[:, :, None])
        + generated * alpha[:, :, None]
    )
    output[active] = (
        np.clip(mixed[active], 0, 255)
        if float_output
        else np.clip(np.rint(mixed[active]), 0, 255).astype(np.uint8)
    )
    return output


def screened_poisson(
    original,
    generated,
    support,
    screen=0.08,
    float_output=False,
    solver_precision="float32",
):
    if solver_precision not in ("float32", "float64"):
        raise ValueError("Invalid solver precision")
    dtype = np.float64 if solver_precision == "float64" else np.float32
    output = (
        original.astype(np.float32).copy() if float_output else original.copy()
    )
    unknowns = int(support.sum())
    if unknowns == 0:
        return output, {
            "unknowns": 0,
            "converged": True,
            "relative_residual": 0.0,
        }
    height, width = support.shape
    index = np.full((height, width), -1, np.int32)
    index[support] = np.arange(unknowns, dtype=np.int32)
    rows, columns, values = [], [], []
    degree = np.zeros(unknowns, dtype)
    right = np.zeros((unknowns, 3), dtype)
    boundary = original.astype(dtype) - generated.astype(dtype)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        current = index[y0:y1, x0:x1]
        neighbor = index[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        active = current >= 0
        current_indices = current[active]
        neighbor_indices = neighbor[active]
        degree[current_indices] += 1
        inner = neighbor_indices >= 0
        rows.append(current_indices[inner])
        columns.append(neighbor_indices[inner])
        values.append(-np.ones(int(inner.sum()), dtype))
        edge = ~inner
        boundary_values = boundary[
            y0 + dy : y1 + dy, x0 + dx : x1 + dx
        ][active]
        right[current_indices[edge]] += boundary_values[edge]
    diagonal = degree + screen
    rows.append(np.arange(unknowns))
    columns.append(np.arange(unknowns))
    values.append(diagonal)
    matrix = sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(unknowns, unknowns),
    )
    preconditioner = sparse.diags(1 / diagonal)
    correction = np.zeros((unknowns, 3), dtype)
    statuses, residuals = [], []
    for channel in range(3):
        correction[:, channel], status = cg(
            matrix,
            right[:, channel],
            rtol=1e-10 if solver_precision == "float64" else 2e-5,
            atol=1e-9 if solver_precision == "float64" else 1e-4,
            maxiter=2500 if solver_precision == "float64" else 500,
            M=preconditioner,
        )
        statuses.append(int(status))
        residuals.append(
            float(
                np.linalg.norm(
                    matrix @ correction[:, channel] - right[:, channel]
                )
                / max(float(np.linalg.norm(right[:, channel])), 1e-6)
            )
        )
    solved = generated[support] + correction
    output[support] = (
        np.clip(solved, 0, 255)
        if float_output
        else np.clip(np.rint(solved), 0, 255).astype(np.uint8)
    )
    return output, {
        "precision": solver_precision,
        "unknowns": unknowns,
        "converged": all(status == 0 for status in statuses),
        "cg_status": statuses,
        "relative_residual": max(residuals),
    }


def adaptive_compose(
    original,
    generated,
    core,
    valid,
    config,
    alpha,
    support,
    float_output=False,
):
    count, labels, areas, _ = cv2.connectedComponentsWithStats(
        core.astype(np.uint8), 8
    )
    distance = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    radii = ndimage.maximum(distance, labels, np.arange(count))
    selected = (
        areas[:, cv2.CC_STAT_AREA] >= max(256, core.size * 0.02)
    ) & (radii >= 12)
    selected[0] = False
    large = selected[labels]
    output = feather_compose(
        original, generated, alpha, float_output=float_output
    )
    if not large.any():
        return output, alpha, support, {
            "converged": True,
            "adaptive_components": 0,
            "mode": "legacy_feather",
        }
    radius = max(config.padding, 8)
    expanded = (
        cv2.dilate(
            large.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2
            ),
        )
        > 0
    )
    distance = cv2.distanceTransform(
        (~expanded).astype(np.uint8), cv2.DIST_L2, 5
    )
    region = (
        (distance < max(1, config.feather)) | expanded
    ) & valid
    matched, solver = screened_poisson(
        original,
        generated,
        region,
        screen=0.002,
        float_output=float_output,
        solver_precision=config.solver_precision,
    )
    if solver["converged"]:
        output[region] = matched[region]
        alpha = alpha.copy()
        alpha[region] = 1
        support = support | region
    solver.update(
        adaptive_components=int(selected.sum()),
        mode="boundary_harmonization",
        effective_padding=radius,
        screen=0.002,
        applied=bool(solver["converged"]),
    )
    return output, alpha, support, solver


def seam_metric(output, generated, support):
    errors = []
    for axis in (0, 1):
        cut = np.diff(support.astype(np.int8), axis=axis) != 0
        jump = np.diff(output.astype(np.float32), axis=axis) - np.diff(
            generated.astype(np.float32), axis=axis
        )
        if cut.any():
            errors.append(rms(jump)[cut])
    return float(np.mean(np.concatenate(errors))) if errors else 0.0


def process_pair(
    original,
    edited,
    method="A",
    config=None,
    manual_mask=None,
    capture=False,
):
    config = config or Config()
    started = time.perf_counter()
    if method not in ("A", "B"):
        raise ValueError("method must be A or B")
    if config.blend not in ("feather", "poisson", "adaptive"):
        raise ValueError("blend must be feather, poisson or adaptive")
    if (
        config.padding < 0
        or config.feather < 0
        or config.flow_limit < 0
        or config.screen <= 0
        or config.min_area < 1
        or not 0 < config.low <= config.high
    ):
        raise ValueError(
            "Invalid threshold, margin, flow, component area, or screening configuration"
        )
    if config.solver_precision not in ("float32", "float64"):
        raise ValueError("Invalid solver precision")
    cv2.setRNGSeed(config.seed)
    original = np.asarray(original, dtype=np.float32)
    edited = np.asarray(edited, dtype=np.float32)
    if (
        not np.isfinite(original).all()
        or not np.isfinite(edited).all()
        or original.size == 0
        or edited.size == 0
    ):
        raise ValueError("Images must be nonempty and finite")
    if min(original.min(), edited.min()) < 0 or max(
        original.max(), edited.max()
    ) > 255:
        raise ValueError("SDR RGB arrays must be in 0..255")
    if (
        original.ndim != 3
        or original.shape[2] != 3
        or edited.ndim != 3
        or edited.shape[2] != 3
    ):
        raise ValueError("Inputs must be H x W x 3 RGB images")
    if min(*original.shape[:2], *edited.shape[:2]) < 16:
        raise ValueError("Images must be at least 16 pixels on each side")

    generated, valid, registration = global_align(original, edited)
    trace = {"global_aligned": generated.copy()} if capture else None
    flow = None
    flow_stats = None
    if method == "B":
        generated, valid_local, flow_stats, flow = local_align(
            original, generated, config
        )
        y, x = np.mgrid[: valid.shape[0], : valid.shape[1]].astype(np.float32)
        valid = (
            cv2.remap(
                valid.astype(np.uint8),
                x + flow[:, :, 0],
                y + flow[:, :, 1],
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            > 0
        )
        valid &= valid_local
    corrected, color_stats = robust_color(
        original, generated, valid, config.seed
    )
    score, noise = evidence(original, corrected)
    if method == "A":
        core = mask_a(score, config)
        probability = 1 / (
            1 + np.exp(-np.clip((score - (config.low + 0.3)) * 2.2, -12, 12))
        )
    else:
        core, probability = mask_b(score, config)
    if capture:
        trace.update(
            first_corrected=corrected.copy(),
            first_core=core.copy(),
            first_score=score.copy(),
        )
    excluded = cv2.dilate(
        core.astype(np.uint8), np.ones((9, 9), np.uint8)
    ) > 0
    if capture:
        trace["trusted"] = valid & ~excluded
    if (valid & ~excluded).sum() >= max(
        1000, int(original.shape[0] * original.shape[1] * 0.1)
    ):
        corrected, color_stats = robust_color(
            original, generated, valid & ~excluded, config.seed
        )
        score, noise = evidence(original, corrected)
        if method == "A":
            core = mask_a(score, config)
            probability = 1 / (
                1
                + np.exp(
                    -np.clip((score - (config.low + 0.3)) * 2.2, -12, 12)
                )
            )
        else:
            core, probability = mask_b(score, config)
    if manual_mask is not None:
        if manual_mask.shape != original.shape[:2]:
            raise ValueError("manual mask must match original dimensions")
        core = manual_mask > 0
        excluded = cv2.dilate(
            core.astype(np.uint8), np.ones((9, 9), np.uint8)
        ) > 0
        corrected, color_stats = robust_color(
            original, generated, valid & ~excluded, config.seed
        )
        if capture:
            trace["trusted"] = valid & ~excluded
    core &= valid
    alpha, support = support_alpha(core, config)
    support &= valid
    alpha[~valid] = 0
    corrected = np.clip(corrected, 0, 255)
    if method == "A" or config.blend == "feather":
        output_float = feather_compose(
            original, corrected, alpha, float_output=True
        )
        solver = None
    elif config.blend == "adaptive" and manual_mask is None:
        output_float, alpha, support, solver = adaptive_compose(
            original,
            corrected,
            core,
            valid,
            config,
            alpha,
            support,
            float_output=True,
        )
    else:
        output_float, solver = screened_poisson(
            original,
            corrected,
            support,
            0.002 if config.blend == "adaptive" else config.screen,
            float_output=True,
            solver_precision=config.solver_precision,
        )
    output = quantize8(output_float)
    exact = np.all(output == quantize8(original), axis=2)
    exact_float = np.all(output_float == original, axis=2)
    outside = ~support
    outside_max = (
        float(np.max(np.abs(output_float[outside] - original[outside])))
        if outside.any()
        else 0.0
    )
    assert outside_max == 0, "Invariant failed: pixels outside support were changed"
    warnings = []
    if core.mean() > 0.35:
        warnings.append(
            "Large edit candidate: review masks for color/registration failure."
        )
    if (~valid).mean() > 0.001:
        warnings.append(
            "Some borders lack valid generated samples and were preserved from original."
        )
    if solver and not solver["converged"]:
        warnings.append("Poisson solver did not converge; inspect result.")
    stats = {
        "pipeline_version": "2.0-precision",
        "output_precision": "float32",
        "exact_original_float_fraction": float(exact_float.mean()),
        "preview_quantization": "8bit_round",
        "blend": "feather" if method == "A" else config.blend,
        "warnings": warnings,
        "valid_generated_fraction": float(valid.mean()),
        "method": method,
        "seconds": time.perf_counter() - started,
        "size": [original.shape[1], original.shape[0]],
        "core_fraction": float(core.mean()),
        "support_fraction": float(support.mean()),
        "exact_original_fraction": float(exact.mean()),
        "outside_max_channel_error": outside_max,
        "boundary_gradient_mismatch": seam_metric(
            output_float, corrected, support
        ),
        "registration": registration,
        "local_alignment": flow_stats,
        "color_model": color_stats,
        "evidence": noise,
        "solver": solver,
        "config": asdict(config),
        "manual_mask": manual_mask is not None,
    }
    return {
        "output_float": output_float,
        "output": output,
        "aligned": generated,
        "corrected": corrected,
        "core": core,
        "support": support,
        "alpha": alpha,
        "score": score,
        "probability": probability,
        "stats": stats,
        "flow": flow,
        "valid": valid,
        "trace": trace,
    }
